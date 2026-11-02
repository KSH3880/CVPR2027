"""First-departure metrics, one trial per physical joint PUSH box."""
import csv
import json
import math
from pathlib import Path
import numpy as np


def measure(near, xy, velocity, alive, dt, speed_threshold=.05, sustained_seconds=.2):
    """Includes reset sample at index 0. Censor at first member fall."""
    assert len(near) == len(xy) == len(velocity) == len(alive)
    valid = alive.all(axis=-1)
    invalid = np.flatnonzero(~valid)
    end = int(invalid[0]) if len(invalid) else len(valid)
    near, xy, velocity = near[:end], xy[:end], velocity[:end]
    if end == 0:
        return dict(departed=False, ever_ready=False, censored_by_fall=True,
                    premature=False, unready_at_departure=False, ready_fraction=None,
                    departure_seconds=None, ready_seconds=None, ready_to_departure_seconds=None,
                    pre_ready_path_m=0., pre_ready_displacement_m=0.)
    ready_fraction = near.all(axis=-1).mean(axis=-1)
    ready = ready_fraction == 1
    hits = np.flatnonzero(ready)
    first_ready = int(hits[0]) if len(hits) else None
    # Index 0 is reset, not a completed control interval. Six samples at 30Hz
    # confirm 0.2 seconds; the onset is timestamped at the first sample.
    moving = np.linalg.norm(velocity, axis=-1) > speed_threshold
    width = max(1, int(math.ceil(sustained_seconds/dt - 1e-6)))
    departure = None
    for i in range(1, len(moving)-width+1):
        if moving[i:i+width].all():
            departure = i
            break
    stop = first_ready if first_ready is not None else len(xy)-1
    path = float(np.linalg.norm(np.diff(xy[:stop+1], axis=0), axis=-1).sum())
    displacement = float(np.linalg.norm(xy[stop]-xy[0]))
    return dict(departed=departure is not None, ever_ready=first_ready is not None,
        censored_by_fall=bool(len(invalid)),
        premature=bool(departure is not None and (first_ready is None or departure < first_ready)),
        unready_at_departure=bool(departure is not None and not ready[departure]),
        ready_fraction=None if departure is None else float(ready_fraction[departure]),
        departure_seconds=None if departure is None else departure*dt,
        ready_seconds=None if first_ready is None else first_ready*dt,
        ready_to_departure_seconds=None if departure is None or first_ready is None else (departure-first_ready)*dt,
        pre_ready_path_m=path, pre_ready_displacement_m=displacement)


def summarize(rows):
    def ratio(key, subset=rows):
        return sum(bool(r[key]) for r in subset)/len(subset) if subset else None
    started = [r for r in rows if r['departed']]
    def mean(key, subset=rows):
        values = [r[key] for r in subset if r[key] is not None]
        return float(np.mean(values)) if values else None
    return dict(groups=len(rows),departed=len(started),premature_all=ratio('premature'),
        premature_among_departed=ratio('premature',started),
        unready_at_departure=ratio('unready_at_departure',started),
        never_departed=1-ratio('departed') if rows else None,
        never_ready=1-ratio('ever_ready') if rows else None,
        censored_by_fall=ratio('censored_by_fall'),
        ready_fraction_at_departure=mean('ready_fraction',started),
        pre_ready_path_m=mean('pre_ready_path_m'),pre_ready_displacement_m=mean('pre_ready_displacement_m'),
        ready_then_departure_delay_s=mean('ready_to_departure_seconds',
            [r for r in started if r['ever_ready'] and not r['premature']]))


def process(out):
    rows=[]
    for folder in sorted(out.glob('general_seed*')):
        a=np.load(folder/'departure.npz')
        trace=np.load(folder/'trace.npz')
        fall=trace['fall_step']
        assert len(a['near']) == len(trace['valid'])+1
        dt=float(a['dt']); steps=np.arange(len(a['near']))[:,None,None]
        alive=(fall[None]<0)|(steps<fall[None])
        assert np.isfinite(a['xy']).all() and np.isfinite(a['velocity']).all()
        for env in range(a['groups'].shape[0]):
            for group in np.unique(a['groups'][env]):
                ids=np.flatnonzero(a['groups'][env]==group)
                if len(ids)<2 or not np.all(a['templates'][env,ids]==5):continue
                # Copies of each owner's physical box must match exactly.
                xy=a['xy'][:,env][:,ids]; vel=a['velocity'][:,env][:,ids]
                assert np.all(xy==xy[:,:1]) and np.all(vel==vel[:,:1])
                result=measure(a['near'][:,env][:,ids],xy[:,0],vel[:,0],alive[:,env][:,ids],dt)
                rows.append(dict(seed=int(folder.name[len('general_seed'):]),env=env,
                                 group=int(group),members=len(ids),**result))
    assert rows
    summary={str(n):summarize([r for r in rows if r['members']==n]) for n in (2,3,4)}
    summary['all']=summarize(rows)
    summary['per_seed']={str(s):summarize([r for r in rows if r['seed']==s]) for s in sorted({r['seed'] for r in rows})}
    (out/'departure_summary.json').write_text(json.dumps(summary,indent=2))
    with (out/'departure_groups.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    return summary


if __name__=='__main__':
    import sys
    print(json.dumps(process(Path(sys.argv[1])),indent=2))
