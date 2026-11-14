import csv
import json
import math
from pathlib import Path

import numpy as np


def stop_pairs(folder):
    folder = Path(folder)
    result = json.loads((folder/'result.json').read_text())
    with np.load(folder/'trace.npz') as trace:
        if 'waiting' in trace:
            values = trace['waiting']
            rows, owners = trace['waiting_rows'], trace['waiting_owners']
        elif (folder/'waiting_trace.pt').exists():
            import torch
            saved = torch.load(folder/'waiting_trace.pt', map_location='cpu', weights_only=False)
            columns = [saved['columns'].index(k) for k in ('speed_xyz', 'armed', 'released')]
            values = saved['values'].numpy()[..., columns]
            rows, owners = saved['rows'].numpy(), saved['owners'].numpy()
        else:
            return None
        valid = trace['valid'][:, rows, owners]
        if values.shape != (*valid.shape, 3) or not np.isfinite(values).all():
            raise ValueError(f'Invalid waiting telemetry: {folder}')
        packet, parent = trace['packet'], trace['predecessor']
        roots, boxes, goals = trace['roots'], trace['boxes'], trace['goals']
        ever, simultaneous = trace['ever'], trace['simultaneous']
        num_objects = trace['sizes'].shape[1]
        if set(zip(rows.tolist(), owners.tolist())) != set(zip(*np.nonzero(parent >= 0))):
            raise ValueError(f'Waiting pairs do not match BEFORE pairs: {folder}')
        steps = math.ceil(1.0/result['control_dt'] - 1e-6)
        names = {0: 'SIT', 1: 'CLIMB', 3: 'ON_TOP'}
        records = []
        for j, (env, owner) in enumerate(zip(rows, owners)):
            active = valid[:, j] & values[:, j, 1].astype(bool) & ~values[:, j, 2].astype(bool)
            stopped = active & (values[:, j, 0] <= .3)
            hits = (np.flatnonzero(np.convolve(stopped.astype(int), np.ones(steps, int), 'valid') == steps)
                    if len(stopped) >= steps else [])
            end = int(hits[0]+steps-1) if len(hits) else None
            leader = parent[env, owner]
            goal = packet[env, leader, 4] - packet.shape[1] - num_objects
            kind = int(packet[env, owner, 1])
            root_distance = payload_distance = None
            if end is not None:
                target = goals[env, goal, :2]
                root_distance = float(np.linalg.norm(roots[end, env, owner, :2]-target))
                if kind == 3:
                    payload = packet[env, owner, 3] - packet.shape[1]
                    payload_distance = float(np.linalg.norm(boxes[end, env, payload, :2]-target))
            records.append(dict(seed=result['seed'], env=int(env), follower=int(owner), task=names[kind],
                eligible=int(active.sum() >= steps), stopped=int(end is not None),
                stop_step=end+1 if end is not None else None, root_distance=root_distance,
                payload_distance=payload_distance, follower_simultaneous=int(simultaneous[env, owner, 1]),
                leader_ever=int(ever[env, leader, 1])))
        return records


def summarize(records):
    groups = []
    for task in ('SIT', 'CLIMB', 'ON_TOP', 'ALL'):
        pairs = [r for r in records if task == 'ALL' or r['task'] == task]
        eligible = [r for r in pairs if r['eligible']]
        stopped = [r for r in pairs if r['stopped']]
        item = dict(task=task, pairs=len(pairs), eligible=len(eligible), stopped=len(stopped),
                    stop_rate=len(stopped)/len(eligible) if eligible else None)
        for key in ('root_distance', 'payload_distance'):
            values = [r[key] for r in stopped if r[key] is not None]
            for label, percentile in (('p25', 25), ('median', 50), ('p75', 75)):
                item[key+'_'+label] = float(np.percentile(values, percentile)) if values else None
        for label, subset in (('all', pairs), ('eligible', eligible), ('stopped', stopped)):
            item[label+'_success'] = dict(n=len(subset),
                follower_simultaneous=sum(r['follower_simultaneous'] for r in subset),
                same_pair_leader_ever=sum(r['leader_ever'] for r in subset))
        groups.append(item)
    return groups


def write_report(folders, output):
    runs = [stop_pairs(folder) for folder in folders]
    if all(r is None for r in runs):
        return None
    if any(r is None for r in runs):
        raise ValueError('Some seeds have no waiting telemetry; do not mix diagnostics')
    seeds = [json.loads((Path(p)/'result.json').read_text())['seed'] for p in folders]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate seeds in stopping report')
    records = [r for run in runs for r in run]
    report = dict(protocol='same formal rollout; individual-fall freeze', speed_xyz_max=.3,
        duration_seconds=1., distance_statistics='pooled stopped pairs across seeds',
        sources=[str(p) for p in folders], seeds=seeds, groups=summarize(records),
        per_seed={str(seed): summarize(run) for seed, run in zip(seeds, runs)})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'stops.json').write_text(json.dumps(report, indent=2))
    with (output/'stop_pairs.csv').open('w') as f:
        fields = list(records[0]) if records else ['seed', 'env', 'follower', 'task']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return report
