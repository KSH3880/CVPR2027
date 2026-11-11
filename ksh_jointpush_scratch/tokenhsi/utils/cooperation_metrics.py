import math
import statistics
import numpy as np
import torch


CONTACT_DISTANCES = (.06, .08, .12)
STARTS = [('displacement', value) for value in (.2, .5, 1.)] + [('speed', value) for value in (.2, .5, 1.)]


def surface_distance(hands, boxes, sizes):
    delta = hands - boxes[..., None, :3]
    xyz = -boxes[..., None, 3:6]
    uv = torch.cross(xyz.expand_as(delta), delta, dim=-1)
    local = delta + 2*(boxes[..., None, 6:7]*uv + torch.cross(xyz.expand_as(delta), uv, dim=-1))
    excess = local.abs() - sizes[..., None, :]/2
    return (excess.clamp_min(0).norm(dim=-1) + excess.amax(-1).clamp_max(0)).abs()


def group_trials(groups, packet):
    envs, representatives, members = [], [], []
    for env, row in enumerate(groups):
        for group in np.unique(row):
            mask = (row == group) & packet[env, :, 0].astype(bool) & (packet[env, :, 1] == 4)
            if mask.sum() >= 2:
                envs.append(env)
                representatives.append(np.flatnonzero(mask)[0])
                members.append(mask)
    return np.asarray(envs, dtype=int), np.asarray(representatives, dtype=int), np.asarray(members, dtype=bool).reshape(-1, groups.shape[1])


def window_counts(good, contact, end, required_frames=1):
    steps, trials = good.shape
    prefix = np.concatenate([np.zeros((1, trials), dtype=np.int64), good.cumsum(axis=0)])
    if required_frames > steps:
        return dict(start=np.full(trials, -1), window_frames=np.zeros(trials, dtype=int), contact_frames=np.zeros(trials, dtype=int))
    hits = prefix[required_frames:] - prefix[:-required_frames] == required_frames
    hits &= np.arange(required_frames, steps+1)[:, None] <= end
    found = hits.any(axis=0)
    start = np.where(found, hits.argmax(axis=0), -1)
    counts = np.concatenate([np.zeros((1, trials), dtype=np.int64), contact.cumsum(axis=0)])
    ids = np.arange(trials)
    return dict(start=np.where(found, start+1, -1), window_frames=np.where(found, end-start, 0),
                contact_frames=np.where(found, counts[end, ids]-counts[start.clip(0), ids], 0))


def calculate(raw, trace, dt, distances=None):
    envs, reps, members = group_trials(trace['groups'], trace['packet'])
    steps = len(raw['boxes'])
    if distances is None:
        distances = np.empty(raw['hands'].shape[:-1], dtype=raw['hands'].dtype)
        for frame in range(steps):
            distances[frame] = surface_distance(torch.from_numpy(raw['hands'][frame]),
                torch.from_numpy(raw['boxes'][frame]), torch.from_numpy(raw['sizes'])).numpy()
    boxes = raw['boxes'][:, envs, reps, :3].astype(np.float64)
    origin = raw['initial_boxes'][envs, reps, :3].astype(np.float64)
    goals = raw['goals'][envs, reps].astype(np.float64)
    alive = trace['valid'][:, envs]
    group_alive = (alive | ~members[None]).all(-1)
    any_alive = (alive & members[None]).any(-1)
    success = (np.linalg.norm(boxes-goals, axis=-1) <= .2) & any_alive
    first = np.where(success.any(0), success.argmax(0)+1, -1)
    end = np.where(first >= 1, first, steps)
    displacement = np.linalg.norm(boxes[..., :2]-origin[None, ..., :2], axis=-1)
    positions = np.concatenate([origin[None, ..., :2], boxes[..., :2]], axis=0)
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=-1)/dt
    counts = dict(env=envs, representative=reps, members=members, people=members.sum(-1), first_success=first)
    variants = {}
    for kind, value in STARTS:
        good = displacement >= value if kind == 'displacement' else speed >= value
        required = 1 if kind == 'displacement' else math.ceil(1/dt-1e-9)
        for distance in CONTACT_DISTANCES:
            contact = ((distances[:, envs] < distance).all(-1) | ~members[None]).all(-1) & group_alive
            key = f'{kind}_{value:g}_contact_{distance:g}'
            arrays = window_counts(good, contact, end, required)
            counts.update({key+'__'+name: array for name, array in arrays.items()})
            ratio = np.divide(arrays['contact_frames'], arrays['window_frames'],
                              out=np.zeros(len(envs), dtype=float), where=arrays['window_frames'] > 0)*100
            by_size = {}
            for size in sorted(set(members.sum(-1).tolist())):
                mask = members.sum(-1) == size
                by_size[str(size)] = dict(trials=int(mask.sum()), cooperative_time_ratio_percent=float(ratio[mask].mean()),
                    started_percent=float((arrays['start'][mask] >= 1).mean()*100))
            variants[key] = dict(by_size=by_size, equal_size_mean_percent=statistics.mean(
                row['cooperative_time_ratio_percent'] for row in by_size.values()) if by_size else None,
                pooled_mean_percent=float(ratio.mean()) if len(ratio) else None)
    return dict(version=1, trials=len(envs), contact_distances_m=list(CONTACT_DISTANCES),
                start_hand_condition=False, speed_hold_seconds=1, no_start_ratio_percent=0,
                aggregation='trial mean within group size; equal mean across group sizes; then seed mean and sample std',
                variants=variants), counts


def summarize(runs):
    result = {}
    for key in runs[0]['variants']:
        rows = [run['variants'][key] for run in runs]
        values = [row['equal_size_mean_percent'] for row in rows if row['equal_size_mean_percent'] is not None]
        if not values:
            continue
        item = dict(mean_percent=statistics.mean(values), std_percent=statistics.stdev(values) if len(values)>1 else None,
                    seeds_percent=values, by_size={})
        for size in sorted({size for row in rows for size in row['by_size']}, key=int):
            parts = [row['by_size'][size] for row in rows if size in row['by_size']]
            item['by_size'][size] = dict(trials=sum(p['trials'] for p in parts))
            for metric in ('cooperative_time_ratio_percent', 'started_percent'):
                numbers = [p[metric] for p in parts]
                item['by_size'][size][metric] = dict(mean=statistics.mean(numbers),
                    std=statistics.stdev(numbers) if len(numbers)>1 else None, seeds=numbers)
        result[key] = item
    return dict(version=1, runs=len(runs), variants=result)
