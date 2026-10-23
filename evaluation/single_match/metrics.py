"""Position-only point/region scores; independent of simulator and training rewards."""
import numpy as np

TASKS = ('at', 'ontop', 'sit', 'climb', 'push')
PROTOCOL = 'single_match_region_v1'
PUSH_PROTOCOL = 'single_push_point_region_v1'


def rotation_z_row(q):
    """World Z row of rotation matrix, xyzw quaternion."""
    q = np.asarray(q)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1)


def yaw(q):
    x, y, z, w = np.moveaxis(np.asarray(q), -1, 0)
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def geometry(task, root, boxes, sizes, goal, zone_yaw=None):
    """Return inspection point, current target, XY region center/size/yaw."""
    if task not in TASKS:
        raise ValueError('Unknown task: ' + task)
    point = boxes[..., 0, :3] if task in ('at', 'ontop', 'push') else root
    target = np.array(goal, copy=True)
    if task in ('at', 'push'):
        extent = np.ones_like(point[..., :2])  # PUSH fixed 1 m x 1 m region.
        angle = np.zeros_like(point[..., 0]) if zone_yaw is None else zone_yaw
        return point, target, target[..., :2], extent, angle
    index = 1 if task == 'ontop' else 0
    support = boxes[..., index, :]
    size = sizes[..., index, :]
    top = support[..., 2] + (np.abs(rotation_z_row(support[..., 3:7]))*size/2).sum(-1)
    offset = sizes[..., 0, 2]/2 if task == 'ontop' else (.10 if task == 'sit' else .94)
    target = np.concatenate((support[..., :2], (top+offset)[..., None]), -1)
    return point, target, support[..., :2], size[..., :2], yaw(support[..., 3:7])


def classify(task, root, boxes, sizes, goal, zone_yaw=None, dtype=np.float64):
    root, boxes, sizes, goal = [np.asarray(v, dtype=dtype) for v in (root, boxes, sizes, goal)]
    if zone_yaw is not None:
        zone_yaw = np.asarray(zone_yaw, dtype=dtype)
    point, target, center, extent, angle = geometry(task, root, boxes, sizes, goal, zone_yaw)
    delta = point-target
    point_hit = np.linalg.norm(delta, axis=-1) <= dtype(.20)
    if task == 'at':
        return point_hit, point_hit.copy()
    xy = point[..., :2]-center
    u = np.cos(angle)*xy[..., 0]+np.sin(angle)*xy[..., 1]
    v = -np.sin(angle)*xy[..., 0]+np.cos(angle)*xy[..., 1]
    inside = (np.abs(u) <= dtype(.4)*extent[..., 0]) & (np.abs(v) <= dtype(.4)*extent[..., 1])
    return point_hit, inside & (np.abs(delta[..., 2]) <= dtype(.20))


def score_trace(task, trace, dt):
    """active is the pre-step alive mask, including each terminating frame."""
    active = np.asarray(trace['active'], dtype=bool)
    if active.ndim != 2 or not active[0].all() or ((~active[:-1]) & active[1:]).any():
        raise ValueError('Trace must contain exactly one initial episode per environment')
    if not (np.asarray(trace['done'], dtype=bool) & active).sum(0).astype(int).tolist() == [1]*active.shape[1]:
        raise ValueError('Each environment needs exactly one terminating frame')
    arrays = [trace[k] for k in ('root', 'boxes', 'sizes', 'goal')]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Non-finite geometry in trace')
    scores = classify(task, *arrays, zone_yaw=trace.get('zone_yaw'))
    single = classify(task, *arrays, zone_yaw=trace.get('zone_yaw'), dtype=np.float32)
    result = {}
    for name, hit, hit32 in zip(('point', 'region'), scores, single):
        ever = (hit & active).any(0)
        if not np.array_equal(ever, (hit32 & active).any(0)):
            raise ValueError('float32/64 ever disagreement: ' + name)
        times = np.where(hit & active, (np.arange(len(active))+1)[:, None]*dt, np.inf).min(0)
        result[name] = dict(count=int(ever.sum()), trials=len(ever), rate_percent=float(100*ever.mean()),
                            ever=ever.tolist(), first_hit_seconds=[float(t) if np.isfinite(t) else None for t in times],
                            successful_mean_first_hit_seconds=float(times[ever].mean()) if ever.any() else None)
    if task == 'push':
        tilted = rotation_z_row(np.asarray(trace['boxes'])[..., 0, 3:7])[..., 2] < np.cos(np.pi/4)
        initial = np.asarray(trace['initial_box'])
        goal = np.asarray(trace['goal'])
        start_distance = np.linalg.norm(initial[..., :2]-goal[0, ..., :2], axis=-1)
        distances = np.linalg.norm(np.asarray(trace['boxes'])[..., 0, :2]-goal[..., :2], axis=-1)
        ending = (np.asarray(trace['done'], dtype=bool) & active).argmax(0)
        final_distance = distances[ending, np.arange(len(ending))]
        result['push_diagnostics'] = dict(topple_threshold_degrees=45,
            topple_ever_percent=float(100*(tilted & active).any(0).mean()),
            mean_final_distance_m=float(final_distance.mean()),
            mean_final_goal_reduction_m=float((start_distance-final_distance).mean()),
            mean_best_goal_reduction_m=float((start_distance-np.where(active, distances, np.inf).min(0)).mean()))
    return result


def aggregate(records):
    """Equal-weight tasks within seed, then mean/sample std across seeds."""
    if not records:
        raise ValueError('No results')
    grouped = {}
    for row in records:
        key = (row['policy'], row['task'], row.get('start', 'loco'), tuple(row.get('distance_range', [])))
        grouped.setdefault(key, []).append(row)
    output = []
    for key, rows in grouped.items():
        if len({r['seed'] for r in rows}) != len(rows):
            raise ValueError('Duplicate seed result')
        item = dict(policy=key[0], task=key[1], start=key[2], distance_range=list(key[3]), seeds=sorted(r['seed'] for r in rows))
        for metric in ('point', 'region'):
            values = [r['scores'][metric]['rate_percent'] for r in rows]
            item[metric] = dict(mean_percent=float(np.mean(values)), sample_std_percent=float(np.std(values, ddof=1)) if len(values)>1 else None)
        output.append(item)
    return output
