"""CPU float64 scoring; legacy XY80 and opt-in 1109 XY100, independent of rewards."""
import numpy as np

TASKS = ('SIT', 'CLIMB', 'AT', 'ON_TOP', 'PUSH')
HORIZON = 900
SEEDS = (0, 1, 2)
SCENES = 512


def spatial_metrics(task, point, support, support_size, goal, payload_height=0., xy_fraction=.8):
    """Broadcast arrays; quaternion convention is xyzw; sizes are full lengths.

    Return success[..., (Point, Region)], distances_m[..., (Point, Region)].
    Support height uses the actual oriented cuboid's top,
    while the XY region uses its yaw frame as required by section 1.5.
    """
    point, support, size, goal = [np.asarray(x, dtype=np.float64)
                                 for x in (point, support, support_size, goal)]
    task = np.asarray(task)
    q = support[..., 3:7]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    vertical = np.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1)
    top = support[..., 2] + (np.abs(vertical)*size*.5).sum(-1)
    offset = np.where(task == 0, .10, np.where(task == 1, .94,
                      np.asarray(payload_height, dtype=np.float64)*.5))
    target = support[..., :3].copy()
    target[..., 2] = top + offset
    transport = (task == 2) | (task == 4)
    target = np.where(transport[..., None], goal, target)
    if xy_fraction not in (.8, 1.):
        raise ValueError('Supported XY fractions: legacy .8 and protocol 1.0')
    point_distance = np.linalg.norm(point-target, axis=-1)
    point_ok = point_distance <= .20
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    delta = point[..., :2]-support[..., :2]
    local_x = np.cos(yaw)*delta[..., 0]+np.sin(yaw)*delta[..., 1]
    local_y = -np.sin(yaw)*delta[..., 0]+np.cos(yaw)*delta[..., 1]
    half = .5*xy_fraction
    dx = np.maximum(np.abs(local_x)-half*size[..., 0], 0.)
    dy = np.maximum(np.abs(local_y)-half*size[..., 1], 0.)
    dz = point[..., 2]-target[..., 2]
    region = (dx == 0) & (dy == 0) & (np.abs(dz) <= .20)
    region_distance = np.sqrt(dx*dx+dy*dy+dz*dz)
    return (np.stack((point_ok, np.where(transport, point_ok, region)), -1),
            np.stack((point_distance, np.where(transport, point_distance, region_distance)), -1))


def spatial_success(task, point, support, support_size, goal, payload_height=0., xy_fraction=.8):
    return spatial_metrics(task, point, support, support_size, goal, payload_height, xy_fraction)[0]


def trial_minimum_distances(distances, valid):
    """All valid frames, including before/after success and the fall frame."""
    return np.where(np.asarray(valid)[..., None], np.asarray(distances, dtype=np.float64), np.inf).min(axis=0)


def error_statistics(minimum_distances_m, region_success):
    """Both errors use the SAME Region-success trials, section 1.8 (cm)."""
    selected = np.asarray(minimum_distances_m, dtype=np.float64)[np.asarray(region_success, dtype=bool)]*100.
    if len(selected) and not np.isfinite(selected).all():
        raise ValueError('A successful trial must have finite valid distances')
    return {name: dict(mean_cm=float(selected[:, i].mean()) if len(selected) else None,
                       samples=len(selected)) for i, name in enumerate(('point', 'region'))}


def ever_from_trace(success, fallen):
    """Include the fall frame; exclude every later frame for that person."""
    success = np.asarray(success, dtype=bool)
    fallen = np.asarray(fallen, dtype=bool)
    before = np.concatenate((np.zeros_like(fallen[:1]), np.maximum.accumulate(fallen, axis=0)[:-1]), axis=0)
    valid = ~before
    return (success & valid[..., None]).any(0), valid


def counts(success, selected):
    selected = np.asarray(selected, dtype=bool)
    total = int(selected.sum())
    hits = int(np.asarray(success, dtype=bool)[selected].sum())
    return dict(success=hits, trials=total, rate=hits/total if total else None)


def group_scores(individual_ever, relation, group, coupled_ever, *, frame_success=None, frame_valid=None):
    """One row per scene/person. relation: 0 independent, 1 enable, 2 coupled.

    individual_ever is Region ever success. Coupled success comes from the physical
    shared center, never the conjunction of per-person/zone success.
    Enable requires raw Region success and validity arrays shaped (time, scene,
    person), so successes at different times cannot be combined. Validity includes
    each person's fall frame and excludes subsequent frames.
    """
    ever, relation, group, coupled = map(np.asarray, (individual_ever, relation, group, coupled_ever))
    if np.any(relation == 1):
        if frame_success is None or frame_valid is None:
            raise ValueError('Enable requires frame_success and frame_valid for simultaneous success')
        frames = np.asarray(frame_success, dtype=bool)
        valid = np.asarray(frame_valid, dtype=bool)
        if frames.ndim != 3 or frames.shape[1:] != ever.shape or valid.shape != frames.shape:
            raise ValueError('Enable frame arrays must have shape (time, scene, person)')
        frames = frames & valid
    ind = counts(ever, relation == 0)
    enabled, cooperative, sizes = [], [], {str(k): 0 for k in (2, 3, 4)}
    for scene in range(len(group)):
        for gid in np.unique(group[scene]):
            members = group[scene] == gid
            kinds = np.unique(relation[scene, members])
            if len(kinds) != 1:
                raise ValueError('A group cannot mix relationship types')
            kind = int(kinds[0])
            if kind == 1:
                if members.sum() != 2:
                    raise ValueError('Enable requires exactly two people')
                enabled.append(bool(frames[:, scene, members].all(axis=-1).any()))
            elif kind == 2:
                k = int(members.sum())
                if k not in (2, 3, 4):
                    raise ValueError('Coupled requires 2–4 people')
                sizes[str(k)] += 1
                cooperative.append(bool(coupled[scene, members].any()))
            elif kind != 0:
                raise ValueError('Unknown relationship type')
    enable = counts(enabled, np.ones(len(enabled), bool))
    coop = counts(cooperative, np.ones(len(cooperative), bool))
    rates = [x['rate'] for x in (ind, enable, coop)]
    return dict(independent=ind, enabling=enable, coupled=coop, coupled_sizes=sizes,
                mixed=float(np.mean(rates)) if all(x is not None for x in rates) else None)


def seed_statistics(values):
    if len(values) != 3:
        raise ValueError('Report exactly three independent seeds')
    if any(v is None for v in values):
        return dict(mean=None, std=None, ddof=1, seeds=values)
    return dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)), ddof=1, seeds=values)
