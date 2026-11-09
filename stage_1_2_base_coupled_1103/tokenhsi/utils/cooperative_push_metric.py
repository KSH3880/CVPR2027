"""CPU-only t_coop scoring of first-episode, control-rate physical traces."""
import math
import numpy as np


def surface_distance(hands, boxes, sizes):
    """Unsigned distance to all six faces of an oriented solid box (xyzw)."""
    delta = np.asarray(hands, dtype=np.float64) - boxes[..., None, :3]
    q = np.asarray(boxes[..., 3:7], dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError('Invalid box quaternion')
    q = q / norm
    xyz = -q[..., None, :3]  # inverse rotation
    twice = 2 * np.cross(xyz, delta)
    local = delta + q[..., None, 3:4] * twice + np.cross(xyz, twice)
    signed = np.abs(local) - np.asarray(sizes)[..., None, :] / 2
    # Interior points: distance to nearest boundary, not zero (distance to volume).
    return np.linalg.norm(np.maximum(signed, 0), axis=-1) + np.maximum(-signed.max(axis=-1), 0)


def validate_trace(data):
    required = ('hands', 'boxes', 'sizes', 'goals', 'groups', 'templates', 'dt', 'end_step')
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError('t_coop requires physical raw trace; missing: ' + ', '.join(missing))
    boxes = np.asarray(data['boxes'])
    if boxes.ndim != 4 or boxes.shape[-1] != 13:
        raise ValueError('boxes must have shape [frames, envs, agents, 13]')
    t, n, m, _ = boxes.shape
    shapes = dict(hands=(t,n,m,2,3), sizes=(n,m,3), goals=(n,m,3),
                  groups=(n,m), templates=(n,m), end_step=(n,))
    for key, shape in shapes.items():
        if np.shape(data[key]) != shape or not np.isfinite(data[key]).all():
            raise ValueError('Invalid trace field ' + key)
    if t < 2 or not np.isfinite(boxes).all() or np.any(data['sizes'] <= 0):
        raise ValueError('Invalid physical trace')
    dt = float(data['dt'])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be positive')
    end = np.asarray(data['end_step'])
    if np.any((end < 0) | (end >= t) | (end != np.floor(end))):
        raise ValueError('end_step must index the recorded episode terminal frame')
    return dt


def score(data, start_mode='displacement', contact_threshold=.12,
          displacement_threshold=.2, speed_threshold=.2, sustain_seconds=1.):
    """Frame 0 is reset. Measurement uses post-step frames, including terminal.

    A velocity run is ceil(seconds/dt) consecutive post-step samples, representing
    that many control intervals. Its confirmation must precede first success.
    """
    dt = validate_trace(data)
    if start_mode not in ('displacement', 'speed'):
        raise ValueError('start_mode must be displacement or speed')
    for v in (contact_threshold, displacement_threshold, speed_threshold, sustain_seconds):
        if not math.isfinite(v) or v <= 0:
            raise ValueError('Thresholds and duration must be finite and positive')
    boxes = np.asarray(data['boxes'])
    distance = surface_distance(data['hands'], boxes, data['sizes'])
    needed = max(1, math.ceil(sustain_seconds / dt - 1e-10))
    trials = []
    for env, assignments in enumerate(data['groups']):
        for group in np.unique(assignments):
            ids = np.flatnonzero(assignments == group)
            if len(ids) not in (2,3,4) or not np.all(data['templates'][env, ids] == 5):
                continue
            owner = ids[0]
            states = boxes[:,env,owner]
            # Never silently score logical region tokens as shared physical boxes.
            if not np.allclose(boxes[:,env,ids], states[:,None], atol=1e-5, rtol=0):
                raise ValueError('Group members must reference the same physical box')
            if not np.allclose(data['goals'][env,ids], data['goals'][env,owner], atol=1e-5, rtol=0):
                raise ValueError('Group members must share the box-center goal')
            end = int(data['end_step'][env])
            reached = np.flatnonzero(np.linalg.norm(states[:end+1,:3]-data['goals'][env,owner], axis=-1) <= .2)
            success = int(reached[0]) if len(reached) else None
            if success is not None:
                end = success
            start = None
            if start_mode == 'displacement':
                moving = np.flatnonzero(np.linalg.norm(states[1:end+1,:2]-states[0,:2], axis=-1) >= displacement_threshold)
                if len(moving):
                    start = int(moving[0]+1)
            else:
                run = 0
                for frame in range(1, end+1):
                    run = run+1 if np.linalg.norm(states[frame,7:9]) >= speed_threshold else 0
                    if run >= needed and (success is None or frame < success):
                        start = frame-needed+1
                        break
            frames = 0 if start is None else end-start+1
            contact = 0 if not frames else int((distance[start:end+1,env][:,ids] < contact_threshold).all(axis=(1,2)).sum())
            trials.append(dict(env=env, group=int(group), members=ids.tolist(), size=len(ids),
                               start_frame=start, end_frame=end, first_success_frame=success,
                               frames=frames, contact_frames=contact,
                               t_coop=100.*contact/frames if frames else 0.))
    summary = {}
    for key in ('2','3','4','all'):
        rows = [r for r in trials if key == 'all' or r['size'] == int(key)]
        total_frames = sum(r['frames'] for r in rows)
        summary[key] = dict(trials=len(rows), start_failures=sum(r['start_frame'] is None for r in rows),
                            t_coop_mean_percent=float(np.mean([r['t_coop'] for r in rows])) if rows else None,
                            contact_frames=sum(r['contact_frames'] for r in rows), frames=total_frames)
    return dict(protocol='t_coop_v1', config=dict(start_mode=start_mode, contact_threshold=contact_threshold,
                displacement_threshold=displacement_threshold, speed_threshold=speed_threshold,
                sustain_seconds=sustain_seconds, sustain_frames=needed, dt=dt,
                goal_radius=.2, contact_comparison='strict_less', includes_initial=False,
                includes_terminal=True, speed_confirmation='strictly_before_first_success'),
                summary=summary, trials=trials)


def sweep(data, contact_thresholds=(.06,.08,.12), speeds=tuple(i/10 for i in range(2,11)),
          modes=('displacement','speed'), sustain_seconds=1., displacement_threshold=.2):
    results = []
    for mode in modes:
        for threshold in contact_thresholds:
            for speed in (speeds if mode == 'speed' else (.2,)):
                results.append(score(data, mode, threshold, displacement_threshold, speed, sustain_seconds))
    return results
