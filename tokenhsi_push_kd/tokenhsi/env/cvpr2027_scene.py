"""CPU single-scene generation for the CVPR protocol."""
import numpy as np

PRESETS = {'sit': 'sit', 'climb': 'climb', 'at': 'holding_at',
           'on_top': 'holding_ontop', 'push': 'push'}


def sample_scene(rng, roots, sizes, task, attempts=4096):
    """Generate fixed task scenes, reject geometry only, never policy outcomes.

    N=1 has no inter-group path intersections. Conservative circumscribed XY
    radii screen overlap; support is intentionally the endpoint for ON_TOP.
    """
    roots, sizes = np.asarray(roots), np.asarray(sizes)
    n = len(roots)
    boxes = np.zeros((n, 2, 13), dtype=np.float32)
    goals = np.zeros((n, 3), dtype=np.float32)
    pending = np.arange(n)
    lo = 2. if task == 'push' else 1.
    for _ in range(attempts):
        if not len(pending):
            return boxes, goals
        count = len(pending)
        angle = rng.uniform(-np.pi, np.pi, count)
        radius = rng.uniform(lo, 5., count)
        xy = roots[pending, :2] + radius[:, None]*np.stack((np.cos(angle), np.sin(angle)), -1)
        yaw = rng.uniform(-np.pi, np.pi, (count, 2))
        if task == 'push':
            heading = yaw[:, 0] + np.pi/2 + rng.uniform(-np.pi/4, np.pi/4, count)
        else:
            heading = rng.uniform(-np.pi, np.pi, count)
        travel = rng.uniform(lo, 5., count)
        target = xy + travel[:, None]*np.stack((np.cos(heading), np.sin(heading)), -1)
        if task in ('sit', 'climb'):
            target = xy.copy()
        r = np.linalg.norm(sizes[pending, :, :2], axis=-1)*.5
        valid = radius >= r[:, 0]+.3
        valid &= np.linalg.norm(target-roots[pending, :2], axis=-1) >= 1.
        if task == 'on_top':
            valid &= travel >= r[:, 0]+r[:, 1]+.05
            valid &= np.linalg.norm(target-roots[pending, :2], axis=-1) >= r[:, 1]+.3
            approach = xy-roots[pending, :2]
            t = np.clip(((target-roots[pending, :2])*approach).sum(-1)/np.maximum((approach*approach).sum(-1), 1e-12), 0, 1)
            valid &= np.linalg.norm(target-roots[pending, :2]-t[:, None]*approach, axis=-1) >= r[:, 1]+.3
        # Single-person distractor: common scene generation retained, then removed
        # from physics. Its sampled pose remains in the scene/observation archive.
        distractor_angle = rng.uniform(-np.pi, np.pi, count)
        distractor_r = 4.5*np.sqrt(rng.uniform(0., 1., count))
        extra = roots[pending, :2] + distractor_r[:, None]*np.stack((np.cos(distractor_angle), np.sin(distractor_angle)), -1)
        if task != 'on_top':
            valid &= np.linalg.norm(extra-roots[pending, :2], axis=-1) >= r[:, 1]+.3
            valid &= np.linalg.norm(extra-xy, axis=-1) >= r.sum(-1)+.05
            valid &= np.linalg.norm(extra-target, axis=-1) >= r.sum(-1)+.05
            for start, end in ((roots[pending, :2], xy), (xy, target)):
                delta = end-start
                t = np.clip(((extra-start)*delta).sum(-1)/np.maximum((delta*delta).sum(-1), 1e-12), 0, 1)
                valid &= np.linalg.norm(extra-start-t[:, None]*delta, axis=-1) >= r[:, 1]+.3
        accepted = pending[valid]
        boxes[accepted, 0, :2] = xy[valid]
        boxes[accepted, 1, :2] = (target if task == 'on_top' else extra)[valid]
        boxes[accepted, :, 2] = sizes[accepted, :, 2]/2
        boxes[accepted, :, 5] = np.sin(yaw[valid]/2)
        boxes[accepted, :, 6] = np.cos(yaw[valid]/2)
        goals[accepted, :2] = target[valid]
        goals[accepted, 2] = sizes[accepted, 0, 2]/2
        if task in ('sit', 'climb'):
            goals[accepted] = boxes[accepted, 0, :3]
            goals[accepted, 2] += sizes[accepted, 0, 2]/2 + (.1 if task == 'sit' else .94)
        if task == 'on_top':
            goals[accepted, 2] = sizes[accepted, 1, 2]+sizes[accepted, 0, 2]/2
        pending = pending[~valid]
    raise RuntimeError('CVPR scene rejection limit reached; no scene was silently discarded')

