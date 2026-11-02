"""Camera framing for observing a group from behind the pushing direction."""
import numpy as np


def behind_camera(people, box, width, forward):
    people = np.asarray(people, dtype=float)
    box = np.asarray(box, dtype=float)
    forward = np.asarray(forward, dtype=float)[:2]
    norm = np.linalg.norm(forward)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('Camera requires a finite pushing direction')
    forward = forward/norm
    side = np.array([-forward[1], forward[0]])
    span = float(np.ptp(people[:, :2] @ side))
    distance = max(6., max(float(width), span)*.8+1.2)
    target = .65*people.mean(0)+.35*box[:3]
    target[2] = 1.
    eye = target.copy()
    eye[:2] += -distance*forward+.6*side
    eye[2] = 3.4
    return eye, target
