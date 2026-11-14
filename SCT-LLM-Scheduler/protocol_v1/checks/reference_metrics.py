import numpy as np
import torch

from teamhoi_reference import quat_rotate_dimflex, compute_rim_velocity_variance_norm


def region_distances(point, target, support, size):
    point, target, support, size = [torch.as_tensor(v, dtype=torch.float64)
                                     for v in (point, target, support, size)]
    q = support[3:7]/support[3:7].norm()
    x, y, z, w = q.unbind()
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    dx, dy = (point[:2]-support[:2]).unbind()
    local = torch.stack((yaw.cos()*dx+yaw.sin()*dy, -yaw.sin()*dx+yaw.cos()*dy))
    outside = (local.abs()-size[:2]/2).clamp_min(0)
    vertical = (point[2]-target[2]).abs()
    region = torch.cat((outside, vertical[None])).norm()
    return dict(point_error=float((point-target).norm()), region_error=float(region),
                success=bool((outside == 0).all() & (vertical <= .2)))


def support_goal(support, size, offset):
    support, size = [torch.as_tensor(v, dtype=torch.float64) for v in (support, size)]
    q = support[3:7]/support[3:7].norm()
    x, y, z, w = q.unbind()
    row = torch.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
    goal = support[:3].clone()
    goal[2] += (row.abs()*size/2).sum()+offset
    return goal.numpy()


def first_enabling(success, valid):
    joint = (np.asarray(success, bool) & np.asarray(valid, bool)).all(-1)
    return int(joint.argmax()+1) if joint.any() else -1


def first_coupled(centers, goal, valid):
    delta = np.asarray(centers, np.float64)-np.asarray(goal, np.float64)
    good = (np.linalg.norm(delta, axis=-1) <= .2) & np.asarray(valid, bool).any(-1)
    return int(good.argmax()+1) if good.any() else -1


def start_counters(centers, initial, first_success):
    centers, initial = np.asarray(centers, np.float64), np.asarray(initial, np.float64)
    steps = len(centers)
    good = np.linalg.norm(centers[:, :2]-initial[:2], axis=-1) >= .2
    good[np.arange(steps) >= (first_success if first_success >= 1 else steps)] = False
    start = int(good.argmax()+1) if good.any() else -1
    return start, start-1 if start >= 1 else steps, first_success-1 if first_success >= 1 else steps


def cooperative_ratio(count, start_counter, end_counter):
    with np.errstate(divide='ignore', invalid='ignore'):
        value = np.float32(count)/np.float32(end_counter-start_counter)
    return float(np.clip(value, 0, 1))


def surface_distance(hands, box, size):
    hands, box, size = [torch.as_tensor(v, dtype=torch.float32) for v in (hands, box, size)]
    delta, xyz = hands-box[:3], -box[3:6]
    xyz = xyz.expand_as(delta)
    uv = torch.cross(xyz, delta, dim=-1)
    local = delta+2*(box[6]*uv+torch.cross(xyz, uv, dim=-1))
    excess = local.abs()-size/2
    return (excess.clamp_min(0).norm(dim=-1)+excess.amax(-1).clamp_max(0)).abs().numpy()


def minimum_error(distances, valid, successful):
    values = np.asarray(distances, np.float64)[np.asarray(valid, bool)]
    return float(values.min()*100) if successful and len(values) else None


def interaction_rates(independent, enabling, coupled):
    counts = [list(v) for v in (independent, enabling, coupled)]
    rates = [float(np.mean(v)*100) if len(v) else None for v in counts]
    return rates+[float(np.mean(rates)) if None not in rates else None]


def world_rim(pose, size):
    hx, hy = np.asarray(size, np.float32).astype(np.float64)[:2]/2
    s = np.arange(64)*(4*(hx+hy)/64)
    local = np.zeros((64, 3), np.float32)
    a = s < 2*hx
    b = (s >= 2*hx) & (s < 2*hx+2*hy)
    c = (s >= 2*hx+2*hy) & (s < 4*hx+2*hy)
    d = ~(a | b | c)
    local[a, 0], local[a, 1] = -hx+s[a], -hy
    local[b, 0], local[b, 1] = hx, -hy+s[b]-2*hx
    local[c, 0], local[c, 1] = hx-(s[c]-2*hx-2*hy), hy
    local[d, 0], local[d, 1] = -hx, hy-(s[d]-4*hx-2*hy)
    pose = torch.as_tensor(pose, dtype=torch.float32)
    p = torch.from_numpy(local)[None].expand(len(pose), -1, -1)
    return (quat_rotate_dimflex(pose[:, 3:7], p)+pose[:, None, :3]).numpy()


def mean_absolute_jerk(points, start_counter, end_counter):
    value = compute_rim_velocity_variance_norm(
        torch.as_tensor(points, dtype=torch.float32)[None], torch.tensor([start_counter]),
        torch.tensor([end_counter]), torch.tensor([0]), delta_t=.0333)
    return float(value[0])
