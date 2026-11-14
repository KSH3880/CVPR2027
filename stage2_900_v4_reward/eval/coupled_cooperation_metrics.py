import ast
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(None)
def reference_function(directory, filename, name):
    source = Path(directory)/filename
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]


def rim_points(sizes):
    points = []
    for size in sizes.astype(np.float64):
        hx, hy = size[:2]/2
        s = np.arange(64)*(4*(hx+hy)/64)
        p = np.zeros((64, 3))
        a = s < 2*hx
        b = (s >= 2*hx) & (s < 2*hx+2*hy)
        c = (s >= 2*hx+2*hy) & (s < 4*hx+2*hy)
        d = ~(a | b | c)
        p[a, 0], p[a, 1] = -hx+s[a], -hy
        p[b, 0], p[b, 1] = hx, -hy+s[b]-2*hx
        p[c, 0], p[c, 1] = hx-(s[c]-2*hx-2*hy), hy
        p[d, 0], p[d, 1] = -hx, hy-(s[d]-4*hx-2*hy)
        points.append(p)
    return torch.from_numpy(np.array(points, dtype=np.float32))


def world_points(pose, sizes, reference):
    steps, count = pose.shape[:2]
    local = rim_points(sizes)[None].expand(steps, -1, -1, -1).reshape(-1, 64, 3)
    quat = torch.from_numpy(pose[..., 3:7].copy()).reshape(-1, 4)
    rotate = reference_function(reference, 'torch_utils.py', 'quat_rotate_dimflex')
    return (rotate(quat, local).reshape(steps, count, 64, 3)
            + torch.from_numpy(pose[..., :3])[..., None, :]).transpose(0, 1).contiguous()


def counters(pose, initial, first_success, threshold):
    steps = len(pose)
    displacement = np.linalg.norm(pose[..., :2].astype(np.float64)-initial[None, :, :2], axis=-1)
    last = np.where(first_success >= 1, first_success, steps)
    good = (displacement >= threshold) & (np.arange(steps)[:, None] < last)
    start = np.where(good.any(0), good.argmax(0)+1, -1)
    return start, np.where(start >= 1, start-1, steps), np.where(first_success >= 1, first_success-1, steps)


def ratios(count, start, end):
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = count.astype(np.float32)/(end-start).astype(np.float32)
    return raw, np.clip(raw, 0, 1)


def surface_contact(hands, pose, sizes):
    from table2_metrics import surface_distance
    steps, count = pose.shape[:2]
    distance = surface_distance(torch.from_numpy(hands).reshape(steps*count, -1, 2, 3),
                               torch.from_numpy(pose).reshape(-1, 7),
                               torch.from_numpy(sizes)[None].expand(steps, -1, -1).reshape(-1, 3))
    return (distance < .12).all(-1).all(-1).reshape(steps, count).numpy()


def calculate(job, output, reference):
    torch.set_num_threads(1)
    source, destination = Path(job['source']), Path(output)/job['name']
    destination.mkdir(exist_ok=True)
    import json
    filenames = ('initial.pt', 'cooperation.npz', 'trace.npz', 'table2_trials.npz', 'table2.json')
    hashes = {str(source/f): sha(source/f) for f in filenames}
    initial = torch.load(source/'initial.pt', map_location='cpu', weights_only=False)
    recorded = json.loads((source/'table2.json').read_text())
    with np.load(source/'cooperation.npz') as archive:
        pose, sizes, hands = (archive[k] for k in ('physical_boxes', 'physical_sizes', 'hands'))
    with np.load(source/'table2_trials.npz') as archive:
        first_success = archive['first_success']
    with np.load(source/'trace.npz') as archive:
        alive = archive['valid']
    steps, count = pose.shape[:2]
    if (steps, count, hands.shape[2]) != (900, 512, job['people']):
        raise ValueError('Unexpected recorded cohort')
    goal = initial['shared_center_goals'][:, 0].numpy().astype(np.float64)
    success = (np.linalg.norm(pose[..., :3].astype(np.float64)-goal[None], axis=-1) <= .2) & alive.any(-1)
    first = np.where(success.any(0), success.argmax(0)+1, -1)
    if not np.array_equal(first, first_success):
        raise ValueError('First shared-center success mismatch')
    if int((first >= 1).sum()) != recorded['successes']:
        raise ValueError('Group Success Rate mismatch')
    contacts = surface_contact(hands, pose, sizes)
    all_counts = contacts.sum(0)
    arrays = dict(first_success=first, cooperative_frames_all_episode=all_counts)
    initial_center = initial['shared_centers'][:, 0].numpy().astype(np.float64)
    variants = {}
    for threshold in (.2, .5):
        start, a, b = counters(pose, initial_center, first, threshold)
        raw, ratio = ratios(all_counts, a, b)
        key = str(round(threshold*100))
        arrays.update({key+'__start': start, key+'__start_counter': a,
                       key+'__end_counter': b, key+'__tcoop': ratio,
                       key+'__jerk': np.zeros(count, dtype=np.float32)})
        variants[key] = dict(no_start=int((start < 1).sum()), nonpositive_duration=int((b <= a).sum()),
                             upper_clipped=int((raw > 1).sum()), lower_clipped=int((raw < 0).sum()),
                             tcoop_nan=int(np.isnan(ratio).sum()))
    jerk = reference_function(reference, 'task_util.py', 'compute_rim_velocity_variance_norm')
    formula_checks = []
    for offset in range(0, count, 8):
        stop = min(offset+8, count)
        points = world_points(pose[:, offset:stop], sizes[offset:stop], reference)
        ids = torch.arange(stop-offset)
        for key in variants:
            a, b = (torch.from_numpy(arrays[key+suffix][offset:stop])
                    for suffix in ('__start_counter', '__end_counter'))
            value = jerk(points, a, b, ids, delta_t=.0333)
            arrays[key+'__jerk'][offset:stop] = value.numpy()
            if offset == 0:
                velocity = (points[:, 1:]-points[:, :-1])/.0333
                acceleration = (velocity[:, 1:]-velocity[:, :-1])/.0333
                j = ((acceleration[:, 1:]-acceleration[:, :-1])/.0333).abs()
                t = torch.arange(steps-3)[None]
                mask = (t >= a[:, None]) & (t+2 < b[:, None])
                expected = (j*mask[..., None, None]).sum((1, 2, 3))/(mask.sum(1)*64*3).clamp_min(1)
                if not torch.equal(value, expected):
                    raise ValueError('Jerk formula/mask mismatch')
                formula_checks.append(key)
    for key, variant in variants.items():
        c, j = arrays[key+'__tcoop'], arrays[key+'__jerk']
        variant.update(tcoop_percent=float(np.nanmean(c)*100), tcoop_valid=int(np.isfinite(c).sum()),
                       jerk=float(j[j != 0].mean()), jerk_nonzero=int((j != 0).sum()),
                       jerk_zero=int((j == 0).sum()))
    if any(sha(p) != value for p, value in hashes.items()):
        raise ValueError('Original recording modified')
    np.savez_compressed(destination/'trials.npz', **arrays)
    row = dict(**job, steps=steps, trials=count, successes=int((first >= 1).sum()),
               group_success_rate_percent=float((first >= 1).mean()*100), variants=variants,
               first_success_exact=True, source_sha256=hashes, jerk_formula_exact=formula_checks)
    (destination/'result.json').write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
    return row
