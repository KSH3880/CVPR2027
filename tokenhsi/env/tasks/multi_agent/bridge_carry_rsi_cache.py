import hashlib
import json
import os
from pathlib import Path

import torch
from isaacgym.torch_utils import quat_conjugate, quat_mul

from utils import torch_utils


def validate(env):
    scene = env['sensorScene']
    spec = scene['bridge']['carry_boundary_rsi']
    if (spec != dict(version=1, categories=['cross', 'independent'],
                     cache_directory='output/rsi_cache_bridge_carry_v1')
            or not scene['sensor'].get('exclude_own_payload')):
        raise ValueError('Carry boundary RSI requires its v1 hidebox configuration')


def signature(cfg, lib):
    env = cfg['env']
    settings = dict(version=1, sim=cfg['sim'], env={k: env[k] for k in
        ('asset', 'plane', 'powerScale', 'enableSelfCollisionDetection', 'controlFrequencyInv')},
        ranges=env['box']['build']['taskSizeRanges'], density=100., damping=.01,
        screen_seconds=.1, speed=[.3, 1.5], direction_cos=.8660254, history=.1)
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    asset = env['asset']
    digest.update((Path(asset['assetRoot'])/asset['assetFileName']).read_bytes())
    for value in (lib.gts, lib.grs, lib.lrs, lib.grvs, lib.gravs, lib.dvs,
                  lib._motion_dt, lib._motion_weights, lib.obj_gts, lib.obj_grs):
        digest.update(value.cpu().numpy().tobytes())
    for name in ('bridge_carry_rsi_cache.py', 'bridge_carry_rsi_probe.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()[:24]


def object_state(lib, mids, times):
    pos, quat = lib.get_obj_motion_state(mids, times)
    delta = lib._motion_dt[mids]
    lo = (times-delta/2).clamp_min(0.)
    hi = torch.minimum(times+delta/2, lib._motion_lengths[mids])
    p0, q0 = lib.get_obj_motion_state(mids, lo)
    p1, q1 = lib.get_obj_motion_state(mids, hi)
    dt = (hi-lo).clamp_min(1e-6)
    angle = quat_mul(q1, quat_conjugate(q0))
    angle = torch.where(angle[:, 3:4] < 0, -angle, angle)
    norm = angle[:, :3].norm(dim=-1).clamp_min(1e-8)
    angular = angle[:, :3]*(2*torch.atan2(norm, angle[:, 3])/norm/dt)[:, None]
    return torch.cat((pos, quat, (p1-p0)/dt[:, None], angular), -1)


def frame_valid(task, root, body, box, size, times):
    from env.tasks.multi_agent.edge_ontop_reward import vertical_extent
    from utils.geometry import rotate
    heading = torch_utils.calc_heading(root[:, 3:7])
    forward = torch.stack((heading.cos(), heading.sin()), -1)
    speed = root[:, 7:9].norm(dim=-1)
    good = (speed >= .3) & (speed <= 1.5)
    good &= (root[:, 7:9]*forward).sum(-1) >= speed*.8660254
    hands = body[:, task._key_body_ids[:2], :3].mean(1)
    good &= (hands-box[:, :3]).square().sum(-1) <= -torch.log(root.new_tensor(.9))/10
    good &= box[:, 2]-vertical_extent(box[:, 3:7], size/2) > .08
    inverse = box[:, 3:7].clone(); inverse[:, :3] *= -1
    local = rotate(inverse[:, None], body[..., :3]-box[:, None, :3])
    good &= ~((local.abs() < size[:, None]/2-.03).all(-1).any(-1))
    return good & (times >= max(.1, (task._num_amp_obs_steps-1)*task.dt))


class CarryRsiCache:
    def __init__(self, task):
        lib = task._motion_lib['carryWith']
        key = signature(task.cfg, lib)
        directory = os.environ.get('CARRY_RSI_CACHE_DIR', task._scene_cfg['bridge']['carry_boundary_rsi']['cache_directory'])
        self.path = Path(directory)/(key+'.pt')
        if not self.path.exists():
            raise RuntimeError('Prepare carry RSI cache first: '+str(self.path))
        data = torch.load(self.path, map_location='cpu', weights_only=False)
        if data['signature'] != key:
            raise ValueError('Carry RSI cache physics/motion signature mismatch')
        records = data['records']
        self.keys = torch.tensor(list(records), device=task.device)
        counts = [len(v) for v in records.values()]
        if not sum(counts):
            raise ValueError('Carry RSI cache has no valid frames')
        self.counts = torch.tensor(counts, device=task.device)
        self.pool = torch.zeros(len(counts), max(counts), 2, dtype=torch.long, device=task.device)
        for i, frames in enumerate(records.values()):
            if frames:
                self.pool[i, :len(frames)] = torch.tensor(frames, device=task.device)
        print('[carry RSI] cache', self.path, 'supported', sum(c > 0 for c in counts), '/', len(counts), flush=True)

    def sample(self, sizes):
        match = ((sizes[:, None]*20).round().long() == self.keys).all(-1)
        row = match.long().argmax(-1)
        count = self.counts[row]
        valid = match.any(-1) & (count > 0)
        index = (torch.rand(len(sizes), device=sizes.device)*count.clamp_min(1)).long()
        pair = self.pool[row, index]
        return pair[:, 0], pair[:, 1], valid
