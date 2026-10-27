"""Opt-in, size-screened carryWith starts for dependent ON_TOP followers only."""
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import torch

from env.tasks.multi_agent.humanoid_ma_shared_support import LocoRsi
from env.tasks.multi_agent.stage2_solo_rsi_cache import Stage2SoloRsiCache
from utils.size_rsi import SCREEN, task_probabilities


def make_cache(task):
    if task.cfg['env']['before'].get('full_rsi'):
        from env.tasks.multi_agent.before_full_rsi_cache import BeforeFullRsiCache
        return BeforeFullRsiCache(task)
    if not task.cfg['env']['before'].get('follower_carry_rsi'):
        return LocoRsi(task)
    return BeforeCarryRsiCache(task)


class BeforeCarryRsiCache(Stage2SoloRsiCache):
    """Reuse native single-person physics screening; never build other skills."""
    def __init__(self, env):
        self.env = env
        self.spec = dict(env.cfg['env']['sizeAwareRsi'], lateProbability=0.)
        self.strict_geometry = True
        digest = hashlib.sha256()
        settings = dict(version=1, sim=env.cfg['sim'], dt=env.sim_params.dt,
                        screen=SCREEN, spec=self.spec, density=100.,
                        env={k: env.cfg['env'][k] for k in
                             ('asset', 'plane', 'powerScale', 'enableSelfCollisionDetection')})
        digest.update(json.dumps(settings, sort_keys=True).encode())
        asset = env.cfg['env']['asset']
        digest.update((Path(asset['assetRoot'])/asset['assetFileName']).read_bytes())
        lib = env._motion_lib['carryWith']
        for value in (lib.gts, lib.grs, lib.lrs, lib.grvs, lib.gravs, lib.dvs,
                      lib._motion_dt, lib._motion_weights, lib.obj_gts, lib.obj_grs):
            digest.update(value.cpu().numpy().tobytes())
        for path in (Path(__file__), Path('tokenhsi/env/tasks/multi_agent/stage2_solo_rsi_cache.py'),
                     Path('tokenhsi/utils/size_rsi.py'), Path('tokenhsi/utils/stage1_v2.py')):
            digest.update(path.read_bytes())
        output = Path(env.cfg['args'].output_path)
        root = (output/'carry_rsi_cache' if 'output_etc/check' in output.as_posix()
                else Path('output/rsi_cache_before_carry'))
        self.path = root/(digest.hexdigest()[:24]+'.pkl.gz')
        self.entries = {}
        if self.path.exists():
            with gzip.open(self.path, 'rb') as stream:
                self.entries = pickle.load(stream)
        self.sizes = env._box_size.cpu().tolist()
        self.eligible = task_probabilities(env._box_size[:, :env.num_agents],
            env.cfg['env']['box']['build']['taskSizeRanges']).cpu()
        self._build_missing()
        self._prepare_sampling()
        available = self.carry_available
        summary = dict(profiles=len(self.entries), available_slots=int(available.sum()),
                       total_slots=available.numel(), cache=str(self.path))
        print('[BEFORE carry RSI]', json.dumps(summary), flush=True)
        self.path.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        if not available.any():
            raise RuntimeError('No size-compatible carryWith frames; cannot run carry RSI experiment')

    def _profiles(self, kind, owner, pending):
        return super()._profiles(kind, owner, pending) if kind == 'carryWith' else {}

    def _prepare_sampling(self):
        # Evaluation exposes only EVAL_SKILLS=loco. Carry starts are selected by
        # the BEFORE role override, so the pool must not depend on that list.
        env = self.env
        lib = env._motion_lib['carryWith']
        starts = torch.zeros(env.num_envs, env.num_agents, lib.num_motions(), 2, dtype=torch.long)
        counts = torch.zeros_like(starts)
        weights = torch.zeros(env.num_envs, env.num_agents, lib.num_motions())
        offsets, flat, total = {}, [], 0
        for row in range(env.num_envs):
            for owner in range(env.num_agents):
                key = self._key('carryWith', row, owner)
                if key not in offsets:
                    clip_pools = []
                    for mask, _ in self.entries[key]:
                        frames = torch.from_numpy(mask).nonzero().flatten()
                        clip_pools.append([(total, len(frames))]*2)
                        flat.append(frames)
                        total += len(frames)
                    offsets[key] = torch.tensor(clip_pools)
                starts[row, owner] = offsets[key][..., 0]
                counts[row, owner] = offsets[key][..., 1]
                weights[row, owner] = (counts[row, owner, :, 0] > 0)*lib._motion_weights.cpu()
        self.pools = {'carryWith': (starts.to(env.device), counts.to(env.device),
            torch.cat(flat).to(env.device), weights.to(env.device))}
        self.carry_available = weights.sum(-1).to(env.device) > 0

    def _screen_profiles(self, kind, owner, profiles, original_roots, original_dofs):
        super()._screen_profiles(kind, owner, profiles, original_roots, original_dofs)
        # Require an already-held, raised payload and the same own-body clearance
        # used by the BEFORE scene validator. Do not retarget a clip to fit a box.
        from utils.shared_support import rotate
        from env.tasks.multi_agent.edge_ontop_reward import vertical_extent
        lib = self.env._motion_lib['carryWith']
        for clip in range(lib.num_motions()):
            frames = int(lib._motion_num_frames[clip])
            ids = torch.full((frames,), clip, device=self.env.device, dtype=torch.long)
            times = torch.arange(frames, device=self.env.device)*lib._motion_dt[clip]
            bodies = lib.get_motion_state_max(ids, times)[0]
            pos, quat = lib.get_obj_motion_state(ids, times)
            inverse = quat.clone(); inverse[:, :3] *= -1
            local = rotate(inverse[:, None], bodies-pos[:, None])
            hands = bodies[:, self.env._key_body_ids[:2]].mean(1)
            held = (hands-pos).square().sum(-1) <= -torch.log(pos.new_tensor(.9))/10
            for key in profiles:
                size = pos.new_tensor(key[1])/20
                clear = ~((local.abs() < size/2-.03).all(-1).any(-1))
                lifted = pos[:, 2]-vertical_extent(quat, size/2) > .08
                valid, phase = self.entries[key][clip]
                valid &= (clear & lifted & held).cpu().numpy()

    def availability(self, rows, owners, templates):
        result = torch.zeros(len(rows), len(self.env._skill), dtype=torch.bool, device=rows.device)
        result[:, self.env._skill.index('loco')] = True
        if 'carryWith' in self.env._skill:
            result[:, self.env._skill.index('carryWith')] = self.carry_available[rows, owners]
        return result

    def sample(self, skill, rows, owners, templates):
        if skill not in ('loco', 'carryWith'):
            raise ValueError('BEFORE carry RSI supports only loco/carryWith')
        return super().sample(skill, rows, owners, templates)
