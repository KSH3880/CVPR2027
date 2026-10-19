import gzip
import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import torch

from utils.size_rsi import (SCREEN, interaction_phase, late_frames,
                            screen_states, size_key, task_probabilities)


class SizeRsiCache:
    def __init__(self, env):
        self.env = env
        self.spec = env.cfg['env']['sizeAwareRsi']
        digest = hashlib.sha256()
        settings = {'version': 1, 'sim': env.cfg['sim'], 'dt': env.sim_params.dt,
                    'screen': SCREEN, 'spec': self.spec, 'density': 100.,
                    'env': {k: env.cfg['env'][k] for k in
                            ('asset', 'plane', 'powerScale', 'enableSelfCollisionDetection')}}
        digest.update(json.dumps(settings, sort_keys=True).encode())
        asset = env.cfg['env']['asset']
        digest.update((Path(asset['assetRoot']) / asset['assetFileName']).read_bytes())
        for name in ('loco', 'sit', 'climb', 'pickUp', 'carryWith', 'putDown'):
            lib = env._motion_lib[name]
            digest.update(name.encode())
            for tensor in (lib.gts, lib.grs, lib.lrs, lib.grvs, lib.gravs, lib.dvs,
                           lib._motion_dt, lib._motion_weights):
                digest.update(tensor.cpu().numpy().tobytes())
            if hasattr(lib, 'obj_gts'):
                digest.update(lib.obj_gts.cpu().numpy().tobytes())
                digest.update(lib.obj_grs.cpu().numpy().tobytes())
        for file in (Path(__file__), Path('tokenhsi/utils/size_rsi.py')):
            digest.update(file.read_bytes())
        self.path = Path(self.spec['cacheDirectory']) / (digest.hexdigest()[:24] + '.pkl.gz')
        self.entries = {}
        if self.path.exists():
            with gzip.open(self.path, 'rb') as stream:
                self.entries = pickle.load(stream)
        self.sizes = env._box_size.cpu().tolist()
        self.eligible = task_probabilities(env._box_size[:, :2]).cpu()
        self._build_missing()
        self._prepare_sampling()

    def _key(self, kind, row, owner):
        source = size_key(self.sizes[row][owner])
        return (kind, source, size_key(self.sizes[row][2+owner])) if kind == 'putDownOnTop' else (kind, source)

    def _profiles(self, kind, owner, pending):
        task = 1 if kind == 'sit' else 2 if kind == 'climb' else 4 if kind == 'putDownOnTop' else 0
        eligible = self.eligible[:, owner, task] > 0
        representatives = {}
        for row in eligible.nonzero().flatten().tolist():
            key = self._key(kind, row, owner)
            if key not in self.entries and key not in pending:
                representatives.setdefault(key, row)
        pending.update(representatives)
        return representatives

    def _build_missing(self):
        env = self.env
        pending = set()
        work = [(kind, owner, self._profiles(kind, owner, pending))
                for kind in ('sit', 'climb', 'pickUp', 'carryWith', 'putDown', 'putDownOnTop')
                for owner in (0, 1)]
        if not any(profiles for _, _, profiles in work):
            print('[size RSI] cache hit:', self.path, flush=True)
            return
        print('[size RSI] screening', len(pending), 'size/skill profiles:', self.path, flush=True)
        original_roots = env._root_states.clone()
        original_dofs = env._dof_state.clone()
        try:
            for kind, owner, profiles in work:
                if not profiles:
                    continue
                self._screen_profiles(kind, owner, profiles, original_roots, original_dofs)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_suffix('.tmp.' + str(os.getpid()))
                with gzip.open(temporary, 'wb') as stream:
                    pickle.dump(self.entries, stream, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temporary, self.path)
        finally:
            env._root_states.copy_(original_roots)
            env._dof_state.copy_(original_dofs)
        print('[size RSI] cache ready:', len(self.entries), 'profiles', flush=True)

    def _screen_profiles(self, kind, owner, profiles, original_roots, original_dofs):
        from isaacgym import gymtorch
        env = self.env
        skill = 'putDown' if kind == 'putDownOnTop' else kind
        lib = env._motion_lib[skill]
        rows = torch.tensor(list(profiles.values()), device=env.device)
        keys = list(profiles)
        sizes = env._box_size[rows, owner]
        offset = env._agent_spawn_offsets[owner] + env._env_origins[rows]
        records = [[] for _ in keys]
        steps = math.ceil(SCREEN['seconds'] / env.sim_params.dt)
        for clip in range(lib.num_motions()):
            frames = int(lib._motion_num_frames[clip])
            ids = torch.full((frames,), clip, device=env.device, dtype=torch.long)
            times = torch.arange(frames, device=env.device) * lib._motion_dt[clip]
            root, rot, dofs, vel, ang, dof_vel, feet = lib.get_motion_state(ids, times)
            if skill in ('sit', 'climb'):
                pos, quat = lib.get_obj_motion_state_single_frame(ids[:1])
                box_pos = pos.expand(frames, -1)
                box_rot = quat.expand(frames, -1)
                # Phase is tied to the reference box, not the random training height.
                top = float(pos[0, 2]) * 2
                reference_size = torch.tensor([.5, .5, top], device=env.device)
                phase = interaction_phase(skill, root, feet[:, -2:], pos[0], reference_size)
            else:
                box_pos, box_rot = lib.get_obj_motion_state(ids, times)
                phase = torch.ones(frames, dtype=torch.bool, device=env.device)
            end = lib.get_obj_motion_state(ids[:1], lib.get_motion_length(ids[:1]))[0] if kind == 'putDownOnTop' else None
            valid = torch.zeros(len(rows), frames, dtype=torch.bool, device=env.device)
            for frame in range(frames):
                env._root_states.copy_(original_roots)
                env._dof_state.copy_(original_dofs)
                env._humanoid_root_states[..., :2] = env._env_origins[:, None, :2] + 15.
                env._humanoid_root_states[..., 0] += torch.arange(env.num_agents, device=env.device)*3
                env._humanoid_root_states[..., 7:13] = 0.
                env._box_states[..., :2] = env._env_origins[:, None, :2] - 15.
                env._box_states[..., 0] -= torch.arange(env.num_objects, device=env.device)*2
                env._box_states[..., 2] = env._box_size[..., 2]/2
                env._box_states[..., 3:7] = torch.tensor([0., 0., 0., 1.], device=env.device)
                env._box_states[..., 7:13] = 0.
                env._set_env_state(rows, torch.full_like(rows, owner), root[frame]+offset,
                                   rot[frame], dofs[frame], vel[frame], ang[frame], dof_vel[frame])
                env._box_states[rows, owner, :3] = box_pos[frame]+offset
                env._box_states[rows, owner, 2] = sizes[:, 2]/2 if skill in ('sit', 'climb') else torch.maximum(box_pos[frame, 2].expand(len(rows)), sizes[:, 2]/2)
                env._box_states[rows, owner, 3:7] = box_rot[frame]
                checked = [owner]
                if end is not None:
                    env._box_states[rows, 2+owner, :2] = end[0, :2]+offset[:, :2]
                    checked.append(2+owner)
                initial_root = env._humanoid_root_states[rows, owner].clone()
                initial_boxes = env._box_states[rows][:, checked].clone()
                env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env._root_states))
                env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env._dof_state))
                targets = env._dof_pos.contiguous().clone()
                env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(targets))
                ok = torch.ones(len(rows), dtype=torch.bool, device=env.device)
                for _ in range(steps):
                    env.gym.simulate(env.sim)
                    env.gym.fetch_results(env.sim, True)
                    env._refresh_sim_tensors()
                    ok &= screen_states(env._humanoid_root_states[rows, owner],
                                        env._box_states[rows][:, checked], initial_root, initial_boxes)
                    ok &= torch.isfinite(env._dof_state.view(env.num_envs, env.num_agents, -1)[rows, owner]).all(-1)
                valid[:, frame] = ok
                if frame % 300 == 299:
                    print('[size RSI]', kind, 'owner', owner, 'clip', clip,
                          'frame', frame+1, '/', frames, flush=True)
            if skill in ('sit', 'climb') and phase.any():
                last = int(phase.nonzero()[-1])
                valid[:, last+1:] = False
            for record, mask in zip(records, valid.cpu().numpy()):
                record.append((mask, phase.cpu().numpy()))
            print('[size RSI]', kind, 'owner', owner, 'clip', clip,
                  'accepted', int(valid.sum()), '/', valid.numel(), flush=True)
        self.entries.update(zip(keys, records))

    def _prepare_sampling(self):
        env = self.env
        self.pools = {}
        self.available = torch.ones(env.num_envs, 2, len(env._skill), dtype=torch.bool, device=env.device)
        for skill in ('sit', 'climb', 'pickUp', 'carryWith', 'putDown'):
            lib = env._motion_lib[skill]
            kinds = ('putDown', 'putDownOnTop') if skill == 'putDown' else (skill,)
            for kind in kinds:
                starts = torch.zeros(env.num_envs, 2, lib.num_motions(), 2, dtype=torch.long)
                counts = torch.zeros_like(starts)
                flat = []
                weights = torch.zeros(env.num_envs, 2, lib.num_motions())
                offsets = {}
                total = 0
                for row in range(env.num_envs):
                    for owner in (0, 1):
                        key = self._key(kind, row, owner)
                        if key not in self.entries:
                            continue
                        if key not in offsets:
                            clip_pools = []
                            for mask, phase in self.entries[key]:
                                valid = torch.from_numpy(mask)
                                late = late_frames(valid, torch.from_numpy(phase))
                                all_frames = valid.nonzero().flatten()
                                if not len(late):
                                    late = all_frames
                                pair = []
                                for frames in (all_frames, late):
                                    pair.append((total, len(frames)))
                                    flat.append(frames)
                                    total += len(frames)
                                clip_pools.append(pair)
                            offsets[key] = torch.tensor(clip_pools)
                        starts[row, owner] = offsets[key][..., 0]
                        counts[row, owner] = offsets[key][..., 1]
                        weights[row, owner] = (counts[row, owner, :, 0] > 0)*lib._motion_weights.cpu()
                self.pools[kind] = (starts.to(env.device), counts.to(env.device),
                                    torch.cat(flat).to(env.device) if flat else torch.empty(0, device=env.device, dtype=torch.long),
                                    weights.to(env.device))
                if kind != 'putDownOnTop':
                    self.available[..., env._skill.index(skill)] = weights.sum(-1).to(env.device) > 0

    def availability(self, rows, owners, templates):
        result = self.available[rows, owners].clone()
        ontop = templates == 4
        if ontop.any():
            weights = self.pools['putDownOnTop'][3]
            result[ontop, self.env._skill.index('putDown')] = weights[rows[ontop], owners[ontop]].sum(-1) > 0
        return result

    def sample(self, skill, rows, owners, templates):
        lib = self.env._motion_lib[skill]
        if skill == 'loco':
            clips = lib.sample_motions(len(rows))
            return clips, lib.sample_time_rsi(clips)
        clips = torch.empty_like(rows)
        times = torch.empty(len(rows), device=rows.device)
        for kind in (('putDown', 'putDownOnTop') if skill == 'putDown' else (skill,)):
            chosen = (templates == 4) if kind == 'putDownOnTop' else (templates != 4) if skill == 'putDown' else torch.ones_like(rows, dtype=torch.bool)
            if not chosen.any():
                continue
            r, a = rows[chosen], owners[chosen]
            starts, counts, frames, weights = self.pools[kind]
            clip = torch.multinomial(weights[r, a], 1).flatten()
            late = (torch.rand(len(r), device=r.device) < self.spec['lateProbability']).long()
            count = counts[r, a, clip, late]
            frame = frames[starts[r, a, clip, late] + (torch.rand(len(r), device=r.device)*count).long()]
            clips[chosen] = clip
            times[chosen] = frame*lib._motion_dt[clip]
        return clips, times
