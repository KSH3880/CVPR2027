import gzip
import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import torch

from env.tasks.multi_agent.size_rsi_cache import SizeRsiCache
from utils.size_rsi import SCREEN, interaction_phase, late_frames, screen_states, size_key


class JointPushSoloCache(SizeRsiCache):
    @staticmethod
    def source(kind, owner):
        return owner + (4 if kind == 'sit' else 8 if kind == 'climb' else 0)

    def _key(self, kind, row, owner):
        source = size_key(self.sizes[row][self.source(kind, owner)])
        return (kind, source, size_key(self.sizes[row][12+owner])) if kind == 'putDownOnTop' else (kind, source)

    def _profiles(self, kind, owner, pending):
        representatives = {}
        for row in range(self.env.num_envs):
            key = self._key(kind, row, owner)
            if key not in self.entries and key not in pending:
                representatives.setdefault(key, row)
        pending.update(representatives)
        return representatives

    def _build_missing(self):
        legacy_name = self.path.stem
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
        directory = self.env.cfg['env']['stage2JointPush']['solo_rsi_cache_directory']
        self.path = Path(directory)/(legacy_name+'_'+digest+'.gz')
        if self.path.exists():
            with gzip.open(self.path, 'rb') as stream:
                self.entries.update(pickle.load(stream))
        env = self.env
        pending = set()
        work = [(kind, owner, self._profiles(kind, owner, pending))
                for kind in ('sit','climb','pickUp','carryWith','putDown','putDownOnTop')
                for owner in range(env.num_agents)]
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if not pending:
            return
        print('[jointpush solo RSI] screening',len(pending),'profiles:',self.path,flush=True)
        roots, dofs = env._root_states.clone(), env._dof_state.clone()
        try:
            for kind, owner, profiles in work:
                if not profiles:
                    continue
                self._screen_profiles(kind,owner,profiles,roots,dofs)
                self.path.parent.mkdir(parents=True,exist_ok=True)
                temporary = self.path.with_suffix('.tmp.'+str(os.getpid()))
                with gzip.open(temporary,'wb') as stream:
                    pickle.dump(self.entries,stream,protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temporary,self.path)
        finally:
            from isaacgym import gymtorch
            env._root_states.copy_(roots)
            env._dof_state.copy_(dofs)
            env.gym.set_actor_root_state_tensor(env.sim,gymtorch.unwrap_tensor(env._root_states))
            env.gym.set_dof_state_tensor(env.sim,gymtorch.unwrap_tensor(env._dof_state))
            env.gym.set_dof_position_target_tensor(env.sim,gymtorch.unwrap_tensor(env._dof_pos.contiguous()))
            env._refresh_sim_tensors()

    def _screen_profiles(self, kind, owner, profiles, original_roots, original_dofs):
        from isaacgym import gymtorch
        env = self.env
        strict = self.spec.get('geometryFilter') == 'body_points_v1'
        if strict:
            from utils.stage1_v2 import pose_geometry_valid, sampled_phase
        skill = 'putDown' if kind == 'putDownOnTop' else kind
        lib = env._motion_lib[skill]
        rows = torch.tensor(list(profiles.values()), device=env.device)
        keys = list(profiles)
        source = self.source(kind, owner)
        support = 12+owner
        sizes = env._box_size[rows, source]
        offset = env._agent_spawn_offsets[owner] + env._env_origins[rows]
        records = [[] for _ in keys]
        steps = math.ceil(SCREEN['seconds'] / env.sim_params.dt)
        for clip in range(lib.num_motions()):
            frames = int(lib._motion_num_frames[clip])
            ids = torch.full((frames,), clip, device=env.device, dtype=torch.long)
            times = torch.arange(frames, device=env.device) * lib._motion_dt[clip]
            root, rot, dofs, vel, ang, dof_vel, feet = lib.get_motion_state(ids, times)
            if strict:
                bodies = lib.get_motion_state_max(ids, times)[0]
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
            phases = (sampled_phase(skill, root, feet[:, -2:], box_pos[0, :2], sizes)
                      if strict and skill in ('sit', 'climb')
                      else phase[None].expand(len(rows), -1))
            end = lib.get_obj_motion_state(ids[:1], lib.get_motion_length(ids[:1]))[0] if kind == 'putDownOnTop' else None
            valid = torch.zeros(len(rows), frames, dtype=torch.bool, device=env.device)
            for frame in range(frames):
                env._root_states.copy_(original_roots)
                env._dof_state.copy_(original_dofs)
                env._humanoid_root_states[..., :2] = env._env_origins[:, None, :2] + 15.
                env._humanoid_root_states[..., 0] += torch.arange(env.num_agents, device=env.device)*3
                env._humanoid_root_states[..., 7:13] = 0.
                env._box_states[..., :2] = env._env_origins[:, None, :2] - 15.
                env._box_states[..., 0] -= torch.arange(env.num_physical_objects, device=env.device)*2
                env._box_states[..., 2] = env._box_size[..., 2]/2
                env._box_states[..., 3:7] = torch.tensor([0., 0., 0., 1.], device=env.device)
                env._box_states[..., 7:13] = 0.
                env._set_env_state(rows, torch.full_like(rows, owner), root[frame]+offset,
                                   rot[frame], dofs[frame], vel[frame], ang[frame], dof_vel[frame])
                env._box_states[rows, source, :3] = box_pos[frame]+offset
                env._box_states[rows, source, 2] = sizes[:, 2]/2 if skill in ('sit', 'climb') else torch.maximum(box_pos[frame, 2].expand(len(rows)), sizes[:, 2]/2)
                env._box_states[rows, source, 3:7] = box_rot[frame]
                checked = [source]
                if end is not None:
                    env._box_states[rows, support, :2] = end[0, :2]+offset[:, :2]
                    checked.append(support)
                initial_root = env._humanoid_root_states[rows, owner].clone()
                initial_boxes = env._box_states[rows][:, checked].clone()
                env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env._root_states))
                env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env._dof_state))
                targets = env._dof_pos.contiguous().clone()
                env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(targets))
                ok = torch.ones(len(rows), dtype=torch.bool, device=env.device)
                if strict:
                    ok &= pose_geometry_valid(bodies[frame]+offset[:, None],
                        feet[frame, -2:]+offset[:, None], initial_boxes,
                        env._box_size[rows][:, checked])
                for _ in range(steps):
                    env.gym.simulate(env.sim)
                    env.gym.fetch_results(env.sim, True)
                    env._refresh_sim_tensors()
                    ok &= screen_states(env._humanoid_root_states[rows, owner],
                                        env._box_states[rows][:, checked], initial_root, initial_boxes)
                    ok &= torch.isfinite(env._dof_state.view(env.num_envs, env.num_agents, -1)[rows, owner]).all(-1)
                    if strict:
                        body = env._rigid_body_pos[rows, owner]
                        ok &= pose_geometry_valid(body, body[:, env._key_body_ids[-2:]],
                            env._box_states[rows][:, checked], env._box_size[rows][:, checked])
                        # The cache is also used at progress=0, where runtime fall
                        # termination has a grace period. Screen fallen poses now.
                        low = body[..., 2] < env._termination_heights
                        low[:, env._contact_body_ids] = False
                        contact = env._contact_forces[rows, owner].clone()
                        contact[:, env._contact_body_ids] = 0
                        fallen = low.any(-1) & (contact.abs().sum(-1)>.1).any(-1)
                        ok &= ~fallen
                valid[:, frame] = ok
                if frame % 300 == 299:
                    print('[size RSI]', kind, 'owner', owner, 'clip', clip,
                          'frame', frame+1, '/', frames, flush=True)
            if strict and skill in ('sit', 'climb'):
                index = torch.arange(frames, device=env.device)[None]
                last = torch.where(phases, index, -1).amax(-1)
                valid &= (last[:, None] < 0) | (index <= last[:, None])
            elif skill in ('sit', 'climb') and phase.any():
                last = int(phase.nonzero()[-1])
                valid[:, last+1:] = False
            for record, mask, row_phase in zip(records, valid.cpu().numpy(), phases.cpu().numpy()):
                record.append((mask, row_phase))
            print('[size RSI]', kind, 'owner', owner, 'clip', clip,
                  'accepted', int(valid.sum()), '/', valid.numel(), flush=True)
        self.entries.update(zip(keys, records))

    def _prepare_sampling(self):
        env = self.env
        self.pools = {}
        self.available = torch.ones(env.num_envs, env.num_agents, len(env._skill), dtype=torch.bool, device=env.device)
        for skill in ('sit', 'climb', 'pickUp', 'carryWith', 'putDown'):
            lib = env._motion_lib[skill]
            kinds = ('putDown', 'putDownOnTop') if skill == 'putDown' else (skill,)
            for kind in kinds:
                starts = torch.zeros(env.num_envs, env.num_agents, lib.num_motions(), 2, dtype=torch.long)
                counts = torch.zeros_like(starts)
                flat = []
                weights = torch.zeros(env.num_envs, env.num_agents, lib.num_motions())
                offsets = {}
                total = 0
                for row in range(env.num_envs):
                    for owner in range(env.num_agents):
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



class LazyJointPushSoloCache:
    def __init__(self, task):
        self.task = task
        self.cache = None

    def _load(self):
        if self.cache is None:
            self.cache = JointPushSoloCache(self.task)
        return self.cache

    def availability(self, *args):
        return self._load().availability(*args)

    def sample(self, *args):
        return self._load().sample(*args)
