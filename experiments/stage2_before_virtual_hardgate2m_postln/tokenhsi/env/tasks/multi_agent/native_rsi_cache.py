import gzip
import math
import os
import pickle

import torch

from utils.size_rsi import SCREEN, screen_states, size_key


class NativeRsiCache:

    def _key(self, kind, row, owner):
        return kind, size_key(self.sizes[row][owner])

    def _profiles(self, kind, owner, pending):
        eligible = self.eligible[:, owner, 0] > 0
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
                for kind in self.skills
                for owner in range(env.num_agents)]
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
        strict = self.spec.get('geometryFilter') == 'body_points_v1'
        if strict:
            from utils.stage1_geometry import pose_geometry_valid
        skill = kind
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
            if strict:
                bodies = lib.get_motion_state_max(ids, times)[0]
            box_pos, box_rot = lib.get_obj_motion_state(ids, times)
            phases = torch.ones(len(rows), frames, dtype=torch.bool, device=env.device)
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
                env._box_states[rows, owner, :3] = box_pos[frame]+offset
                env._box_states[rows, owner, 2] = torch.maximum(box_pos[frame, 2].expand(len(rows)), sizes[:, 2]/2)
                env._box_states[rows, owner, 3:7] = box_rot[frame]
                checked = [owner]
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
            for record, mask, row_phase in zip(records, valid.cpu().numpy(), phases.cpu().numpy()):
                record.append((mask, row_phase))
            print('[size RSI]', kind, 'owner', owner, 'clip', clip,
                  'accepted', int(valid.sum()), '/', valid.numel(), flush=True)
        self.entries.update(zip(keys, records))



    def sample(self, skill, rows, owners, templates):
        lib = self.env._motion_lib[skill]
        if skill == 'loco':
            clips = lib.sample_motions(len(rows))
            return clips, lib.sample_time_rsi(clips)
        starts, counts, frames, weights = self.pools[skill]
        clip = torch.multinomial(weights[rows, owners], 1).flatten()
        # Keep both RNG draws used by the original sampler, including lateProbability=0.
        late = (torch.rand(len(rows), device=rows.device) < self.spec['lateProbability']).long()
        count = counts[rows, owners, clip, late]
        frame = frames[starts[rows, owners, clip, late] + (torch.rand(len(rows), device=rows.device)*count).long()]
        return clip, frame*lib._motion_dt[clip]
