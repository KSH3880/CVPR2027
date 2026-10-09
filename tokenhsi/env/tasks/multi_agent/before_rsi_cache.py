import math

import torch

from env.tasks.multi_agent.size_rsi_cache import SizeRsiCache
from utils.size_rsi import SCREEN, interaction_phase, screen_states, size_key


class BeforeSizeRsiCache(SizeRsiCache):
    def __init__(self, env):
        self.families = env._before_families.cpu().tolist()
        super().__init__(env)

    def objects(self, rows, owner):
        family = self.env._before_families[rows]
        source = torch.full_like(rows, owner)
        support = torch.full_like(rows, 2+owner)
        if owner == 1:
            source = torch.where((family == 1) | (family == 2), 0, source)
            support = torch.where(family == 3, 0, support)
        return source, support

    def _key(self, kind, row, owner):
        family = self.families[row]
        source = 0 if owner == 1 and family in (1, 2) else owner
        support = 0 if owner == 1 and family == 3 else 2+owner
        key = (kind, size_key(self.sizes[row][source]))
        return key + (size_key(self.sizes[row][support]),) if kind == 'putDownOnTop' else key

    def _screen_profiles(self, kind, owner, profiles, original_roots, original_dofs):
        from isaacgym import gymtorch
        env = self.env
        skill = 'putDown' if kind == 'putDownOnTop' else kind
        lib = env._motion_lib[skill]
        rows = torch.tensor(list(profiles.values()), device=env.device)
        keys = list(profiles)
        source, support = self.objects(rows, owner)
        sizes = env._box_size[rows, source]
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
                env._box_states[rows, source, :3] = box_pos[frame]+offset
                env._box_states[rows, source, 2] = sizes[:, 2]/2 if skill in ('sit', 'climb') else torch.maximum(box_pos[frame, 2].expand(len(rows)), sizes[:, 2]/2)
                env._box_states[rows, source, 3:7] = box_rot[frame]
                checked = source[:, None]
                if end is not None:
                    env._box_states[rows, support, :2] = end[0, :2]+offset[:, :2]
                    checked = torch.stack((source, support), -1)
                initial_root = env._humanoid_root_states[rows, owner].clone()
                initial_boxes = env._box_states[rows[:, None], checked].clone()
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
                        env._box_states[rows[:, None], checked], initial_root, initial_boxes)
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
