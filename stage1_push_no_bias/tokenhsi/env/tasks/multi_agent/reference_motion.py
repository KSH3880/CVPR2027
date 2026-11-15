"""ReferenceMotionMixin: shared fixed Stage1/BEFORE environment behavior."""
import os
import yaml
import torch
from env.tasks.humanoid import dof_to_obs
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import quat_rotate, quat_mul
from utils import torch_utils
from utils.edge_scenario_spec import classify_templates

class ReferenceMotionMixin:
    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def _load_motion(self, motion_file):
        assert self._dof_offsets[-1] == self.num_dof
        ext = os.path.splitext(motion_file)[1]
        if ext == ".yaml":
            with open(os.path.join(os.getcwd(), motion_file), "r") as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)
            self._skill_categories = list(motion_config["motions"].keys())
            self._motion_lib = {}
            for skill in self._skill_categories:
                self._motion_lib[skill] = MotionLib(
                    motion_file=motion_file,
                    skill=skill,
                    dof_body_ids=self._dof_body_ids,
                    dof_offsets=self._dof_offsets,
                    key_body_ids=self._key_body_ids.cpu().numpy(),
                    device=self.device,
                )
        else:
            raise NotImplementedError
        return

    def _amp_family(self, env_ids=None, slot_agent=None):
        from utils.unified_training import family_from_templates

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if slot_agent is None:
            slot_agent = torch.arange(self.num_agents, device=self.device).repeat(
                len(env_ids)
            )
            env_ids = env_ids.repeat_interleave(self.num_agents)
        return family_from_templates(
            classify_templates(
                self.relation_runtime.graph, env_ids, slot_agent, with_climb=True
            )
        )

    def _fetch_conditioned_amp_demo(self, num_samples):
        from utils.unified_training import EXPERT_PROBS, append_family

        family_probs = (0.65, 0.1, 0.25)
        family = torch.multinomial(
            torch.tensor(family_probs, device=self.device),
            num_samples,
            replacement=True,
        )
        weights = torch.tensor(EXPERT_PROBS, device=self.device)[family]
        skills = torch.multinomial(weights, 1).squeeze(-1)
        result = torch.empty(num_samples, self.get_num_amp_obs(), device=self.device)
        for uid, name in enumerate(self.cfg["env"]["skill"]):
            indices = (skills == uid).nonzero(as_tuple=False).flatten()
            if not len(indices):
                continue
            lib = self._motion_lib[name]
            motions = lib.sample_motions(len(indices))
            truncate = self.dt * (self._num_amp_obs_steps - 1)
            times = lib.sample_time(motions, truncate_time=truncate) + truncate
            motion_obs = self.build_amp_obs_demo(motions, times, lib).reshape(
                len(indices), self._num_amp_obs_steps, -1
            )
            labels = family[indices, None].expand(-1, self._num_amp_obs_steps)
            result[indices] = append_family(motion_obs, labels).flatten(1)
        return result

    def fetch_amp_obs_demo(self, num_samples):
        return self._fetch_conditioned_amp_demo(num_samples)

    def build_amp_obs_demo(self, motion_ids, motion_times0, motion_lib):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps
        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
            motion_lib.get_motion_state(motion_ids, motion_times)
        )
        amp_obs_demo = build_amp_observations(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_pos,
            self._local_root_obs,
            self._root_height_obs,
            self._dof_obs_size,
            self._dof_offsets,
        )
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros(
            (num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float32,
        )
        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)
        if self._reset_default_slots is not None:
            self._init_amp_obs_default(*self._reset_default_slots)
        for sk_name in self._skill:
            if self._reset_ref_slots.get(sk_name) is not None:
                self._init_amp_obs_ref(
                    self._reset_ref_slots[sk_name],
                    self._reset_ref_motion_ids[sk_name],
                    self._reset_ref_motion_times[sk_name],
                    sk_name,
                )
        return

    def _init_amp_obs_default(self, slot_env, slot_agent):
        curr = self._curr_amp_obs_buf[slot_env, slot_agent].unsqueeze(-2)
        self._hist_amp_obs_buf[slot_env, slot_agent] = curr
        return

    def _init_amp_obs_ref(self, slots, motion_ids, motion_times, skill_name):
        slot_env, slot_agent = slots
        dt = self.dt
        motion_ids = torch.tile(
            motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1]
        )
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (
            torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1
        )
        motion_times = motion_times + time_steps
        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
            self._motion_lib[skill_name].get_motion_state(motion_ids, motion_times)
        )
        amp_obs_demo = build_amp_observations(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_pos,
            self._local_root_obs,
            self._root_height_obs,
            self._dof_obs_size,
            self._dof_offsets,
        )
        from utils.unified_training import append_family

        labels = self._amp_family(slot_env, slot_agent).repeat_interleave(
            self._num_amp_obs_steps - 1
        )
        amp_obs_demo = append_family(amp_obs_demo, labels, getattr(self, "_amp_num_families", 3))
        self._hist_amp_obs_buf[slot_env, slot_agent] = amp_obs_demo.view(
            slot_env.shape[0], self._num_amp_obs_steps - 1, self._num_amp_obs_per_step
        )
        return

    def _update_hist_amp_obs(self, env_ids=None):
        for i in reversed(range(self._amp_obs_buf.shape[2] - 1)):
            self._amp_obs_buf[:, :, i + 1] = self._amp_obs_buf[:, :, i]
        return

    def _compute_amp_observations(self, env_ids=None):
        N, M, nb, nd = (self.num_envs, self.num_agents, self.num_bodies, self.num_dof)
        if env_ids is None:
            B = N * M
            body_pos = self._rigid_body_pos.reshape(B, nb, 3)
            body_rot = self._rigid_body_rot.reshape(B, nb, 4)
            body_vel = self._rigid_body_vel.reshape(B, nb, 3)
            body_ang_vel = self._rigid_body_ang_vel.reshape(B, nb, 3)
            dof_pos = self._dof_pos.reshape(B, nd)
            dof_vel = self._dof_vel.reshape(B, nd)
            obs = build_amp_observations(
                body_pos[:, 0, :],
                body_rot[:, 0, :],
                body_vel[:, 0, :],
                body_ang_vel[:, 0, :],
                dof_pos,
                dof_vel,
                body_pos[:, self._key_body_ids, :],
                self._local_root_obs,
                self._root_height_obs,
                self._dof_obs_size,
                self._dof_offsets,
            )
            from utils.unified_training import append_family

            obs = append_family(obs, self._amp_family(), getattr(self, "_amp_num_families", 3))
            self._curr_amp_obs_buf[:] = obs.view(N, M, self._num_amp_obs_per_step)
        else:
            K = env_ids.shape[0]
            B = K * M
            kin = self._kinematic_humanoid_rigid_body_states[env_ids].reshape(B, nb, 13)
            dof_pos = self._dof_pos[env_ids].reshape(B, nd)
            dof_vel = self._dof_vel[env_ids].reshape(B, nd)
            obs = build_amp_observations(
                kin[:, 0, 0:3],
                kin[:, 0, 3:7],
                kin[:, 0, 7:10],
                kin[:, 0, 10:13],
                dof_pos,
                dof_vel,
                kin[:, self._key_body_ids, 0:3],
                self._local_root_obs,
                self._root_height_obs,
                self._dof_obs_size,
                self._dof_offsets,
            )
            from utils.unified_training import append_family

            obs = append_family(obs, self._amp_family(env_ids), getattr(self, "_amp_num_families", 3))
            self._curr_amp_obs_buf[env_ids] = obs.view(K, M, self._num_amp_obs_per_step)
        return


@torch.jit.script
def build_amp_observations(
    root_pos,
    root_rot,
    root_vel,
    root_ang_vel,
    dof_pos,
    dof_vel,
    key_body_pos,
    local_root_obs,
    root_height_obs,
    dof_obs_size,
    dof_offsets,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)
    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat(
        (
            root_h_obs,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1,
    )
    return obs

