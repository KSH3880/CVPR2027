"""TokenHSI five-token integration adapted from the intern delivery.

Only task observation/registration and indexed reset plumbing are retained.
PUSH task semantics live in humanoid_push_kd.py; no delivered reward is used.
"""
from enum import Enum
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from utils import torch_utils
from env.tasks.multi_task.humanoid_traj_sit_carry_climb import HumanoidTrajSitCarryClimb, compute_location_observations


class HumanoidTrajSitCarryClimbPush(HumanoidTrajSitCarryClimb):

    class TaskUID(Enum):
        traj = 0
        sit = 1
        carry = 2
        climb = 3
        push = 4

    def register_task_climb_pre_init(self, cfg):
        super().register_task_climb_pre_init(cfg)
        self.register_task_push_pre_init(cfg)
        return

    def register_task_climb_post_init(self, cfg):
        super().register_task_climb_post_init(cfg)
        self.register_task_push_post_init(cfg)
        return

    def _box_corners(self, size):
        sgn = torch.tensor([[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1],
                            [1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]], device=size.device, dtype=torch.float)
        return sgn.unsqueeze(0) * size.unsqueeze(1) / 2

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        root_states = self._humanoid_root_states[env_ids]
        task_mask = self._task_mask[env_ids]

        traj_samples = self._fetch_traj_samples(env_ids)
        old = compute_location_observations(root_states, traj_samples,
                                            self._sit_tar_pos[env_ids], self._sit_object_states[env_ids],
                                            self._sit_obj_lib._every_env_object_bps[env_ids], self._sit_obj_lib._every_env_object_facings[env_ids],
                                            self._box_states[env_ids], self._box_lib._box_bps[env_ids], self._box_tar_pos[env_ids],
                                            self._climb_object_states[env_ids], self._climb_obj_lib._every_env_object_bps[env_ids], self._climb_tar_pos[env_ids],
                                            task_mask, task_mask, False)
        push = compute_push_observations(root_states, self._push_box_states[env_ids], self._push_box_bps[env_ids], self._push_tar_pos[env_ids])
        obs = torch.cat([old, push], dim=-1)

        if self._enable_apply_mask_on_task_obs:
            mask = task_mask.float() @ self.get_multi_task_info()["each_subtask_obs_mask"].float()
            obs = obs * mask

        if self._enable_task_mask_obs:
            obs = torch.cat([obs, task_mask.float()], dim=-1)
        return obs

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = {}
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}
        if (len(env_ids) > 0):
            self._reset_task_indicator(env_ids)
            self._reset_ref_state_init(env_ids)
            self._reset_task_traj(env_ids)
            self._reset_task_sit(env_ids)
            self._reset_task_carry(env_ids)
            self._reset_task_climb(env_ids)
            self._reset_task_push(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
            self._init_amp_obs(env_ids)
        return

    def _reset_env_tensors(self, env_ids):
        # a single indexed root-state write: with several writes before one simulate call, not all of them land
        # (the sit chair stayed at its build pose, inside the PUSH box)
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

        if self._enable_IET:
            self._IET_step_buf[env_ids] = 0
            self._IET_triggered_buf[env_ids] = 0

        if self._is_eval:
            self._success_buf[env_ids] = 0
            self._precision_buf[env_ids] = float('Inf')
            traj_env_mask = self._task_indicator[env_ids] == HumanoidTrajSitCarryClimbPush.TaskUID["traj"].value
            self._precision_buf[env_ids[traj_env_mask]] = 0 # not Inf

        ids = [self._humanoid_actor_ids[env_ids], self._sit_object_actor_ids[env_ids], self._box_actor_ids[env_ids],
               self._climb_object_actor_ids[env_ids], self._push_box_actor_ids[env_ids], self._push_marker_actor_ids[env_ids]]
        if self._carry_reset_random_height:
            ids += [self._platform_actor_ids[env_ids], self._tar_platform_actor_ids[env_ids]]
        ids = torch.cat(ids, dim=0).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(ids), len(ids))
        hum = self._humanoid_actor_ids[env_ids].to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(hum), len(hum))
        return


@torch.jit.script
def compute_push_observations(root_states, box_states, box_bps, tar_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_states[:, 3:7])

    box_pos = box_states[:, 0:3]
    box_rot = box_states[:, 3:7]
    local_box_vel = quat_rotate(heading_rot, box_states[:, 7:10])
    local_box_ang_vel = quat_rotate(heading_rot, box_states[:, 10:13])
    local_box_pos = quat_rotate(heading_rot, box_pos - root_pos)
    local_box_rot_obs = torch_utils.quat_to_tan_norm(quat_mul(heading_rot, box_rot))

    B, N = box_bps.shape[0], box_bps.shape[1]
    box_pos_exp = box_pos.unsqueeze(1).expand(B, N, 3).reshape(-1, 3)
    box_rot_exp = box_rot.unsqueeze(1).expand(B, N, 4).reshape(-1, 4)
    bps_world = quat_rotate(box_rot_exp, box_bps.reshape(-1, 3)) + box_pos_exp
    heading_exp = heading_rot.unsqueeze(1).expand(B, N, 4).reshape(-1, 4)
    root_exp = root_pos.unsqueeze(1).expand(B, N, 3).reshape(-1, 3)
    bps_local = quat_rotate(heading_exp, bps_world - root_exp).reshape(B, N * 3)

    local_tar_pos = quat_rotate(heading_rot, tar_pos - root_pos)
    return torch.cat([local_box_vel, local_box_ang_vel, local_box_pos, local_box_rot_obs, bps_local, local_tar_pos], dim=-1)
