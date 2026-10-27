"""Verbatim reference Table reward/reset kernels; original source stays read-only."""
from typing import Tuple
import torch
from isaacgym.torch_utils import quat_rotate
from utils import torch_utils

@torch.jit.script
def fall_termination_intermediate_process(object_pos, object_bbox_height, platform_pos, tar_platform_pos):
    # type: (Tensor, Tensor, Tensor, Tensor,) -> Tensor
    fall_threshold = object_bbox_height.clone() / 2 + 0.1
    obj_fall = object_pos[..., -1] < fall_threshold
    disable_mask = torch.logical_or(
        torch.sum((object_pos[..., 0:2] - platform_pos[..., 0:2]) ** 2, dim=-1) < 0.5 ** 2,
        torch.sum((object_pos[..., 0:2] - tar_platform_pos[..., 0:2]) ** 2, dim=-1) < 0.5 ** 2,
    )
    obj_fall[disable_mask] = False
    return obj_fall

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights,
                           box_states, enable_obj_fall_termination, object_bbox_height, platform_pos, tar_platform_pos,
                           human_root_lin_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, bool, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(torch.ones_like(fall_height), fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

        if enable_obj_fall_termination:
            obj_has_fallen = fall_termination_intermediate_process(box_states[..., 0:3], object_bbox_height, platform_pos, tar_platform_pos)
            terminated = torch.where(obj_has_fallen, torch.ones_like(reset_buf), terminated)

        human_or_obj_flied = torch.logical_or(box_states[..., 2] >= 3.0, rigid_body_pos[:, 0, 2] >= 3.0)
        terminated = torch.where(human_or_obj_flied, torch.ones_like(reset_buf), terminated) # fly termination

        obj_far_from_human = torch.norm(box_states[..., 0:2] - rigid_body_pos[:, 0, 0:2], p=2, dim=-1) >= 3.0
        obj_is_init_state = torch.norm(box_states[..., 0:2] - platform_pos[..., 0:2], p=2, dim=-1) <= 1.0
        obj_far_from_human[obj_is_init_state] = False
        terminated = torch.where(obj_far_from_human, torch.ones_like(reset_buf), terminated)
    
        human_or_obj_speed_too_large = torch.logical_or(
            torch.norm(human_root_lin_vel, p=2, dim=-1) > 5.0,
            torch.norm(box_states[..., 7:10], p=2, dim=-1) > 5.0,
        )
        terminated = torch.where(human_or_obj_speed_too_large, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

@torch.jit.script
def compute_carry_reward(box_pos, prev_box_pos, tar_box_pos, dt, tar_vel, box_size, only_vel_reward, box_vel_penalty, box_vel_pen_coeff, box_vel_penalty_thre,
                         box_rot, default_up_facings,
                         leave_coeff, leave_init_pos_rwd, init_box_pos, root_pos, prev_root_pos):
    # type: (Tensor, Tensor, Tensor, float, float, Tensor, bool, bool, float, float, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor) -> Tensor
    
    # this reward encourages the character to carry the box to a target position

    pos_diff = tar_box_pos - box_pos # xyz
    pos_err_xy = torch.sum(pos_diff[..., 0:2] ** 2, dim=-1)
    pos_err_xyz = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward_far = torch.exp(-0.5 * pos_err_xy)
    pos_reward_near = torch.exp(-10.0 * pos_err_xyz)

    min_speed = tar_vel

    tar_dir = tar_box_pos[..., 0:2] - box_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)

    delta_root_pos = box_pos - prev_box_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = min_speed - tar_dir_speed
    # tar_vel_err = torch.clamp_min(tar_vel_err, 0.0) # constrain vel around the peak value
    vel_reward = torch.exp(-2.0 * (tar_vel_err * tar_vel_err)) # use a smaller coeff (-5.0 >> -2.0)
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    height_mask = box_pos[..., 2] <= (box_size[..., 2] / 2 + 0.1) # avoid learning to kick the box
    pos_reward_far[height_mask] = 0.0
    vel_reward[height_mask] = 0.0

    obj_facing_dir = quat_rotate(box_rot, default_up_facings)
    tar_facing_dir = default_up_facings.clone()
    dir_err = torch.sum(obj_facing_dir[..., 0:3] * tar_facing_dir[..., 0:3], dim=-1) # xyz

    human2obj = torch.norm(root_pos[..., 0:2] - box_pos[..., 0:2], p=2, dim=-1)

    dist_mask = torch.logical_and(
        pos_err_xy < 0.7 ** 2, # because some objects are very large, we need to increase the shreshold
        human2obj < 0.7,  # avoid the humanoid walk away from the obj after task finish
    )

    pos_reward_far[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    dir_reward = torch.clamp_min(dir_err, 0.0)
    dir_reward[~dist_mask] = 0.0

    ultimate_near_reward = pos_reward_near * dir_reward

    if only_vel_reward:
        reward = 0.4 * vel_reward + 0.4 * ultimate_near_reward
    else:
        reward = 0.2 * pos_reward_far + 0.2 * vel_reward + 0.4 * ultimate_near_reward

    reward[~(human2obj < 0.7)] = 0.0 # avoid the humanoid walk away from the obj after task finish

    if box_vel_penalty:
        min_speed_penalty = box_vel_penalty_thre
        root_vel_norm = torch.norm(root_vel, p=2, dim=-1)
        root_vel_norm = torch.clamp_min(root_vel_norm, min_speed_penalty)
        root_vel_err = min_speed_penalty - root_vel_norm
        root_vel_penalty = -1 * box_vel_pen_coeff * (1 - torch.exp(-2.0 * (root_vel_err * root_vel_err)))
        reward += root_vel_penalty

        delta_human_root_pos = root_pos - prev_root_pos
        human_root_vel = delta_human_root_pos / dt

        hu_root_vel_norm = torch.norm(human_root_vel, p=2, dim=-1)
        hu_root_vel_norm = torch.clamp_min(hu_root_vel_norm, min_speed_penalty)
        hu_root_vel_err = min_speed_penalty - hu_root_vel_norm
        hu_root_vel_penalty = -1 * box_vel_pen_coeff * (1 - torch.exp(-2.0 * (hu_root_vel_err * hu_root_vel_err)))
        reward += hu_root_vel_penalty

    if leave_init_pos_rwd:
        xy_dist = torch.norm(init_box_pos[..., 0:2] - box_pos[..., 0:2], p=2, dim=-1)
        # leave_rwd = torch.clamp_max(xy_dist ** 2, 1.0)
        leave_rwd = torch.clamp_max(xy_dist * (1 / 3), 1.0) # 应该走的越远 奖励值越大
        leave_rwd[height_mask] = 0.0
        leave_rwd[~(human2obj < 0.7)] = 0.0
        leave_rwd[dist_mask] = 1.0
        reward += leave_coeff * leave_rwd

    return reward

@torch.jit.script
def compute_putdown_reward(box_pos, tar_pos, box_rot, box_bbox_height, box_bps,
                           humanoid_rigid_body_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    object_pos_exp = torch.broadcast_to(box_pos.unsqueeze(-2), (box_pos.shape[0], box_bps.shape[1], box_pos.shape[1])) # (num_envs, 3) >> (num_envs, 8, 3)
    object_rot_exp = torch.broadcast_to(box_rot.unsqueeze(-2), (box_rot.shape[0], box_bps.shape[1], box_rot.shape[1])) # (num_envs, 4) >> (num_envs, 8, 4)
    object_bps_world_space = quat_rotate(object_rot_exp.reshape(-1, 4), box_bps.reshape(-1, 3)) + object_pos_exp.reshape(-1, 3) # (num_envs * 8, 3)
    object_bps_world_space = object_bps_world_space.reshape(-1, 8, 3)

    target_value = (tar_pos[:, 2] - box_bbox_height / 2)

    bottom_anchor = 0.25 * (object_bps_world_space[:, 0] + object_bps_world_space[:, 1] + object_bps_world_space[:, 2] + object_bps_world_space[:, 3])
    reward = (torch.abs((bottom_anchor[:, -1] - target_value)) <= 0.05) * 1.0

    reward = torch.clamp_max(reward, 1.0)

    pos_err_xy = torch.sum((tar_pos[..., :2] - box_pos[..., :2]) ** 2, dim=-1)
    reward[(pos_err_xy > 0.3 ** 2)] = 0.0 # use a large range

    human2obj = torch.norm(humanoid_rigid_body_pos[:, 0][..., 0:2] - box_pos[..., 0:2], p=2, dim=-1)
    reward[~(human2obj < 0.7)] = 0.0
    
    return reward

@torch.jit.script
def compute_walk_reward(root_pos, prev_root_pos, box_pos, dt, tar_vel, only_vel_reward):
    # type: (Tensor, Tensor, Tensor, float, float, bool) -> Tensor

    # this reward encourages the character to walk towards the box and stay close to it

    pos_diff = box_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-0.5 * pos_err)

    min_speed = tar_vel

    tar_dir = box_pos[..., 0:2] - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = min_speed - tar_dir_speed
    # tar_vel_err = torch.clamp_min(tar_vel_err, 0.0) # constrain vel around the peak value
    vel_reward = torch.exp(-5.0 * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    dist_mask = pos_err < 1.0 ** 2 # because some objects are very large, we need to increase the shreshold
    pos_reward[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    if only_vel_reward:
        reward = 1.0 * vel_reward
    else:
        reward = 0.5 * pos_reward + 0.5 * vel_reward
    return reward

@torch.jit.script
def compute_handheld_reward(humanoid_rigid_body_pos, box_pos, hands_ids, only_height):
    # type: (Tensor, Tensor, Tensor, bool) -> Tensor
    if only_height:
        hands2box_pos_err = torch.sum((humanoid_rigid_body_pos[:, hands_ids, 2] - box_pos[:, 2].unsqueeze(-1)) ** 2, dim=-1) # height
    else:
        hands2box_pos_err = torch.sum((humanoid_rigid_body_pos[:, hands_ids].mean(dim=1) - box_pos) ** 2, dim=-1) # xyz
    hands2box = torch.exp(-5.0 * hands2box_pos_err)

    root_pos = humanoid_rigid_body_pos[:, 0, :]
    box2human = torch.sum((box_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1) # 2d
    hands2box[box2human > 0.7 ** 2] = 0 # disable this reward when the box is not close to the humanoid

    return hands2box

@torch.jit.script
def compute_straight_upperbody_penalty(box_pos, box_size, box_init_pos, box_tar_pos, 
                                      root_pos, root_rot, head_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    intermediate_mask = torch.logical_and(
        torch.norm(box_pos[..., 0:2] - box_init_pos[..., 0:2], p=2, dim=-1) >= 0.7,
        torch.norm(box_pos[..., 0:2] - box_tar_pos[..., 0:2], p=2, dim=-1) >= 0.7,
    )

    height_mask = box_pos[..., 2] >= box_size[..., 2] / 2 + 0.1

    allow_mask = intermediate_mask & height_mask

    head2root = torch.nn.functional.normalize(head_pos - root_pos, dim=-1)

    humanoid_default_facings = torch.zeros_like(head2root)
    humanoid_default_facings[..., 0] = 1.0

    humanoid_rotated_facings = quat_rotate(
        torch_utils.calc_heading_quat(root_rot), humanoid_default_facings
    )

    dir_err = torch.sum(head2root * humanoid_rotated_facings, dim=-1) # xyz

    min_target = -0.2579215 # cos((90+15)°)
    penalty = min_target - dir_err
    penalty = torch.clamp_min(penalty, 0.0)
    penalty = -1 * (1 - torch.exp(-100.0 * (penalty * penalty)))

    penalty[~allow_mask] = 0.0
   
    return penalty
