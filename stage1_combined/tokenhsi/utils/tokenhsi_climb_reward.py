# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from torch import Tensor

@torch.jit.script
def compute_climb_reward(root_pos, prev_root_pos, object_pos, dt, tar_pos, rigid_body_pos, feet_ids, char_h, valid_radius,
                         climb_vel_penalty, climb_vel_pen_coeff, climb_vel_penalty_thre):
    # type: (Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, float, Tensor, bool, float, float) -> Tensor

    pos_diff = object_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-0.5 * pos_err)

    min_speed = 1.5

    tar_dir = object_pos[..., 0:2] - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = min_speed - tar_dir_speed
    # tar_vel_err = torch.clamp_min(tar_vel_err, 0.0) # constrain vel around the peak value
    vel_reward = torch.exp(-2.0 * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    dist_mask = (pos_err <= valid_radius ** 2)
    pos_reward[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    pos_reward_near = torch.exp(-10 * torch.sum((tar_pos - root_pos) ** 2, dim=-1))

    feet_height_err = (rigid_body_pos[:, feet_ids, -1].mean(dim=1) - (tar_pos[..., 2] - char_h)) ** 2 # height
    feet_height_reward = torch.exp(-50.0 * feet_height_err)
    feet_height_reward[~dist_mask] = 0.0

    reward = 0.0 * pos_reward + 0.2 * vel_reward + 0.5 * pos_reward_near + 0.3 * feet_height_reward

    if climb_vel_penalty:
        thre_tensor = torch.ones_like(valid_radius)
        thre_tensor[pos_err <= 1.5 ** 2] = 1.5
        thre_tensor[pos_err <= valid_radius ** 2] = climb_vel_penalty_thre
        min_speed_penalty = thre_tensor
        root_vel_norm = torch.norm(root_vel, p=2, dim=-1) # torch.abs(root_vel[..., -1]) # only consider Z
        root_vel_norm = torch.clamp_min(root_vel_norm, min_speed_penalty)
        root_vel_err = min_speed_penalty - root_vel_norm
        root_vel_penalty = -1 * climb_vel_pen_coeff * (1 - torch.exp(-2.0 * (root_vel_err * root_vel_err)))
        dist_mask = (pos_err <= 1.5 ** 2)
        root_vel_penalty[~dist_mask] = 0.0
        reward += root_vel_penalty

    return reward
