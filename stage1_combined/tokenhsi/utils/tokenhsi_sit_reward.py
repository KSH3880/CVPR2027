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
from isaacgym.torch_utils import get_euler_xyz

@torch.jit.script
def compute_sit_reward(root_pos, prev_root_pos, root_rot, prev_root_rot, 
                       object_root_pos, tar_pos, tar_speed, dt,
                       sit_vel_penalty, sit_vel_pen_coeff, sit_vel_penalty_thre, sit_ang_vel_pen_coeff, sit_ang_vel_penalty_thre,):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, bool, float, float, float, float,) -> Tensor

    d_obj2human_xy = torch.sum((object_root_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1)
    reward_far_pos = torch.exp(-0.5 * d_obj2human_xy)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir = object_root_pos[..., 0:2] - root_pos[..., 0:2] # d* is a horizontal unit vector pointing from the root to the object's location
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    reward_far_vel = torch.exp(-2.0 * tar_vel_err * tar_vel_err)

    reward_far_final = 0.0 * reward_far_pos + 1.0 * reward_far_vel
    dist_mask = (d_obj2human_xy <= 0.5 ** 2)
    reward_far_final[dist_mask] = 1.0

    # when humanoid is close to the object
    reward_near = torch.exp(-10.0 * torch.sum((tar_pos - root_pos) ** 2, dim=-1))

    reward = 0.7 * reward_near + 0.3 * reward_far_final

    if sit_vel_penalty:
        min_speed_penalty = sit_vel_penalty_thre
        root_vel_norm = torch.norm(root_vel, p=2, dim=-1)
        root_vel_norm = torch.clamp_min(root_vel_norm, min_speed_penalty)
        root_vel_err = min_speed_penalty - root_vel_norm
        root_vel_penalty = -1 * sit_vel_pen_coeff * (1 - torch.exp(-2.0 * (root_vel_err * root_vel_err)))
        dist_mask = (d_obj2human_xy <= 1.5 ** 2)
        root_vel_penalty[~dist_mask] = 0.0
        reward += root_vel_penalty

        root_z_ang_vel = torch.abs((get_euler_xyz(root_rot)[2] - get_euler_xyz(prev_root_rot)[2]) / dt)
        root_z_ang_vel = torch.clamp_min(root_z_ang_vel, sit_ang_vel_penalty_thre)
        root_z_ang_vel_err = sit_ang_vel_penalty_thre - root_z_ang_vel
        root_z_ang_vel_penalty = -1 * sit_ang_vel_pen_coeff * (1 - torch.exp(-0.5 * (root_z_ang_vel_err ** 2)))
        root_z_ang_vel_penalty[~dist_mask] = 0.0
        reward += root_z_ang_vel_penalty

    return reward
