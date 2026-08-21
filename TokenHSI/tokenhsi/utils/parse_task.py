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

from env.tasks.humanoid import Humanoid
from env.tasks.humanoid_amp import HumanoidAMP

######### basic interaction skills
from env.tasks.basic_interaction_skills.humanoid_traj import HumanoidTraj
from env.tasks.basic_interaction_skills.humanoid_sit import HumanoidSit
from env.tasks.basic_interaction_skills.humanoid_carry import HumanoidCarry
from env.tasks.basic_interaction_skills.humanoid_climb import HumanoidClimb
#########

######### multi task
from tokenhsi.env.tasks.multi_task.humanoid_traj_sit_carry_climb import HumanoidTrajSitCarryClimb
#########

######### policy adaptation

# skill composition
from env.tasks.comp_interaction_skills.humanoid_comp_sit_carry import HumanoidCompSitCarry
from env.tasks.comp_interaction_skills.humanoid_comp_climb_carry import HumanoidCompClimbCarry
from env.tasks.comp_interaction_skills.humanoid_comp_traj_carry import HumanoidCompTrajCarry

# object shape variation
from env.tasks.adapt_interaction_skills.humanoid_adapt_carry_box2objs import HumanoidAdaptCarryBox2Objs

# terrain shape variation
from env.tasks.adapt_interaction_skills.humanoid_adapt_traj_ground2terrain import HumanoidAdaptTrajGround2Terrain
from env.tasks.adapt_interaction_skills.humanoid_adapt_carry_ground2terrain import HumanoidAdaptCarryGround2Terrain

# long-horizon task completion
from tokenhsi.env.tasks.longterm_task_completion.humanoid_longterm_4basicskills import HumanoidLongTerm4BasicSkills

#########

from env.tasks.vec_task_wrappers import VecTaskPythonWrapper

from isaacgym import rlgpu

import json
import numpy as np


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]")

def parse_task(args, cfg, cfg_train, sim_params):

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    try:
        task = eval(args.task)(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless)
    except NameError as e:
        print(e)
        warn_task_name()
    env = VecTaskPythonWrapper(task, rl_device, cfg_train.get("clip_observations", np.inf), cfg_train.get("clip_actions", 1.0))

    return task, env
