import math
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from utils.task_role_spec import CARRY_DISTILL_VARIANT

ROOT = Path(__file__).resolve().parents[1]
SA = "stage2"


class Wrapper(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.a2c_network = network


def configs(suffix=SA):
    env = (
        ROOT
        / "data/cfg/multi_agent"
        / (suffix + ".yaml")
    )
    train = (
        ROOT
        / "data/cfg/train/rlg"
        / (suffix + ".yaml")
    )
    return yaml.safe_load(env.read_text())["env"], yaml.safe_load(train.read_text())


def source_reward():
    reward = deepcopy(configs()[0]["relationReward"])
    reward.update(
        mode="state_relation_edge_stage1_v1",
        schema_version=9,
        stage1_variant=CARRY_DISTILL_VARIANT,
    )
    reward.pop("joint_carry")
    reward["success"]["saturation"] = "paired_placement_current_success"
    return reward


def network(suffix=SA, source=False):
    env, train = configs(suffix)
    params = train["params"]["network"]
    if source:
        params.pop("coordination")
    builder = AMPMultiAgentBuilder()
    builder.load(params)
    net = builder.build(
        "amp",
        actions_num=32,
        input_shape=(1268 if "grasp" in suffix else 634,),
        amp_input_shape=(1320,),
        value_size=1,
        num_agents=2,
        num_objects=4,
        humanoid_obs_size=230,
        object_obs_size=39,
        goal_obs_size=6,
        observation_mode="clean_scene",
        scene_entity_sizes=[223, 30, 1],
        scene_kinematic_size=7,
        scene_arena_scale=5.0,
        relation_reward_mode=source_reward()["mode"]
        if source
        else env["relationReward"]["mode"],
        relation_graph_spec=env["relationGraph"],
        device="cpu",
    )
    return Wrapper(net)


def build_network(stage2=False):
    return network(source=not stage2).a2c_network


def joint_observations(n=8):
    obs = scene()[0].expand(n, -1).clone()
    obs[:, :446] = torch.randn(n, 446) * 0.1
    obs[:, -10:] = torch.tensor([1, 2, 0, 2, 6, 1, 2, 1, 2, 6])
    return obs


def scene():
    m, o = 2, 4
    node_end = m * 223 + o * 30 + m
    pose_end = node_end + (2 * m + o) * 7
    obs = torch.zeros(1, pose_end + m * 5)
    objects = obs[:, m * 223 : m * 223 + o * 30].reshape(1, o, 30)
    objects[..., 6:] = torch.tensor(
        [
            -0.2,
            -0.4,
            -0.2,
            -0.2,
            -0.4,
            0.2,
            -0.2,
            0.4,
            -0.2,
            -0.2,
            0.4,
            0.2,
            0.2,
            -0.4,
            -0.2,
            0.2,
            -0.4,
            0.2,
            0.2,
            0.4,
            -0.2,
            0.2,
            0.4,
            0.2,
        ]
    )
    objects[..., 5] = 2
    poses = obs[:, node_end:pose_end].reshape(1, 8, 7)
    poses[..., 6] = 1
    poses[:, 3, 5:7] = math.sqrt(0.5)
    obs[:, pose_end:] = torch.tensor([1, 2, 0, 3, 6, 1, 3, 1, 5, 4])
    return obs, node_end, pose_end
