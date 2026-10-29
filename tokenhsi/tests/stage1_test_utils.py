"""Shared Stage1 CPU fixtures; no simulator imports."""

from pathlib import Path
import torch
import yaml
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from utils.edge_scenario_spec import compose_canonical_graph
from utils.task_role_spec import task_graph_packet

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "data/cfg/multi_agent/stage1.yaml").read_text())["env"]
TRAIN = yaml.safe_load((ROOT / "data/cfg/train/rlg/stage1.yaml").read_text())


def network():
    torch.set_num_threads(1)
    builder = AMPMultiAgentBuilder()
    builder.load(TRAIN["params"]["network"])
    net = builder.build(
        "amp",
        actions_num=32,
        input_shape=(634,),
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
        relation_reward_mode=CFG["relationReward"]["mode"],
        relation_graph_spec=CFG["relationGraph"],
        device="cpu",
    )
    for enc in (net.actor_encoder, net.critic_encoder):
        enc.gta_diagnostics_first_forward = False
    return net


def all_pairs():
    return compose_canonical_graph(
        torch.cartesian_prod(torch.arange(1, 5), torch.arange(1, 5))
    )


def observations(graph):
    n = graph.edge_valid.shape[0]
    kin = torch.randn(n, 8, 7)
    kin[..., 3:] = torch.nn.functional.normalize(kin[..., 3:], dim=-1)
    return torch.cat(
        (torch.randn(n, 568), kin.flatten(1), task_graph_packet(graph, n)), -1
    )


def fused(enc, packet, types=None):
    return enc.context_fusion(
        packet,
        enc.edge_encoder,
        enc.entity_types if types is None else types,
        enc.build_relation_bias(),
    )[0]
