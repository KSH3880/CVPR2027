from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from learning.multi_agent.before_network import BeforeBuilder
from learning.multi_agent.before_network import freeze_actor, transfer_stage1
from utils.before_spec import validate_env, checkpoint_contract, sample_graph
from utils.before_spec import check_checkpoint
from utils.task_role_spec import task_graph_packet
from utils.relation_task_spec import checkpoint_metadata

ROOT = Path(__file__).resolve().parents[2]
BASE = 'stage2_before'
DEFAULT = 'stage2_before_nonemask_rsi433_near50'


def config(name, folder='multi_agent'):
    return yaml.safe_load((ROOT/f'tokenhsi/data/cfg/{folder}/{name}.yaml').read_text())


def build(name):
    torch.manual_seed(26)
    env = config(name)['env']
    builder = BeforeBuilder()
    builder.load(config(name, 'train/rlg')['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(1268,), amp_input_shape=(1320,),
        value_size=1, num_agents=4, num_objects=8, observation_mode='clean_scene',
        scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7, scene_arena_scale=16.,
        relation_reward_mode=env['relationReward']['mode'], relation_graph_spec=env['relationGraph'], device='cpu')
    net.actor_encoder.relation_graph_spec['shuffle_token_order'] = False
    net.coordination.shuffle_human_order = False
    net.critic_coordination.shuffle_human_order = False
    return net


def observations():
    env = config(DEFAULT)['env']
    graph, _, _ = sample_graph(torch.full((2, 4, 3), .4), env['box']['build']['taskSizeRanges'], 'before_stack')
    poses = torch.zeros(2, 16, 7); poses[..., 6] = 1
    return torch.cat((torch.randn(2, 1136), poses.flatten(1), task_graph_packet(graph, 2)), -1)
