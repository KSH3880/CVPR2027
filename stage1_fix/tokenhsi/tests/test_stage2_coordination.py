from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml

from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder, EdgeEncoder
from learning.multi_agent.coordination_head import GroundedEdgeCoordination
from learning.multi_agent.stage2_transfer import transfer_stage1_weights
from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from env.tasks.multi_agent.edge_ontop_task import SampledOnTopTaskMixin
from utils.edge_ontop_spec import expand_graph, permute_graph
from utils.edge_stage1_spec import semantic_graph_packet
from utils.edge_stage1_spec import compile_stage1_graph
from utils.edge_stage2_spec import (STAGE2_CONTEXT_MODE, classify_family, compile_graph,
                                    sample_graph, validate_graph, validate_stage2_config)
from utils.edge_scenario_spec import classify_templates, scenario_templates
from utils.relation_task_spec import check_checkpoint_metadata, checkpoint_metadata


ROOT = Path(__file__).resolve().parents[1]
ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_stage2_coordination.yaml').read_text())['env']
PLANE_ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_stage2_coordination_sit_plane.yaml').read_text())['env']
STAGE1 = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_plane.yaml').read_text())['env']


def explicit_graph(agents):
    edges = [
        dict(owner=0, source='H_0', target='O_0', relation='HOLDING', required_goal=False),
        dict(owner=0, source='O_0', target='G_0', relation='AT', required_goal=True),
        dict(owner=1, source='H_1', target='O_0', relation='CLIMB', required_goal=True),
    ]
    if agents >= 3:
        edges += [
            dict(owner=2, source='H_2', target='O_2', relation='HOLDING', required_goal=False),
            dict(owner=2, source='O_2', target='O_0', relation='ON_TOP', required_goal=True),
        ]
    if agents >= 4:
        edges.append(dict(owner=3, source='H_3', target='O_3', relation='SIT', required_goal=True))
    return dict(mode='stage2_explicit', semantic_only=True, edge_capacity=2 * agents, edges=edges)


def network(agents=2, objects=3, graph=None, stage2=True):
    path = 'amp_ma_stage2_coordination.yaml' if stage2 else 'amp_ma_carry_relation.yaml'
    params = yaml.safe_load((ROOT / 'data/cfg/train/rlg' / path).read_text())['params']['network']
    graph = graph or (ENV if stage2 else STAGE1)['relationGraph']
    builder = AMPMultiAgentBuilder()
    builder.load(params)
    edges = len((compile_graph if stage2 else compile_stage1_graph)(
        graph, agents, objects).ids)
    width = 223 * agents + 30 * objects + agents + 7 * (2 * agents + objects) + 5 * edges
    return builder.build('stage2_test', actions_num=32, input_shape=(width,),
        amp_input_shape=(1290,), value_size=1, num_agents=agents, num_objects=objects,
        humanoid_obs_size=230, object_obs_size=39, goal_obs_size=6,
        observation_mode='clean_scene', scene_entity_sizes=[223, 30, 1],
        scene_kinematic_size=7, scene_arena_scale=5.,
        relation_reward_mode=STAGE2_CONTEXT_MODE if stage2 else STAGE1['relationReward']['mode'],
        relation_graph_spec=graph, device='cpu')


def observation(graph, agents, objects, batch=2):
    nodes = 223 * agents + 30 * objects + agents
    count = 2 * agents + objects
    edges = graph.edge_valid.shape[-1]
    obs = torch.randn(batch, nodes + count * 7 + edges * 5)
    pose = obs[:, nodes:nodes + count * 7].view(batch, count, 7)
    pose.zero_()
    pose[..., 6] = 1.
    obs[:, -edges * 5:] = semantic_graph_packet(graph, batch)
    return obs


class Wrapper(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.a2c_network = network


def test_sampler_and_explicit_graph_contract():
    graph = sample_graph(400, ENV['relationGraph'])
    validate_graph(graph)
    assert graph.edge_valid.sum(-1).min() >= 2
    assert graph.edge_valid.sum(-1).max() <= 4
    for preset in ('place_climb', 'place_sit', 'place_stack'):
        validate_graph(sample_graph(8, ENV['relationGraph'], preset=preset))
    for agents in (2, 3, 4):
        validate_graph(compile_graph(explicit_graph(agents), agents, agents))
    for name, agents, valid_edges in (
            ('stage2_three_agent_shared.yaml', 3, 5),
            ('stage2_four_agent_pairs.yaml', 4, 6)):
        spec = yaml.safe_load((ROOT / 'data/cfg/multi_agent/graphs' / name).read_text())
        graph = compile_graph(spec, agents, agents)
        assert int(graph.edge_valid.sum()) == valid_edges
        assert len(scenario_templates(spec)) == len(ENV['templateRsi'])
    assert 'climb' in checkpoint_metadata(ENV['relationReward'])['relation_taxonomy']
    bad = explicit_graph(3)
    bad['edges'][3]['target'] = 'O_0'
    with pytest.raises(ValueError, match='Joint HOLDING'):
        compile_graph(bad, 3, 3)


def test_stage2_plane_34_reward_and_sampling():
    reward = PLANE_ENV['relationReward']
    sampler = PLANE_ENV['relationGraph']
    validate_stage2_config(reward)
    graph = sample_graph(500, sampler, generator=torch.Generator().manual_seed(19))
    fractions = torch.bincount(classify_family(graph), minlength=4).float() / 500
    torch.testing.assert_close(fractions, torch.tensor([.2, .2, .2, .4]), atol=.07, rtol=0)

    independent = sample_graph(300, sampler, preset='independent',
                               generator=torch.Generator().manual_seed(20))
    env_ids = torch.arange(300).repeat_interleave(2)
    agents = torch.arange(2).repeat(300)
    templates = classify_templates(independent, env_ids, agents, True)
    fractions = torch.bincount(templates, minlength=5).float() / 600
    torch.testing.assert_close(fractions, torch.tensor([.1, .1, .1, .35, .35]),
                               atol=.06, rtol=0)

    sit_graph = sample_graph(1, sampler, preset='place_sit')
    phi = torch.ones(1, 4)
    z_error = torch.zeros_like(phi)
    feet_error = torch.zeros_like(phi)
    region_error = torch.full_like(phi, -1.)
    phi[0, 2] = .5
    assert interaction_own_success(phi, z_error, feet_error, sit_graph, reward,
                                   region_error)[0, 2]
    assert not interaction_own_success(phi, z_error, feet_error, sit_graph,
                                       ENV['relationReward'], region_error)[0, 2]

    phi[0, 2] = 1.
    runtime = Stage1ContextRuntime.__new__(Stage1ContextRuntime)
    runtime.graph = sit_graph
    runtime.config = reward
    runtime.phi = torch.zeros_like(phi)
    runtime.own_success = torch.zeros_like(phi, dtype=torch.bool)
    runtime.achieved = torch.zeros_like(phi, dtype=torch.bool)
    runtime.done = torch.zeros(1, 2, dtype=torch.bool)
    result = runtime.step(phi, phi, z_error, feet_error, region_error)
    torch.testing.assert_close(result['agent_task_reward'], torch.tensor([[1.14, .6]]))

    old_checkpoint = {'relation_metadata': checkpoint_metadata(ENV['relationReward'])}
    with pytest.raises(ValueError, match='relation reward config differs'):
        check_checkpoint_metadata(old_checkpoint, checkpoint_metadata(reward))


def test_stage2_near_start_only_moves_independent_targets():
    graph = sample_graph(50, PLANE_ENV['relationGraph'],
                         generator=torch.Generator().manual_seed(24))
    family = classify_family(graph)
    assert (family == 0).any() and (family != 0).any()
    reward = deepcopy(PLANE_ENV['relationReward'])
    reward['independent_training']['near_start']['probability_start'] = 1.
    boxes = torch.zeros(50, 3, 13)
    boxes[:, 1, 0] = 3.
    boxes[:, 2, 1] = 4.
    task = SimpleNamespace(_relation_cfg=reward, device='cpu', num_agents=2,
        num_objects=3, relation_runtime=SimpleNamespace(graph=graph),
        _logical_box_order=torch.arange(3).expand(50, -1),
        _box_states=boxes, _tar_pos=torch.full((50, 2, 3), 9.),
        _reset_ref_slots={}, _hard_skill_training_step=0)
    task._logical_box_values = lambda values, ids: values[ids]
    original_boxes = boxes.clone()
    original_goals = task._tar_pos.clone()
    SampledOnTopTaskMixin._configure_hard_skill_near_starts(task, torch.arange(50))
    torch.testing.assert_close(task._box_states[family != 0], original_boxes[family != 0])
    torch.testing.assert_close(task._tar_pos[family != 0], original_goals[family != 0])
    assert ((task._box_states[family == 0] != original_boxes[family == 0]).any() or
            (task._tar_pos[family == 0] != original_goals[family == 0]).any())


def test_attention_permutation_padding_and_empty():
    torch.manual_seed(3)
    graph = sample_graph(3, ENV['relationGraph'])
    humans = torch.randn(3, 2, 64)
    nodes = torch.randn(3, 7, 64)
    packet = semantic_graph_packet(graph, 3)
    edge_encoder = EdgeEncoder(4, 2, num_relation_types=11)
    types = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    module = GroundedEdgeCoordination()
    original = module(humans, nodes, packet, edge_encoder, types)
    order = torch.rand(3, 4).argsort(-1)
    shuffled = permute_graph(graph, order)
    torch.testing.assert_close(module(humans, nodes,
        semantic_graph_packet(shuffled, 3), edge_encoder, types), original)
    padded = torch.cat((packet.view(3, 4, 5), torch.zeros(3, 2, 5)), 1).flatten(1)
    torch.testing.assert_close(module(humans, nodes, padded, edge_encoder, types), original)
    assert torch.count_nonzero(module(humans, nodes, torch.zeros_like(packet),
                                      edge_encoder, types)) == 0


def test_transfer_identity_gradients_and_variable_agents():
    torch.manual_seed(7)
    source = network(stage2=False)
    target = network()
    checkpoint = dict(model=Wrapper(source).state_dict(),
        relation_metadata=checkpoint_metadata(STAGE1['relationReward']), epoch=123)
    report = transfer_stage1_weights(Wrapper(target), checkpoint)
    assert report['source_epoch'] == 123
    assert len(report['new_tensors']) == 8
    graph = sample_graph(2, ENV['relationGraph'], preset='place_climb')
    obs = observation(graph, 2, 3)
    source.eval(); target.eval()
    torch.testing.assert_close(source.eval_actor(obs)[0], target.eval_actor(obs)[0],
                               atol=1e-5, rtol=1e-5)
    assert all(not p.requires_grad for p in target.actor_encoder.parameters())
    optimizer = torch.optim.Adam((p for p in target.parameters() if p.requires_grad), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        target.eval_actor(obs)[0].square().mean().backward()
        optimizer.step()
    assert target.coordination.grounding[0].weight.grad is not None
    assert target.coordination.grounding[0].weight.grad.abs().sum() > 0
    weights = target.state_dict()
    for agents in (3, 4):
        graph_spec = explicit_graph(agents)
        expanded = network(agents, agents, graph_spec)
        expanded.load_state_dict(weights, strict=True)
        graph = expand_graph(compile_graph(graph_spec, agents, agents), 2)
        obs = observation(graph, agents, agents)
        assert expanded.eval_actor(obs)[0].shape == (2 * agents, 32)
