from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from env.tasks.multi_agent.edge_ontop_task import SampledOnTopTaskMixin
from env.tasks.multi_agent.scene_features import build_gta_pose_records, scenario_neutral_targets
from learning.multi_agent.amp_network_builder_ma import RelationEncoder
from utils.edge_context_spec import HOLDING, AT
from utils.edge_interaction_spec import SIT, CLIMB
from utils.edge_ontop_spec import ON_TOP
from utils.edge_scenario_spec import (TEMPLATES, classify_templates, compose_graph,
    paired_placement_success, sample_graph, valid_binding_pair, valid_bindings)
from utils.edge_stage1_spec import (STAGE1_CONTEXT_MODE, semantic_graph_packet,
    validate_sampler)
from utils.relation_task_spec import (checkpoint_metadata,
    check_checkpoint_metadata, validate_relation_config)


ROOT = Path(__file__).resolve().parents[1]
ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_no_climb.yaml').read_text())['env']


def test_config_motion_pool_and_checkpoint_contract():
    validate_relation_config(ENV['relationReward'])
    validate_sampler(ENV['relationGraph'])
    assert ENV['relationReward']['schema_version'] == 8
    assert 'climb' not in ENV['relationReward']
    assert ENV['skill'] == ['loco', 'sit', 'omomo', 'pickUp', 'carryWith', 'putDown']
    assert ENV['box']['reset']['groundStandaloneSitClimb'] is True
    motions = yaml.safe_load((ROOT / 'data/dataset_loco_sit_carry_no_climb.yaml').read_text())['motions']
    assert list(motions) == ['loco', 'sit', 'omomo', 'pickUp', 'carryWith', 'putDown']
    assert checkpoint_metadata(ENV['relationReward'])['relation_taxonomy'] == {
        'holding': 6, 'at': 7, 'ontop': 8, 'sit': 9}
    old = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_distance_stage1_climb_rsi.yaml').read_text())['env']
    with pytest.raises(ValueError, match='schema mismatch'):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(old['relationReward'])},
                                  checkpoint_metadata(ENV['relationReward']))


def test_inactive_scenario_targets_are_env_local_not_global_zero():
    origins = torch.tensor([[0., 0., 0.], [320., -240., 0.]])
    ids = torch.tensor([1])
    targets = scenario_neutral_targets(origins, ids, 2)
    human_pos = origins[ids, None].expand(-1, 2, -1)
    object_pos = origins[ids, None].expand(-1, 3, -1)
    human_rot = torch.tensor([0., 0., 0., 1.]).expand(1, 2, 4)
    object_rot = torch.tensor([0., 0., 0., 1.]).expand(1, 3, 4)
    poses = build_gta_pose_records(
        human_pos, human_rot, object_pos, object_rot, targets, origins[ids])
    assert poses[..., :3].abs().max() == 0


def test_sampler_templates_random_bindings_and_no_climb():
    graph = sample_graph(4096, ENV['relationGraph'], generator=torch.Generator().manual_seed(5))
    assert set(graph.edge_relation[graph.edge_valid].tolist()) == {HOLDING, AT, ON_TOP, SIT}
    assert not (graph.edge_relation[graph.edge_valid] == CLIMB).any()
    assert (graph.edge_valid.sum(-1) <= 4).all()
    assert (graph.required_goal.sum(-1) == 2).all()
    slots = torch.arange(4096).repeat_interleave(2)
    agents = torch.arange(2).repeat(4096)
    template = classify_templates(graph, slots, agents)
    assert set(template.tolist()) == set(range(4))
    for pair in (("HOLDING_AT", "SIT"), ("HOLDING_AT", "HOLDING_ON_TOP"),
                 ("HOLDING_ON_TOP", "SIT"), ("HOLDING_ON_TOP", "HOLDING_ON_TOP")):
        assert valid_bindings(pair)


def test_invalid_filter_keeps_intended_shared_examples():
    assert valid_binding_pair(('HOLDING_AT', 'SIT'), ((0, 0), (0,)))
    assert valid_binding_pair(('HOLDING_AT', 'HOLDING_ON_TOP'), ((0, 0), (1, 0)))
    assert valid_binding_pair(('HOLDING_ON_TOP', 'SIT'), ((0, 1), (0,)))
    assert valid_binding_pair(('HOLDING_ON_TOP', 'HOLDING_ON_TOP'), ((0, 1), (1, 2)))
    assert not valid_binding_pair(('HOLDING', 'HOLDING'), ((0,), (0,)))
    assert not valid_binding_pair(('HOLDING_ON_TOP', 'HOLDING_ON_TOP'), ((0, 1), (1, 0)))
    assert not valid_binding_pair(('SIT', 'HOLDING_ON_TOP'), ((1,), (0, 1)))


def test_bound_third_object_near_owner_is_not_treated_as_free_distractor():
    task = SampledOnTopTaskMixin()
    task.device = 'cpu'; task.num_agents = 2; task.num_objects = 3
    task._scenario_no_climb = True; task._primitive_stage1 = False
    task._box_min_agent_dist = 1.; task._reset_ref_slots = {}
    task.relation_runtime = SimpleNamespace(graph=compose_graph(
        [('HOLDING', 'HOLDING')], [((2,), (0,))]))
    task._box_states = torch.zeros(1, 3, 13)
    task._box_states[0, :, :3] = torch.tensor([[2., 0., .2], [-2., 0., .2], [0., 0., .2]])
    task._box_size = torch.full((1, 3, 3), .4)
    task._humanoid_root_states = torch.zeros(1, 2, 13)
    task._humanoid_root_states[0, 1, 1] = 3.
    task._tar_pos = torch.zeros(1, 2, 3)
    task._logical_box_values = lambda values, ids=None: values if ids is None else values[ids]
    assert not task._ontop_scene_infeasible(torch.tensor([0])).any()


def test_paired_placement_saturation_is_current_and_raw_success_stays_false():
    graph = compose_graph([('HOLDING_AT', 'SIT')], [((0, 0), (0,))])
    hold = ((graph.edge_relation == HOLDING) & graph.edge_valid).nonzero()[0, 1]
    at = ((graph.edge_relation == AT) & graph.edge_valid).nonzero()[0, 1]
    sit = ((graph.edge_relation == SIT) & graph.edge_valid).nonzero()[0, 1]
    phi = torch.zeros(1, 4); progress = torch.full((1, 4), .4); z = torch.zeros(1, 4)
    phi[0, at] = phi[0, sit] = 1.
    runtime = Stage1ContextRuntime(1, graph, ENV['relationReward'], 'cpu')
    out = runtime.step(phi, progress, z)
    assert not out['own_success'][0, hold]
    assert out['paired_placement_success'][0, hold]
    assert out['reward_saturated'][0, hold]
    assert out['total'][0, hold] == pytest.approx(.6)
    assert out['local_task_reward'][0].tolist() == pytest.approx([1.2, .6])
    assert out['agent_task_reward'][0].tolist() == pytest.approx([1.14, .66])
    phi[0, at] = 0.
    out = runtime.step(phi, progress, z)
    assert not out['reward_saturated'][0, hold]
    assert runtime.done[0].tolist() == [False, True]


def test_scenario_task_reward_mixes_point_nine_self_point_one_teammate():
    graph = compose_graph([('HOLDING', 'SIT')], [((0,), (1,))])
    phi = torch.tensor([[1., .2, 0., 0.]])
    progress = torch.zeros_like(phi)
    runtime = Stage1ContextRuntime(1, graph, ENV['relationReward'], 'cpu')
    out = runtime.step(phi, progress, torch.zeros_like(phi))
    local = out['local_task_reward'][0]
    assert local.tolist() == pytest.approx([.6, .04])
    assert out['agent_task_reward'][0].tolist() == pytest.approx([
        .9 * .6 + .1 * .04, .9 * .04 + .1 * .6])


def test_semantic_only_network_has_no_context_encoder():
    graph = sample_graph(3, ENV['relationGraph'], generator=torch.Generator().manual_seed(9))
    net = RelationEncoder([223, 30, 1], 2, 3, 16, 2, 2, 32,
        lambda size: nn.Sequential(nn.Linear(size, 16), nn.ReLU()),
        observation_mode='clean_scene', kinematic_size=7,
        relation_bias_mode='edge_mlp', gta_cfg={'enable': True},
        relation_reward_mode=STAGE1_CONTEXT_MODE,
        relation_graph_spec=ENV['relationGraph'])
    assert net.suffix_width == 20
    assert not any('context_encoder' in name for name, _ in net.named_parameters())
    with torch.no_grad():
        net.edge_encoder.bias_projection.normal_(std=.1)
    width = 224 * 2 + 30 * 3
    obs = torch.randn(3, width + 7 * 7 + 20)
    obs[:, width:width + 49].reshape(3, 7, 7)[..., 3:7] = torch.tensor([0., 0., 0., 1.])
    obs[:, -20:] = semantic_graph_packet(graph, 3)
    out = net(obs)
    assert out.shape == (3, 2, 16) and torch.isfinite(out).all()
    out.square().sum().backward()
    assert net.edge_encoder.relation_embed.weight.grad[SIT].abs().sum() > 0
