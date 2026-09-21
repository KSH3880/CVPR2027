"""Schema-4 graph, geometry, reward, packet and network acceptance tests."""
import copy
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from utils.edge_context_spec import HOLDING, AT
from utils.edge_ontop_spec import ON_TOP, graph_packet
from utils.edge_interaction_spec import *
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata, validate_relation_config
from env.tasks.multi_agent.edge_context_reward import edge_context
from env.tasks.multi_agent.edge_interaction_reward import *
from learning.multi_agent.amp_network_builder_ma import RelationEncoder


torch.set_num_threads(1)
ROOT = Path(__file__).parents[1]
ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_distance_edge_context_interaction.yaml').read_text())['env']
CFG, SPEC = ENV['relationReward'], ENV['relationGraph']


def _scene():
    objects = torch.zeros(1, 3, 13)
    objects[..., 6] = 1
    objects[..., 2] = .2
    sizes = torch.tensor([[[.5, .5, .4]] * 3])
    hands = torch.zeros(1, 2, 2, 3)
    feet = torch.zeros(1, 2, 2, 3)
    roots = torch.zeros(1, 2, 3)
    goals = torch.zeros(1, 2, 3)
    return hands, feet, roots, objects, sizes, goals


def test_config_schema_and_sampler_structure():
    validate_relation_config(CFG)
    adjusted = copy.deepcopy(CFG)
    adjusted['climb']['feet_height_tolerance'] = .08
    validate_relation_config(adjusted)
    adjusted['climb']['feet_height_tolerance'] = 0
    with pytest.raises(ValueError):
        validate_relation_config(adjusted)
    validate_sampler(SPEC)
    graph = sample_graph(50000, SPEC, generator=torch.Generator().manual_seed(44))
    assert set(graph.edge_relation[graph.edge_valid].tolist()) == {HOLDING, AT, ON_TOP, SIT, CLIMB}
    assert (graph.edge_valid.sum(-1) >= 2).all() and (graph.edge_valid.sum(-1) <= 4).all()
    for agent in (0, 1):
        owned = graph.edge_valid & (graph.edge_owner == agent)
        assert (owned.sum(-1) <= 2).all()
        has_hold = ((graph.edge_relation == HOLDING) & owned).any(-1)
        has_sit = ((graph.edge_relation == SIT) & owned).any(-1)
        has_climb = ((graph.edge_relation == CLIMB) & owned).any(-1)
        has_at = ((graph.edge_relation == AT) & owned).any(-1)
        has_top = ((graph.edge_relation == ON_TOP) & owned).any(-1)
        pattern = torch.where(~has_hold & has_sit, 1,
            torch.where(~has_hold & has_climb, 2,
            torch.where(has_hold & has_at, 3,
            torch.where(has_hold & has_top, 4,
            torch.where(has_hold & has_climb, 5,
            torch.where(has_hold & has_sit, 6, 0))))))
        actual = torch.stack([(pattern == i).float().mean() for i in range(7)])
        torch.testing.assert_close(actual, torch.tensor(list(SPEC['pattern_probabilities'].values())),
                                   atol=.005, rtol=0)
    # Joint sampling preserves marginals and excludes two simultaneous human consumers.
    interaction = ((graph.edge_relation == SIT) | (graph.edge_relation == CLIMB)) & graph.edge_valid
    assert (interaction.sum(-1) <= 1).all()
    assert not (graph.term_index[((graph.edge_relation == SIT) | (graph.edge_relation == CLIMB)) & graph.edge_valid] >= 0).any()
    for preset in PRESETS:
        validate_graph(sample_graph(4, SPEC, preset=preset))


def test_pre_term_and_required_goal_semantics():
    graph = sample_graph(1, SPEC, preset='hold_climb')
    hold = (graph.edge_relation[0] == HOLDING).nonzero().flatten()[0]
    climb = (graph.edge_relation[0] == CLIMB).nonzero().flatten()[0]
    assert graph.required_goal[0, hold] and graph.required_goal[0, climb]
    assert graph.prereq_mask[0, climb, hold]
    assert graph.term_index[0, hold] == -1
    graph = sample_graph(1, SPEC, preset='at_then_sit')
    at = (graph.edge_relation[0] == AT).nonzero().flatten()[0]
    sit = (graph.edge_relation[0] == SIT).nonzero().flatten()[0]
    assert graph.prereq_mask[0, sit, at]
    assert graph.edge_dst[0, sit] == graph.edge_src[0, at]


def test_sit_target_progress_state_and_current_success():
    graph = sample_graph(1, SPEC, preset='sit_only')
    hands, feet, roots, objects, sizes, goals = _scene()
    edge = (graph.edge_relation[0] == SIT).nonzero().flatten()[0]
    target_z = .2 + CFG['sit']['target_local_offset'][2]
    roots[0, 0] = torch.tensor([0., 0., target_z])
    phi, diag = evaluate_interaction_edges(hands, feet, roots, objects, sizes, goals, graph, CFG, .94)
    assert phi[0, edge] == pytest.approx(1.)
    assert diag['progress'][0, edge] == pytest.approx(1.)
    success = interaction_own_success(phi, diag['z_error'], diag['feet_height_error'], graph, CFG)
    assert success[0, edge]
    roots[0, 0, 2] += .2
    phi, diag = evaluate_interaction_edges(hands, feet, roots, objects, sizes, goals, graph, CFG, .94)
    assert not interaction_own_success(phi, diag['z_error'], diag['feet_height_error'], graph, CFG)[0, edge]


def test_climb_root_state_feet_validation_and_no_feet_dense_term():
    graph = sample_graph(1, SPEC, preset='climb_only')
    hands, feet, roots, objects, sizes, goals = _scene()
    edge = (graph.edge_relation[0] == CLIMB).nonzero().flatten()[0]
    roots[0, 0] = torch.tensor([0., 0., 1.34])  # box top .4 + char_h .94
    feet[0, 0, :, 2] = .4
    phi, diag = evaluate_interaction_edges(hands, feet, roots, objects, sizes, goals, graph, CFG, .94)
    assert phi[0, edge] == pytest.approx(1.)
    assert diag['z_surface'][0, edge] == pytest.approx(.4)
    assert interaction_own_success(phi, diag['z_error'], diag['feet_height_error'], graph, CFG)[0, edge]
    feet[0, 0, :, 2] = .451
    other_phi, other_diag = evaluate_interaction_edges(hands, feet, roots, objects, sizes, goals, graph, CFG, .94)
    torch.testing.assert_close(other_phi, phi)  # feet do not change the state reward
    assert not interaction_own_success(other_phi, other_diag['z_error'], other_diag['feet_height_error'], graph, CFG)[0, edge]


def test_saturation_is_current_and_hold_sit_stays_simultaneous():
    graph = sample_graph(1, SPEC, preset='hold_sit')
    runtime = InteractionContextRuntime(1, graph, CFG, 'cpu')
    phi = torch.zeros(1, 4); progress = torch.zeros_like(phi); z = torch.zeros_like(phi); feet = torch.zeros_like(phi)
    sit = (graph.edge_relation[0] == SIT).nonzero().flatten()[0]
    hold = (graph.edge_relation[0] == HOLDING).nonzero().flatten()[0]
    phi[0, sit] = 1
    out = runtime.step(phi, progress, z, feet)
    assert out['total'][0, sit] == pytest.approx(.6)
    assert out['total'][0, hold] == 0
    assert not runtime.done[0, 0]
    phi.zero_()
    assert not runtime.step(phi, progress, z, feet)['reward_saturated'].any()


def test_explicit_three_edge_inference_and_checkpoint_separation():
    spec = yaml.safe_load((ROOT / 'data/cfg/multi_agent/graphs/edge_interaction_three_edges.yaml').read_text())
    graph = compile_interaction_graph(spec, 2, 3)
    assert graph.edge_relation.tolist() == [HOLDING, CLIMB, ON_TOP]
    meta = checkpoint_metadata(CFG)
    check_checkpoint_metadata({'relation_metadata': meta}, meta)
    old = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_distance_edge_context_ontop.yaml').read_text())['env']['relationReward']
    with pytest.raises(ValueError):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(old)}, meta)


def test_relation_embedding_packet_and_forward():
    graph = compose_graph(torch.tensor([[6, 0], [5, 0], [6, 4]]),
                          torch.tensor([[2, 2], [2, 2], [1, 2]]))
    net = RelationEncoder([223, 30, 1], 2, 3, 16, 2, 2, 32,
        lambda size: nn.Sequential(nn.Linear(size, 16), nn.ReLU()),
        observation_mode='clean_scene', kinematic_size=7,
        relation_bias_mode='edge_mlp', gta_cfg={'enable': True},
        relation_reward_mode=INTERACTION_CONTEXT_MODE, relation_graph_spec=SPEC)
    assert net.edge_encoder.relation_embed.num_embeddings == 11
    with torch.no_grad():
        net.edge_encoder.bias_projection.normal_(std=.1)
    width = 224 * 2 + 30 * 3
    obs = torch.randn(3, width + 7 * 7 + 7 * 4)
    obs[:, width:width + 49].reshape(3, 7, 7)[..., 3:7] = torch.tensor([0., 0., 0., 1.])
    obs[:, -28:] = graph_packet(graph, edge_context(torch.rand(3, 4), graph))
    out = net(obs)
    assert out.shape == (3, 2, 16) and torch.isfinite(out).all()
    out.square().sum().backward()
    assert net.edge_encoder.relation_embed.weight.grad[SIT].abs().sum() > 0
    assert net.edge_encoder.relation_embed.weight.grad[CLIMB].abs().sum() > 0
