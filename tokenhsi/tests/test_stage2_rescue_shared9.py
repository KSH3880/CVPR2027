"""Shared9 resource assignment, immutable sizes, all-loco RSI and AMP transfer."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml

from utils.stage2_shared_spec import (PAIRS, scene_pools, sample_role_sizes,
    box_order, initial_body_collisions)
from utils.edge_stage2_spec import (sample_graph, classify_family, validate_graph,
    configure_evaluation_graph, STAGE2_RESCUE_SAMPLER, STAGE2_GENERAL_RESCUE_SAMPLER)
from utils.edge_context_spec import HOLDING
from utils.edge_ontop_spec import ON_TOP, permute_graph
from utils.edge_stage1_spec import semantic_graph_packet
from utils.edge_scenario_spec import agent_object_indices, classify_templates
from utils.unified_training import validate_unified_env
from learning.multi_agent.stage2_transfer import transfer_stage1_weights
from utils.relation_task_spec import checkpoint_metadata
from test_stage2_rescue_klclimb50 import network, observation, Wrapper, SOURCE

ROOT = Path(__file__).resolve().parents[1]
ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_stage2_rescue_shared9.yaml').read_text())['env']
SPEC = ENV['relationGraph']


@pytest.mark.parametrize('agents,objects', [(2, 4), (3, 3), (4, 6), (5, 8)])
def test_klclimb_evaluation_expands_shared9_without_changing_reward_or_training(agents, objects):
    env = deepcopy(ENV)
    env.update(numAgents=agents, numObjects=objects)
    configure_evaluation_graph(env, True, 'klclimb50')
    validate_unified_env(env, evaluation=True)
    assert env['relationGraph']['sampler'] == (
        STAGE2_RESCUE_SAMPLER if (agents, objects) == (2, 4) else STAGE2_GENERAL_RESCUE_SAMPLER)
    assert env['relationGraph']['edge_capacity'] == 2 * agents
    assert 'stage2RoleSizes' not in env['box']['build']
    assert env['relationReward'] == ENV['relationReward']
    assert env['templateRsi'] == ENV['templateRsi']
    assert env['skillDiscProb'] == ENV['skillDiscProb']
    assert 'stage2RoleSizes' in ENV['box']['build']
    for preset in ('random_scenario', 'place_climb', 'place_sit', 'place_stack', 'independent'):
        graph = sample_graph(16, env['relationGraph'], preset=preset,
                             generator=torch.Generator().manual_seed(91))
        validate_graph(graph)
        assert graph.num_agents == agents and graph.num_objects == objects


def test_klclimb_single_viewer_draws_independent_and_all_cooperative_families_across_resets():
    env = deepcopy(ENV)
    configure_evaluation_graph(env, True, 'klclimb50')
    generator = torch.Generator().manual_seed(71)
    families = [int(classify_family(sample_graph(1, env['relationGraph'], generator=generator))[0])
                for _ in range(256)]
    assert set(families) == {0, 1, 2, 3}
    assert .85 < sum(family > 0 for family in families) / len(families) < .95


def test_evaluation_override_retains_cpa_and_explicit_shared9_and_rejects_training():
    env = deepcopy(ENV)
    env.update(agentCollisionMode='cpa', agentCollisionTTCDiscount=.99)
    original = deepcopy(env)
    configure_evaluation_graph(env, True, 'shared9')
    assert env['relationReward'] == original['relationReward']
    assert env['relationGraph']['family_probabilities'] == original['relationGraph']['family_probabilities']
    assert env['agentCollisionMode'] == 'cpa' and env['agentCollisionTTCDiscount'] == .99
    for sampler in ('klclimb50', 'shared9'):
        untouched = deepcopy(ENV)
        with pytest.raises(ValueError, match='evaluation'):
            configure_evaluation_graph(untouched, False, sampler)
        assert untouched == ENV


@pytest.mark.parametrize('agents,objects', [(2, 2), (2, 4), (3, 3), (4, 4), (4, 6), (5, 5), (5, 8), (8, 16)])
def test_default_shared9_evaluation_preserves_all_pair_rules_and_expands_resources(agents, objects):
    from utils.stage2_shared_eval import family_counts, training_spec
    from utils.edge_context_spec import task_instance
    env = deepcopy(ENV)
    env.update(numAgents=agents, numObjects=objects)
    configure_evaluation_graph(env, True)
    assert env['relationGraph']['sampler'] == 'multi_agent_stage2_rescue_shared9'
    assert env['relationGraph']['family_probabilities'] == SPEC['family_probabilities']
    assert env['relationReward'] == ENV['relationReward']
    assert env['box'] == ENV['box']
    assert task_instance(training_spec(env['relationGraph']), 2, 4) == task_instance(SPEC, 2, 4)
    validate_unified_env(env, evaluation=True)
    with pytest.raises(ValueError, match='evaluation only'):
        validate_unified_env(env)
    graph = sample_graph(256, env['relationGraph'], generator=torch.Generator().manual_seed(7))
    validate_graph(graph, allow_shared_ontop=True)
    counts = family_counts(graph).sum(0)
    assert counts[0] > 0 and counts[1:].sum() > 0
    if objects >= 3 * (agents // 2) + agents % 2:
        assert (counts > 0).all()
    for row in range(256):
        holds = graph.edge_valid[row] & (graph.edge_relation[row] == HOLDING)
        sources = graph.edge_dst[row, holds]
        assert len(sources.unique()) == len(sources)
        # Every held/moving source fits the ordinary and payload training ranges.
        assert ((sources - agents) >= agents // 2).all()


def test_single_viewer_reset_sampling_matches_shared9_probabilities_and_all_nine_presets():
    from utils.stage2_shared_eval import family_counts
    env = deepcopy(ENV)
    configure_evaluation_graph(env, True)
    graph = sample_graph(2048, env['relationGraph'], generator=torch.Generator().manual_seed(27))
    proportions = family_counts(graph).sum(0) / 2048
    expected = torch.tensor([.1, .15, .15, .15, .075, .075, .075, .075, .075, .075])
    torch.testing.assert_close(proportions, expected, atol=.025, rtol=0)
    generator = torch.Generator().manual_seed(91)
    families = [int(family_counts(sample_graph(1, env['relationGraph'], generator=generator)).argmax())
                for _ in range(256)]
    assert set(families) == set(range(10))
    for m, o in [(2, 4), (4, 6), (5, 8)]:
        expanded = deepcopy(ENV); expanded.update(numAgents=m, numObjects=o)
        configure_evaluation_graph(expanded, True)
        for index, name in enumerate(PAIRS, 1):
            fixed = sample_graph(8, expanded['relationGraph'], preset=name)
            counts = family_counts(fixed)
            assert (counts[:, index] == m // 2).all()
            assert counts.sum() == 8 * (m // 2)


def test_shared9_eval_sizes_serve_moving_at_and_shared_payload_roles_and_body_checks_all_humans():
    from utils.stage2_shared_eval import sample_sizes
    for m, o in [(2, 4), (4, 6), (5, 8)]:
        sizes = sample_sizes(16, m, o, ENV['box']['build']['stage2RoleSizes'])
        for axis, (low, high) in enumerate(ENV['box']['build']['stage2RoleSizes']['support']):
            assert (sizes[:, :m//2, axis] >= low - 1e-6).all()
            assert (sizes[:, :m//2, axis] <= high + 1e-6).all()
        for role in ('ordinary', 'payload'):
            for axis, (low, high) in enumerate(ENV['box']['build']['stage2RoleSizes'][role]):
                assert (sizes[:, m//2:, axis] >= low - 1e-6).all()
                assert (sizes[:, m//2:, axis] <= high + 1e-6).all()
    bodies = torch.tensor([[[[0., 0., 1.]], [[4., 0., 1.]], [[8., 0., 1.]], [[12., 0., 1.]]]])
    boxes = torch.zeros(1, 6, 13); boxes[..., :3] = 100; boxes[..., 6] = 1
    sizes = torch.ones(1, 6, 3)
    assert not initial_body_collisions(bodies, boxes, sizes).any()
    bodies[:, 3] = bodies[:, 2]
    assert initial_body_collisions(bodies, boxes, sizes).all()


def test_shared9_fixed_impossible_preset_fails_instead_of_replacing_the_scenario():
    env = deepcopy(ENV); env.update(numAgents=2, numObjects=2)
    configure_evaluation_graph(env, True)
    with pytest.raises(ValueError, match='Not enough role-compatible objects'):
        sample_graph(1, env['relationGraph'], preset='ontop_ontop')


def test_contract_separates_loco_initialization_from_all_eight_amp_experts():
    validate_unified_env(ENV)
    assert ENV['ampTaskConditioning'] is False and ENV['numAMPObsSteps'] == 10
    assert ENV['skillDiscProb'] == [.26, .2, .08, .06, .1, .1, .1, .1]
    for field, value in [('ampTaskConditioning', True), ('numAMPObsSteps', 9), ('numObjects', 5)]:
        bad = deepcopy(ENV); bad[field] = value
        with pytest.raises(ValueError):
            validate_unified_env(bad)
    bad = deepcopy(ENV); bad['templateRsi']['SIT'] = [.5, .5, 0., 0., 0., 0., 0., 0.]
    with pytest.raises(ValueError, match='loco'):
        validate_unified_env(bad)
    bad = deepcopy(ENV); bad['relationGraph']['family_probabilities']['at_climb'] = .3
    with pytest.raises(ValueError, match='15%'):
        validate_unified_env(bad)


def test_pool_quotas_and_reset_subsets_retain_physical_asset_roles():
    torch.manual_seed(11)
    pools = scene_pools(2048)
    assert pools.bincount().tolist() == [205, 922, 921]
    assert set(scene_pools(1).tolist()) <= {0, 1, 2}
    sizes = sample_role_sizes(pools, ENV['box']['build']['stage2RoleSizes'])
    for role, values in [('ordinary', sizes[pools != 2]), ('support', sizes[pools == 2, 0]),
                         ('payload', sizes[pools == 2, 1:3])]:
        for axis, (low, high) in enumerate(ENV['box']['build']['stage2RoleSizes'][role]):
            assert (values[..., axis] >= low - 1e-6).all()
            assert (values[..., axis] <= high + 1e-6).all()
            torch.testing.assert_close(values[..., axis] / .05,
                                       (values[..., axis] / .05).round(), atol=2e-6, rtol=0)
    counts = torch.zeros(10)
    for _ in range(4):
        ids = torch.randperm(2048)[:512]
        graph = sample_graph(len(ids), SPEC, scene_pools=pools[ids])
        family = classify_family(graph)
        assert ((family == 0) == (pools[ids] == 0)).all()
        assert (((family >= 1) & (family <= 3)) == (pools[ids] == 1)).all()
        assert ((family >= 4) == (pools[ids] == 2)).all()
        order = box_order(graph, pools[ids])
        torch.testing.assert_close(order.sort(-1).values, torch.arange(4).expand(len(ids), -1))
        logical_sizes = sizes[ids].gather(1, order[..., None].expand(-1, -1, 3))
        for row in (pools[ids] == 2).nonzero().flatten().tolist():
            active = graph.edge_valid[row]
            for edge in active.nonzero().flatten().tolist():
                relation = int(graph.edge_relation[row, edge])
                if relation != HOLDING:
                    target = int(graph.edge_dst[row, edge]) - 2
                    torch.testing.assert_close(logical_sizes[row, target], sizes[ids[row], 0])
                if relation == ON_TOP:
                    source = int(graph.edge_src[row, edge]) - 2
                    assert int(order[row, source]) in (1, 2)
        counts += torch.bincount(family, minlength=10)
    expected = torch.tensor([.1, .15, .15, .15, .075, .075, .075, .075, .075, .075])
    torch.testing.assert_close(counts / counts.sum(), expected, atol=.025, rtol=0)


@pytest.mark.parametrize('preset', tuple(PAIRS))
@pytest.mark.parametrize('swapped', (False, True))
def test_each_pair_has_two_correctly_owned_bundles_and_unique_sources(preset, swapped):
    graph = sample_graph(8, SPEC, preset=preset, role_swap=swapped)
    validate_graph(graph, allow_shared_ontop=True)
    assert (classify_family(graph) == 1 + tuple(PAIRS).index(preset)).all()
    assert (graph.required_goal.sum(-1) == 2).all()
    humans = torch.arange(2).repeat(8)
    rows = torch.arange(8).repeat_interleave(2)
    templates = classify_templates(graph, rows, humans, True).reshape(8, 2)
    names = ('HOLDING', 'SIT', 'CLIMB', 'HOLDING_AT', 'HOLDING_ON_TOP')
    expected = [names.index(k) for k in PAIRS[preset]]
    if swapped:
        expected.reverse()
    torch.testing.assert_close(templates, torch.tensor(expected).expand(8, -1))
    primary = agent_object_indices(graph, rows, humans).reshape(8, 2)
    if preset in ('climb_climb', 'sit_climb', 'sit_sit', 'at_climb', 'at_sit'):
        assert (primary[:, 0] == primary[:, 1]).all()
    else:
        assert (primary[:, 0] != primary[:, 1]).all()
    if preset == 'ontop_ontop':
        with pytest.raises(ValueError, match='support'):
            validate_graph(graph)
        for row in range(8):
            top = graph.edge_valid[row] & (graph.edge_relation[row] == ON_TOP)
            assert graph.edge_src[row, top].unique().numel() == 2
            assert graph.edge_dst[row, top].unique().numel() == 1


def test_initial_collision_filter_catches_other_objects_humans_and_rotated_boxes():
    bodies = torch.tensor([[[[0., 0., 1.]], [[3., 0., 1.]]]]).repeat(4, 1, 1, 1)
    boxes = torch.zeros(4, 4, 13); boxes[..., 6] = 1
    boxes[..., :3] = torch.tensor([10., 0., .25])
    sizes = torch.full((4, 4, 3), .5)
    boxes[1, 3, :3] = torch.tensor([0., 0., 1.])
    bodies[2, 1] = bodies[2, 0] + .01
    boxes[3, 0, :3] = torch.tensor([.3, .3, 1.])
    boxes[3, 0, 3:7] = torch.tensor([0., 0., .3826834, .9238795])
    sizes[3, 0] = torch.tensor([1., .2, .5])
    assert initial_body_collisions(bodies, boxes, sizes).tolist() == [False, True, True, True]


def test_transfer_amp_shape_and_active_attention_permutations_for_all_nine():
    torch.manual_seed(37); torch.set_num_threads(1)
    source = network(False, 3)
    reward = deepcopy(SOURCE['relationReward'])
    reward['stage1_variant'] = 'scenario_independent_stage1_self_sum_rescue'
    checkpoint = dict(model=Wrapper(source).state_dict(), relation_metadata=checkpoint_metadata(reward), epoch=8000)
    target = network(True, graph_spec=SPEC).eval()
    transfer_stage1_weights(Wrapper(target), checkpoint)
    assert target._disc_mlp[0].weight.shape == (1024, 1290)
    assert all(not p.requires_grad for p in target.actor_encoder.parameters())
    reference = network(False, 4).eval()
    Wrapper(reference).load_state_dict(checkpoint['model'], strict=True)
    for preset in PAIRS:
        graph = sample_graph(4, SPEC, preset=preset)
        obs = observation(graph)
        with torch.no_grad():
            torch.testing.assert_close(target.eval_actor(obs)[0], reference.eval_actor(obs)[0], atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        target.action_head[0][0].weight[:, 64:].normal_(std=.1)
    for preset in PAIRS:
        graph = sample_graph(4, SPEC, preset=preset)
        obs = observation(graph)
        mu, sigma = target.eval_actor(obs)
        shuffled = permute_graph(graph, torch.rand(4, 4).argsort(-1))
        edge_obs = torch.cat((obs[:, :-20], semantic_graph_packet(shuffled, 4)), -1)
        edge_mu, edge_sigma = target.eval_actor(edge_obs)
        torch.testing.assert_close(edge_mu, mu, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(edge_sigma, sigma, atol=0, rtol=0)
        original = target.actor_encoder.forward
        with patch.object(target.actor_encoder, 'forward', side_effect=lambda value, **kw: original(value, token_order=torch.tensor([6,4,1,3,7,0,5,2]), **kw)):
            token_mu, _ = target.eval_actor(obs)
        torch.testing.assert_close(token_mu, mu, atol=2e-5, rtol=2e-5)
        node_order = torch.tensor([1,0,3,2,5,4,7,6])
        relabeled = replace(graph, edge_src=node_order[graph.edge_src],
            edge_dst=node_order[graph.edge_dst], edge_owner=1-graph.edge_owner)
        human_obs = torch.cat((obs[:,:446].reshape(4,2,223).flip(1).flatten(1),
            obs[:,446:566].reshape(4,4,30)[:,[1,0,3,2]].flatten(1), obs[:,566:568].flip(1),
            obs[:,568:624].reshape(4,8,7)[:,node_order].flatten(1), semantic_graph_packet(relabeled,4)), -1)
        relabeled_mu, _ = target.eval_actor(human_obs)
        torch.testing.assert_close(relabeled_mu.reshape(4,2,32), mu.reshape(4,2,32).flip(1), atol=2e-5, rtol=2e-5)
        target.zero_grad(); mu.square().mean().backward()
        assert target.coordination.grounding[0].weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in target.actor_encoder.parameters())
