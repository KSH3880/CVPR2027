"""Schema-5 independent Stage-1 sampler, reward, packet and network tests."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from utils.edge_context_spec import HOLDING, AT
from utils.edge_ontop_spec import ON_TOP, graph_packet
from utils.edge_interaction_spec import SIT, CLIMB
from utils.edge_stage1_spec import *
from utils.relation_task_spec import (checkpoint_metadata, check_checkpoint_metadata,
    validate_relation_config)
from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime, stage1_context
from learning.multi_agent.amp_network_builder_ma import RelationEncoder


torch.set_num_threads(1)
ROOT = Path(__file__).parents[1]
ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_distance_edge_context_stage1.yaml').read_text())['env']
CFG, SPEC = ENV['relationReward'], ENV['relationGraph']
COMMON_ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/'
    'approach_distance_edge_context_stage1_common_boxes.yaml').read_text())['env']
FIXED_ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/'
    'approach_distance_edge_context_stage1_fixed_boxes_sit_fix.yaml').read_text())['env']
PRIMITIVE_ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/'
    'approach_distance_edge_context_stage1_primitives_rsi.yaml').read_text())['env']
CLIMB_ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/'
    'approach_distance_stage1_climb_rsi.yaml').read_text())['env']


def test_climb_only_semantic_packet_and_reward_contract():
    cfg, spec = CLIMB_ENV['relationReward'], CLIMB_ENV['relationGraph']
    validate_relation_config(cfg)
    validate_sampler(spec)
    validate_relation_rsi(CLIMB_ENV['relationRsi'], CLIMB_ENV['skill'])
    assert 'context' not in cfg and 'contextReward' not in cfg
    assert cfg['observation']['graph_packet_fields'] == list(STAGE1_SEMANTIC_FIELDS)
    assert CLIMB_ENV['box'] == PRIMITIVE_ENV['box']
    graph = sample_graph(20, spec, generator=torch.Generator().manual_seed(17))
    assert (graph.edge_relation[graph.edge_valid] == CLIMB).all()
    assert (graph.edge_valid.sum(-1) == 2).all()
    with pytest.raises(ValueError, match='CLIMB-only'):
        sample_graph(1, spec, preset='holding')
    packet = semantic_graph_packet(graph, 20)
    assert packet.shape == (20, 20)
    valid, src, dst, relation, owner = parse_semantic_packet(packet)
    assert torch.equal(valid, graph.edge_valid)
    assert torch.equal(relation[valid], graph.edge_relation[valid])
    meta = checkpoint_metadata(cfg)
    assert meta['graph_record_width'] == 5 and meta['context_dim_per_edge'] == 0
    with pytest.raises(ValueError):
        check_checkpoint_metadata(
            {'relation_metadata': checkpoint_metadata(PRIMITIVE_ENV['relationReward'])}, meta)


def test_climb_only_progress_pinning_success_and_no_context_network():
    cfg, spec = CLIMB_ENV['relationReward'], CLIMB_ENV['relationGraph']
    graph = sample_graph(1, spec, preset='climb')
    climb = ((graph.edge_relation == CLIMB) & graph.edge_valid).nonzero(as_tuple=False)[0, 1]
    hands, feet, roots, objects, sizes, goals = _scene()
    roots[0, 0] = torch.tensor([.6, 0., 1.34])
    feet[0, 0, :, 2] = .46
    phi, diag = evaluate_interaction_edges(hands, feet, roots, objects, sizes, goals,
        graph, cfg, .94)
    assert diag['progress'][0, climb] == pytest.approx(1.)
    assert phi[0, climb] < .6
    roots[0, 0, 0] = .2
    phi, diag = evaluate_interaction_edges(hands, feet, roots, objects, sizes, goals,
        graph, cfg, .94)
    runtime = Stage1ContextRuntime(1, graph, cfg, 'cpu')
    result = runtime.step(phi, diag['progress'], diag['z_error'], diag['feet_height_error'])
    assert result['own_success'][0, climb] and result['total'][0, climb] == pytest.approx(.6)
    assert diag['feet_individual_max_error'][0, climb] == pytest.approx(.06)
    assert runtime.suffix().shape == (1, 20)
    feet[0, 0, :, 2] = .471
    phi, diag = evaluate_interaction_edges(hands, feet, roots, objects, sizes, goals,
        graph, cfg, .94)
    assert not runtime.step(phi, diag['progress'], diag['z_error'],
        diag['feet_height_error'])['own_success'][0, climb]

    net = RelationEncoder([223, 30, 1], 2, 3, 16, 2, 2, 32,
        lambda size: nn.Sequential(nn.Linear(size, 16), nn.ReLU()),
        observation_mode='clean_scene', kinematic_size=7,
        relation_bias_mode='edge_mlp', gta_cfg={'enable': True},
        relation_reward_mode=STAGE1_CONTEXT_MODE, relation_graph_spec=spec)
    with torch.no_grad():
        net.edge_encoder.bias_projection.normal_(std=.1)
    assert net.suffix_width == 20
    assert not any('context_encoder' in name for name, _ in net.named_parameters())
    width = 224 * 2 + 30 * 3
    obs = torch.randn(1, width + 7 * 7 + 20)
    obs[:, width:width + 49].reshape(1, 7, 7)[..., 3:7] = torch.tensor([0., 0., 0., 1.])
    obs[:, -20:] = runtime.suffix()
    out = net(obs)
    assert out.shape == (1, 2, 16) and torch.isfinite(out).all()
    out.square().sum().backward()
    assert net.edge_encoder.relation_embed.weight.grad[CLIMB].abs().sum() > 0


def test_primitive_relation_rsi_graph_and_checkpoint_contract():
    env = PRIMITIVE_ENV
    validate_relation_config(env['relationReward'])
    validate_sampler(env['relationGraph'])
    validate_relation_rsi(env['relationRsi'], env['skill'])
    assert env['relationRsi']['HOLDING'] == [.5, 0, 0, 0, .5, 0, 0]
    assert env['relationRsi']['SIT'] == [.5, .5, 0, 0, 0, 0, 0]
    assert env['relationRsi']['CLIMB'] == [.5, 0, .5, 0, 0, 0, 0]
    assert env['box'] == COMMON_ENV['box']
    assert env['skillDiscProb'] == COMMON_ENV['skillDiscProb']
    assert env['relationReward']['schema_version'] == 6
    with pytest.raises(ValueError, match='schema mismatch'):
        check_checkpoint_metadata(
            {'relation_metadata': checkpoint_metadata(COMMON_ENV['relationReward'])},
            checkpoint_metadata(env['relationReward']))
    graph = sample_graph(30000, env['relationGraph'],
                         generator=torch.Generator().manual_seed(15))
    assert (graph.edge_valid.sum(-1) == 2).all()
    assert (graph.edge_relation[graph.edge_valid].unique().sort().values ==
            torch.tensor([HOLDING, SIT, CLIMB])).all()
    assert not (graph.edge_dst[graph.edge_valid] == 4).any()
    assert not graph.prereq_mask.any()
    assert not (graph.term_index >= 0).any()
    for a in (0, 1):
        owned = graph.edge_valid & (graph.edge_owner == a)
        assert (owned.sum(-1) == 1).all()
        assert (graph.edge_dst[owned] == 2 + a).all()
        counts = torch.stack([((graph.edge_relation == relation) & owned).any(-1).float().mean()
                              for relation in (HOLDING, SIT, CLIMB)])
        torch.testing.assert_close(counts, torch.full((3,), 1 / 3), atol=.01, rtol=0)
    for preset in ('holding_at', 'holding_sit'):
        with pytest.raises(ValueError, match='Primitive'):
            sample_graph(1, env['relationGraph'], preset=preset)


def test_primitive_relation_rsi_rejects_mismatched_skill():
    rows = deepcopy(PRIMITIVE_ENV['relationRsi'])
    rows['SIT'] = [0.5, 0, 0, 0, 0, .5, 0]
    with pytest.raises(ValueError, match='does not match'):
        validate_relation_rsi(rows, PRIMITIVE_ENV['skill'])


def _scene(n=1):
    objects = torch.zeros(n, 3, 13); objects[..., 6] = 1; objects[..., 2] = .2
    sizes = torch.tensor([[[.5, .5, .4]] * 3]).expand(n, -1, -1).clone()
    hands = torch.zeros(n, 2, 2, 3)
    feet = torch.zeros(n, 2, 2, 3)
    roots = torch.zeros(n, 2, 3)
    goals = torch.zeros(n, 2, 3)
    return hands, feet, roots, objects, sizes, goals


def _patterns(graph):
    result = []
    for agent in (0, 1):
        owned = graph.edge_valid & (graph.edge_owner == agent)
        hold = ((graph.edge_relation == HOLDING) & owned).any(-1)
        sit = ((graph.edge_relation == SIT) & owned).any(-1)
        climb = ((graph.edge_relation == CLIMB) & owned).any(-1)
        at = ((graph.edge_relation == AT) & owned).any(-1)
        top = ((graph.edge_relation == ON_TOP) & owned).any(-1)
        result.append(torch.where(~hold & sit, 1,
            torch.where(~hold & climb, 2,
            torch.where(hold & at, 3,
            torch.where(hold & top, 4,
            torch.where(hold & sit, 5,
            torch.where(hold & climb, 6, 0)))))))
    return torch.stack(result, -1)


def test_stage1_config_sampling_marginals_and_ox_exclusion():
    validate_relation_config(CFG)
    validate_sampler(SPEC)
    graph = sample_graph(100000, SPEC, generator=torch.Generator().manual_seed(71))
    pattern = _patterns(graph)
    expected = torch.tensor(list(SPEC['pattern_probabilities'].values()))
    for agent in (0, 1):
        actual = torch.stack([(pattern[:, agent] == i).float().mean() for i in range(7)])
        torch.testing.assert_close(actual, expected, atol=.004, rtol=0)
    ox = graph.num_agents + 2
    ox_users = torch.zeros(len(pattern), 2, dtype=torch.bool)
    for agent in (0, 1):
        ox_users[:, agent] = ((graph.edge_dst == ox) & graph.edge_valid &
                              (graph.edge_owner == agent)).any(-1)
    assert (ox_users.sum(-1) <= 1).all()
    assert not graph.prereq_mask.any()
    assert not (graph.term_index >= 0).any()
    assert torch.equal(graph.required_goal, graph.edge_valid)


def test_stage1_bindings_never_use_teammate_object():
    graph = sample_graph(20000, SPEC, generator=torch.Generator().manual_seed(9))
    for agent in (0, 1):
        owned = graph.edge_valid & (graph.edge_owner == agent)
        teammate = graph.num_agents + 1 - agent
        assert not ((graph.edge_dst == teammate) & owned).any()
        hold = owned & (graph.edge_relation == HOLDING)
        assert (graph.edge_dst[hold] == graph.num_agents + agent).all()
        standalone = owned & ((graph.edge_relation == SIT) | (graph.edge_relation == CLIMB))
        has_hold = hold.any(-1)[:, None]
        expected = torch.where(has_hold, graph.num_agents + 2,
                               graph.num_agents + agent)
        assert (graph.edge_dst[standalone] == expected.expand_as(graph.edge_dst)[standalone]).all()


def test_grounded_variant_floors_only_standalone_interaction_targets():
    variant = yaml.safe_load((ROOT / 'data/cfg/multi_agent/'
        'approach_distance_edge_context_stage1_ground_sit_climb.yaml').read_text())['env']
    original = yaml.safe_load((ROOT / 'data/cfg/multi_agent/'
        'approach_distance_edge_context_stage1.yaml').read_text())['env']
    assert variant['box']['reset'].pop('groundStandaloneSitClimb') is True
    assert variant == original
    assert checkpoint_metadata(variant['relationReward']) == checkpoint_metadata(CFG)

    graph = compose_graph(torch.tensor([[1, 0], [0, 2], [5, 0], [4, 1]]))
    env_ids = torch.tensor([0, 1, 3])  # Keep env 2 and its HOLDING+SIT untouched.
    expected_mask = torch.tensor([[True, False], [False, True],
                                  [False, True]])
    torch.testing.assert_close(standalone_interaction_owners(graph, env_ids), expected_mask)
    assignments = torch.tensor([[2, 0], [1, 2], [0, 1], [1, 0]])
    boxes = torch.zeros(4, 3, 13)
    boxes[..., 2] = .85
    sizes = torch.full((4, 3, 3), .4)
    platforms = torch.full((4, 2, 3), .6)
    defaults = torch.full((4, 2, 3), 20.)
    ground_standalone_interaction_targets(graph, env_ids, assignments,
                                          boxes, sizes, platforms, defaults)
    expected_boxes = torch.full((4, 3), .85)
    expected_boxes[0, 2] = .2
    expected_boxes[1, 2] = .2
    expected_boxes[3, 0] = .2
    torch.testing.assert_close(boxes[..., 2], expected_boxes)
    expected_platform_z = torch.full((4, 2), .6)
    expected_platform_z[0, 0] = 20.
    expected_platform_z[1, 1] = 20.
    expected_platform_z[3, 1] = 20.
    torch.testing.assert_close(platforms[..., 2], expected_platform_z)


def test_common_boxes_variant_has_independent_full_range_and_pair_stack_limit():
    variant = yaml.safe_load((ROOT / 'data/cfg/multi_agent/'
        'approach_distance_edge_context_stage1_ground_sit_climb.yaml').read_text())['env']
    common = deepcopy(COMMON_ENV)
    validate_relation_config(common['relationReward'])
    validate_sampler(common['relationGraph'])
    old_cfg = variant['relationReward']
    new_cfg = common['relationReward']
    assert new_cfg['sit'] == {'state_definition': 'box_top_plus_pelvis_clearance',
                              'near_distance_scale': 10.0, 'pelvis_clearance': .12}
    assert {k: v for k, v in new_cfg.items() if k != 'sit'} == {
        k: v for k, v in old_cfg.items() if k != 'sit'}
    with pytest.raises(ValueError, match='reward config differs'):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(old_cfg)},
                                  checkpoint_metadata(new_cfg))

    build = common['box']['build']
    assert build['randomSize'] and not build['randomModeEqualProportion']
    assert build['baseSize'] == [.4, .4, .4]
    scales = [torch.arange(build['scaleRange' + axis][0],
                           build['scaleRange' + axis][1] + build['scaleSampleInterval'],
                           build['scaleSampleInterval'])
              for axis in 'XYZ']
    sizes = [build['baseSize'][i] * scales[i] for i in range(3)]
    for values, lo, hi in zip(sizes, (.4, .4, .25), (.65, .65, .55)):
        assert values[0] == pytest.approx(lo)
        assert values[-1] == pytest.approx(hi)
    assert [len(values) for values in sizes] == [6, 6, 7]
    old_build = variant['box']['build']
    common['box']['build'] = old_build
    common['relationReward'] = old_cfg
    assert common == variant

    box_sizes = torch.tensor([[[.65, .4, .55], [.4, .65, .55], [.5, .5, .55]]])
    assert max_stage1_stack_height(box_sizes).item() == pytest.approx(1.1)
    assert box_sizes[..., 2].sum().item() > variant['box']['reset']['maxTopSurfaceHeight']


def test_fixed_box_sit_fix_changes_only_sit_geometry_from_grounded_variant():
    old = yaml.safe_load((ROOT / 'data/cfg/multi_agent/'
        'approach_distance_edge_context_stage1_ground_sit_climb.yaml').read_text())['env']
    fixed = deepcopy(FIXED_ENV)
    validate_relation_config(fixed['relationReward'])
    validate_sampler(fixed['relationGraph'])
    assert fixed['box']['build']['baseSize'] == [.5, .5, .4]
    assert fixed['box']['build']['randomSize'] is False
    assert fixed['box']['reset']['groundStandaloneSitClimb'] is True
    assert fixed['relationReward']['sit'] == COMMON_ENV['relationReward']['sit']
    assert checkpoint_metadata(fixed['relationReward']) == checkpoint_metadata(COMMON_ENV['relationReward'])
    with pytest.raises(ValueError, match='reward config differs'):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(old['relationReward'])},
                                  checkpoint_metadata(fixed['relationReward']))
    fixed['relationReward'] = old['relationReward']
    assert fixed == old


@pytest.mark.parametrize('env', [COMMON_ENV, FIXED_ENV], ids=['variable', 'fixed'])
def test_box_top_sit_target_tracks_actual_top_and_success(env):
    cfg = env['relationReward']
    sit_graph = sample_graph(1, SPEC, preset='sit')
    sit = ((sit_graph.edge_relation == SIT) & sit_graph.edge_valid).nonzero(as_tuple=False)[0, 1]
    for height in (.25, .4, .55):
        hands, feet, roots, objects, sizes, goals = _scene()
        sizes[0, 0, 2] = height
        objects[0, 0, 2] = height / 2
        roots[0, 0, 2] = height + cfg['sit']['pelvis_clearance']
        phi, diag = evaluate_interaction_edges(hands, feet, roots, objects, sizes,
                                                goals, sit_graph, cfg, .94)
        assert phi[0, sit] == pytest.approx(1.)
        assert diag['target'][0, sit, 2] == pytest.approx(height + .12)
        assert diag['progress'][0, sit] == pytest.approx(1.)
        runtime = Stage1ContextRuntime(1, sit_graph, cfg, 'cpu')
        assert runtime.step(phi, diag['progress'], diag['z_error'],
                            diag['feet_height_error'])['own_success'][0, sit]
        roots[0, 0, 2] = height - .062  # Old chair offset target below the solid box top.
        phi_bad, diag_bad = evaluate_interaction_edges(hands, feet, roots, objects,
            sizes, goals, sit_graph, cfg, .94)
        assert phi_bad[0, sit] < cfg['satisfaction_threshold']
        assert not runtime.step(phi_bad, diag_bad['progress'], diag_bad['z_error'],
                                diag_bad['feet_height_error'])['own_success'][0, sit]


def test_stage1_presets_and_role_swap():
    for preset in PRESETS:
        graph = sample_graph(4, SPEC, preset=preset)
        validate_graph(graph)
    normal = sample_graph(1, SPEC, preset='holding_sit')
    swapped = sample_graph(1, SPEC, preset='holding_sit', role_swap=True)
    assert _patterns(normal).tolist() == [[5, 0]]
    assert _patterns(swapped).tolist() == [[0, 5]]


def test_stage1_context_is_constant_and_masked():
    graph = sample_graph(8, SPEC, generator=torch.Generator().manual_seed(3))
    context = stage1_context(torch.rand(8, 4), graph)
    assert (context[graph.edge_valid] == 1).all()
    assert (context[~graph.edge_valid] == 0).all()


def test_stage1_has_own_success_only_and_no_sharing():
    graph = sample_graph(1, SPEC, preset='holding_at')
    runtime = Stage1ContextRuntime(1, graph, CFG, 'cpu')
    phi = torch.zeros(1, 4); progress = torch.zeros_like(phi)
    z = torch.zeros_like(phi); feet = torch.zeros_like(phi)
    at = ((graph.edge_relation == AT) & graph.edge_valid).nonzero(as_tuple=False)[0, 1]
    hold = ((graph.edge_relation == HOLDING) & graph.edge_valid &
            (graph.edge_owner == 0)).nonzero(as_tuple=False)[0, 1]
    phi[0, at] = 1
    out = runtime.step(phi, progress, z, feet)
    assert out['total'][0, at] == pytest.approx(.6)
    assert out['total'][0, hold] == 0
    assert not out['reward_saturated'][0, hold]
    torch.testing.assert_close(out['agent_task_reward'], out['local_task_reward'])
    assert out['agent_task_reward'][0, 1] == 0


def test_stage1_keeps_established_sit_and_climb_geometry():
    hands, feet, roots, objects, sizes, goals = _scene()
    sit_graph = sample_graph(1, SPEC, preset='sit')
    sit = ((sit_graph.edge_relation == SIT) & sit_graph.edge_valid).nonzero(as_tuple=False)[0, 1]
    roots[0, 0, 2] = .2 + CFG['sit']['target_local_offset'][2]
    phi, diag = evaluate_interaction_edges(hands, feet, roots, objects, sizes, goals,
                                            sit_graph, CFG, .94)
    assert phi[0, sit] == pytest.approx(1.)

    hands, feet, roots, objects, sizes, goals = _scene()
    climb_graph = sample_graph(1, SPEC, preset='climb')
    climb = ((climb_graph.edge_relation == CLIMB) & climb_graph.edge_valid).nonzero(as_tuple=False)[0, 1]
    roots[0, 0, 2] = 1.34
    feet[0, 0, :, 2] = .4
    phi, diag = evaluate_interaction_edges(hands, feet, roots, objects, sizes, goals,
                                            climb_graph, CFG, .94)
    runtime = Stage1ContextRuntime(1, climb_graph, CFG, 'cpu')
    assert runtime.step(phi, diag['progress'], diag['z_error'],
                        diag['feet_height_error'])['own_success'][0, climb]
    feet[0, 0, :, 2] = .451
    phi2, diag2 = evaluate_interaction_edges(hands, feet, roots, objects, sizes, goals,
                                             climb_graph, CFG, .94)
    torch.testing.assert_close(phi2, phi)
    assert not runtime.step(phi2, diag2['progress'], diag2['z_error'],
                            diag2['feet_height_error'])['own_success'][0, climb]


def test_stage1_packet_network_and_checkpoint_are_separate():
    graph = compose_graph(torch.tensor([[1, 2], [2, 1], [5, 0]]))
    net = RelationEncoder([223, 30, 1], 2, 3, 16, 2, 2, 32,
        lambda size: nn.Sequential(nn.Linear(size, 16), nn.ReLU()),
        observation_mode='clean_scene', kinematic_size=7,
        relation_bias_mode='edge_mlp', gta_cfg={'enable': True},
        relation_reward_mode=STAGE1_CONTEXT_MODE, relation_graph_spec=SPEC)
    with torch.no_grad():
        net.edge_encoder.bias_projection.normal_(std=.1)
    width = 224 * 2 + 30 * 3
    obs = torch.randn(3, width + 7 * 7 + 7 * 4)
    obs[:, width:width + 49].reshape(3, 7, 7)[..., 3:7] = torch.tensor([0., 0., 0., 1.])
    obs[:, -28:] = graph_packet(graph, stage1_context(torch.rand(3, 4), graph))
    out = net(obs)
    assert out.shape == (3, 2, 16) and torch.isfinite(out).all()
    out.square().sum().backward()
    assert net.edge_encoder.relation_embed.weight.grad[SIT].abs().sum() > 0
    assert net.edge_encoder.relation_embed.weight.grad[CLIMB].abs().sum() > 0

    meta = checkpoint_metadata(CFG)
    check_checkpoint_metadata({'relation_metadata': meta}, meta)
    interaction = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_distance_edge_context_interaction.yaml').read_text())['env']['relationReward']
    with pytest.raises(ValueError):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(interaction)}, meta)
