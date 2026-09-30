from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from env.tasks.multi_agent.edge_interaction_reward import (
    evaluate_interaction_edges, interaction_own_success)
from env.tasks.multi_agent.edge_ontop_task import SampledOnTopTaskMixin
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from env.tasks.multi_agent.edge_context_reward import owner_sum
from utils.edge_context_spec import AT, HOLDING
from utils.edge_interaction_spec import CLIMB, SIT
from utils.edge_ontop_spec import ON_TOP
from utils.edge_scenario_spec import sample_graph
from utils.edge_stage1_spec import (mix_late_climb_rsi_times,
    sample_ontop_putdown_times, validate_sampler)
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata, validate_relation_config


ROOT = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent'
BASE = yaml.safe_load((ROOT / 'approach_scenario_stage1_plane.yaml').read_text())['env']
NORMALIZED = yaml.safe_load((ROOT / 'approach_scenario_stage1_sit_plane_normalized.yaml').read_text())['env']
CURRICULUM = yaml.safe_load((ROOT / 'approach_scenario_stage1_skill_curriculum.yaml').read_text())['env']
PRESERVED = yaml.safe_load((ROOT / 'approach_scenario_stage1_skill_curriculum_reward_preserved.yaml').read_text())['env']
AT_GOAL_FIX = yaml.safe_load((ROOT / 'approach_scenario_stage1_at_goal_fix.yaml').read_text())['env']
SELF_SUM = yaml.safe_load((ROOT / 'approach_scenario_stage1_self_sum.yaml').read_text())['env']
SLOW_NEAR = yaml.safe_load((ROOT / 'approach_scenario_stage1_slow_near_start.yaml').read_text())['env']
PLACEMENT_FOCUS = yaml.safe_load((ROOT / 'approach_scenario_stage1_placement_focus.yaml').read_text())['env']
PAIRED_AT = yaml.safe_load((ROOT / 'approach_scenario_stage1_paired_at_no_near.yaml').read_text())['env']
PAIRED_PLACEMENT = yaml.safe_load((ROOT / 'approach_scenario_stage1_paired_placement_no_near.yaml').read_text())['env']
ONTOP_PUTDOWN = yaml.safe_load((ROOT / 'approach_scenario_stage1_ontop_putdown.yaml').read_text())['env']


def test_at_goal_fix_keeps_experiment_32_settings_and_checkpoint_separate():
    validate_sampler(AT_GOAL_FIX['relationGraph'])
    validate_relation_config(AT_GOAL_FIX['relationReward'])
    original = deepcopy(PRESERVED)
    original['relationReward']['stage1_variant'] = AT_GOAL_FIX['relationReward']['stage1_variant']
    assert AT_GOAL_FIX == original
    assert checkpoint_metadata(AT_GOAL_FIX['relationReward']) != checkpoint_metadata(
        PRESERVED['relationReward'])


def test_self_sum_and_slow_near_configs_change_only_intended_fields():
    for env in (SELF_SUM, SLOW_NEAR):
        validate_sampler(env['relationGraph'])
        validate_relation_config(env['relationReward'])
        assert env['relationGraph']['template_probabilities'] == {
            'HOLDING': .1, 'SIT': .1, 'CLIMB': .1,
            'HOLDING_AT': .35, 'HOLDING_ON_TOP': .35}
        assert env['relationReward']['edge_aggregation'] == 'self_sum_teammate_mean'
    old = deepcopy(AT_GOAL_FIX)
    old['relationGraph']['template_probabilities'] = SELF_SUM['relationGraph']['template_probabilities']
    old['relationReward']['stage1_variant'] = SELF_SUM['relationReward']['stage1_variant']
    old['relationReward']['edge_aggregation'] = 'self_sum_teammate_mean'
    assert SELF_SUM == old
    new = deepcopy(SELF_SUM)
    new['relationReward']['stage1_variant'] = SLOW_NEAR['relationReward']['stage1_variant']
    new['relationReward']['hard_skill_training']['near_start'].update(
        hold_steps=600000, anneal_steps=600000)
    assert SLOW_NEAR == new
    assert len({str(checkpoint_metadata(env['relationReward'])) for env in
                (AT_GOAL_FIX, SELF_SUM, SLOW_NEAR)}) == 3


def test_placement_focus_samples_both_placements_with_holding_and_keeps_rsi():
    validate_sampler(PLACEMENT_FOCUS['relationGraph'])
    validate_relation_config(PLACEMENT_FOCUS['relationReward'])
    expected = deepcopy(SELF_SUM)
    expected['relationGraph']['template_probabilities'] = {
        'HOLDING': 0., 'SIT': 0., 'CLIMB': 0.,
        'HOLDING_AT': .5, 'HOLDING_ON_TOP': .5}
    expected['relationReward']['stage1_variant'] = (
        'scenario_independent_stage1_placement_focus')
    assert PLACEMENT_FOCUS == expected
    graph = sample_graph(32, PLACEMENT_FOCUS['relationGraph'],
                         generator=torch.Generator().manual_seed(36))
    for relation, count in ((HOLDING, 2), (AT, 1), (ON_TOP, 1), (SIT, 0), (CLIMB, 0)):
        assert ((graph.edge_relation == relation) & graph.edge_valid).sum(-1).tolist() == [count] * 32
    with pytest.raises(ValueError, match='reward config differs'):
        check_checkpoint_metadata(
            {'relation_metadata': checkpoint_metadata(SELF_SUM['relationReward'])},
            checkpoint_metadata(PLACEMENT_FOCUS['relationReward']))


def test_paired_at_has_two_independent_goals_without_near_start():
    expected = deepcopy(PLACEMENT_FOCUS)
    expected['relationGraph']['template_probabilities']['HOLDING_AT'] = 1.
    expected['relationGraph']['template_probabilities']['HOLDING_ON_TOP'] = 0.
    expected['relationReward']['stage1_variant'] = 'scenario_independent_stage1_paired_at_no_near'
    expected['relationReward']['hard_skill_training']['near_start'].update(
        probability_start=0., probability_end=0.)
    assert PAIRED_AT == expected
    validate_sampler(PAIRED_AT['relationGraph'], 2, 3)
    validate_relation_config(PAIRED_AT['relationReward'])
    graph = sample_graph(128, PAIRED_AT['relationGraph'], generator=torch.Generator().manual_seed(38))
    assert ((graph.edge_relation == AT) & graph.edge_valid).sum(-1).eq(2).all()
    assert ((graph.edge_relation == HOLDING) & graph.edge_valid).sum(-1).eq(2).all()
    for row in range(128):
        edges = graph.edge_valid[row] & (graph.edge_relation[row] == AT)
        assert graph.edge_src[row, edges].unique().numel() == 2
        assert graph.edge_dst[row, edges].unique().numel() == 2


def test_four_object_paired_placement_uses_disjoint_supports_and_one_relation_per_scene():
    expected = deepcopy(PLACEMENT_FOCUS)
    expected['numObjects'] = 4
    expected['relationGraph']['sampler'] = 'two_agent_four_object_paired_placement'
    expected['relationReward']['stage1_variant'] = (
        'scenario_independent_stage1_paired_placement_no_near')
    expected['relationReward']['hard_skill_training']['near_start'].update(
        probability_start=0., probability_end=0.)
    assert PAIRED_PLACEMENT == expected
    validate_sampler(PAIRED_PLACEMENT['relationGraph'])
    validate_sampler(PAIRED_PLACEMENT['relationGraph'], 2, 4)
    validate_relation_config(PAIRED_PLACEMENT['relationReward'])
    with pytest.raises(ValueError, match='4 objects'):
        validate_sampler(PAIRED_PLACEMENT['relationGraph'], 2, 3)
    with pytest.raises(ValueError, match='AT and ON_TOP presets'):
        sample_graph(1, PAIRED_PLACEMENT['relationGraph'], preset='holding')
    graph = sample_graph(512, PAIRED_PLACEMENT['relationGraph'],
                         generator=torch.Generator().manual_seed(39))
    at_count = ((graph.edge_relation == AT) & graph.edge_valid).sum(-1)
    top_count = ((graph.edge_relation == ON_TOP) & graph.edge_valid).sum(-1)
    assert ((at_count == 2) & (top_count == 0) | (at_count == 0) & (top_count == 2)).all()
    assert (at_count == 2).any() and (top_count == 2).any()
    assert ((graph.edge_relation == HOLDING) & graph.edge_valid).sum(-1).eq(2).all()
    for row in (top_count == 2).nonzero(as_tuple=True)[0].tolist():
        holds = graph.edge_valid[row] & (graph.edge_relation[row] == HOLDING)
        tops = graph.edge_valid[row] & (graph.edge_relation[row] == ON_TOP)
        sources = graph.edge_dst[row, holds].tolist()
        supports = graph.edge_dst[row, tops].tolist()
        assert len(set(sources + supports)) == 4
    assert checkpoint_metadata(PAIRED_AT['relationReward']) != checkpoint_metadata(
        PAIRED_PLACEMENT['relationReward'])


def test_ontop_putdown_config_keeps_placement_focus_and_isolates_checkpoint():
    validate_sampler(ONTOP_PUTDOWN['relationGraph'])
    validate_relation_config(ONTOP_PUTDOWN['relationReward'])
    expected = deepcopy(PLACEMENT_FOCUS)
    expected['relationReward']['stage1_variant'] = 'scenario_independent_stage1_ontop_putdown'
    expected['relationReward']['hard_skill_training']['ontop_putdown_rsi'] = {
        'phase_range': [.45, .70]}
    expected['templateRsi']['HOLDING_ON_TOP'] = [.4, 0., 0., 0., 0., .1, .4, .1]
    assert ONTOP_PUTDOWN == expected
    assert checkpoint_metadata(ONTOP_PUTDOWN['relationReward']) != checkpoint_metadata(
        PLACEMENT_FOCUS['relationReward'])


def test_ontop_putdown_times_stay_in_configured_phase_range():
    motion = SimpleNamespace(get_motion_length=lambda ids: torch.full_like(ids.float(), 4.5))
    times = sample_ontop_putdown_times(motion, torch.zeros(512, dtype=torch.long), [.45, .70])
    assert times.min() >= 4.5 * .45
    assert times.max() <= 4.5 * .70


def test_ontop_putdown_support_uses_reference_endpoint_and_physical_binding():
    graph = sample_graph(1, ONTOP_PUTDOWN['relationGraph'], preset='holding_ontop')
    edge = ((graph.edge_relation == ON_TOP) & graph.edge_valid).nonzero()[0, 1]
    agent = graph.edge_owner[0, edge].reshape(1)
    logical = int(graph.edge_dst[0, edge] - 2)
    task = SampledOnTopTaskMixin()
    task._relation_cfg = ONTOP_PUTDOWN['relationReward']
    task._reset_ref_slots = {'putDown': (torch.tensor([0]), agent)}
    task._reset_ref_motion_ids = {'putDown': torch.tensor([0])}
    task.relation_runtime = SimpleNamespace(graph=graph)
    task.num_agents = 2
    task._logical_box_order = torch.tensor([[2, 0, 1]])
    task._box_states = torch.zeros(1, 3, 13)
    task._box_size = torch.full((1, 3, 3), .4)
    task._agent_spawn_offsets = torch.tensor([[1., 2., 0.], [3., 4., 0.]])
    task._env_origins = torch.tensor([[10., 20., 0.]])
    task._motion_lib = {'putDown': SimpleNamespace(
        get_motion_length=lambda ids: torch.full_like(ids.float(), 4.5),
        get_obj_motion_state=lambda motion_ids, motion_times: (
            torch.tensor([[.5, .75, .408]]), torch.tensor([[0., 0., 0., 1.]])))}
    task._configure_putdown_ontop_supports()
    physical = task._logical_box_order[0, logical]
    expected_xy = torch.tensor([.5, .75]) + task._agent_spawn_offsets[agent, :2][0] + \
        task._env_origins[0, :2]
    torch.testing.assert_close(task._box_states[0, physical, :2], expected_xy)
    torch.testing.assert_close(task._box_states[0, physical, 2], torch.tensor(.2))


@pytest.mark.parametrize('role_swap', (False, True))
def test_self_sum_keeps_teammate_edge_average(role_swap):
    graph = sample_graph(1, SELF_SUM['relationGraph'], preset='holding_at',
                         role_swap=role_swap, generator=torch.Generator().manual_seed(4))
    runtime = Stage1ContextRuntime(1, graph, SELF_SUM['relationReward'], 'cpu')
    values = torch.ones_like(graph.edge_valid, dtype=torch.float)
    result = runtime.step(values, values, torch.zeros_like(values),
                          region_error=torch.zeros_like(values))
    composite = int(role_swap)
    single = 1 - composite
    torch.testing.assert_close(result['local_task_reward'][0, composite], torch.tensor(1.2))
    torch.testing.assert_close(result['local_task_reward'][0, single], torch.tensor(.6))
    torch.testing.assert_close(result['teammate_task_reward'], torch.full((1, 2), .6))
    torch.testing.assert_close(result['agent_task_reward'][0, composite], torch.tensor(1.14))
    torch.testing.assert_close(result['agent_task_reward'][0, single], torch.tensor(.6))


def test_slow_near_start_hold_and_decay(monkeypatch):
    graph = sample_graph(1, SLOW_NEAR['relationGraph'], preset='holding_at',
                         generator=torch.Generator().manual_seed(4))
    goal = int(graph.edge_dst[0, graph.edge_relation[0] == 7][0] - 5)
    task = SimpleNamespace(_relation_cfg=SLOW_NEAR['relationReward'],
        relation_runtime=SimpleNamespace(graph=graph), device='cpu',
        num_agents=2, num_objects=3, _reset_ref_slots={})
    task._logical_box_order = torch.arange(3).reshape(1, 3)
    task._box_states = _scene(1)[3]
    task._logical_box_values = lambda values, ids: values[ids]
    for step, draw, expected_moved in ((0, .6, True), (600000, .6, True),
                                      (900000, .6, False), (900000, .5, True),
                                      (1200000, .5, False)):
        task._hard_skill_training_step = step
        task._tar_pos = torch.full((1, 2, 3), 9.)
        monkeypatch.setattr(torch, 'rand', lambda *shape, **kw: torch.full(
            shape, draw, device=kw.get('device')))
        SampledOnTopTaskMixin._configure_hard_skill_near_starts(task, torch.tensor([0]))
        assert (task._tar_pos[0, goal, 0] != 9.) == expected_moved


def _scene(n):
    objects = torch.zeros(n, 3, 13)
    objects[..., 6] = 1
    objects[..., 2] = .2
    sizes = torch.full((n, 3, 3), .5)
    sizes[..., 2] = .4
    return (torch.zeros(n, 2, 2, 3), torch.zeros(n, 2, 2, 3),
            torch.zeros(n, 2, 3), objects, sizes, torch.zeros(n, 2, 3))


def test_new_configs_keep_scene_and_checkpoint_contract_separate():
    for env in (NORMALIZED, CURRICULUM, PRESERVED):
        validate_sampler(env['relationGraph'])
        validate_relation_config(env['relationReward'])
        assert env['relationReward']['task_sharing'] == {'self': .9, 'teammate': .1}
        assert env['relationReward']['edge_aggregation'] == 'mean_active'
        for key in BASE:
            if key not in ('relationReward', 'relationGraph', 'templateRsi'):
                assert env[key] == BASE[key]
        with pytest.raises(ValueError, match='reward config differs'):
            check_checkpoint_metadata(
                {'relation_metadata': checkpoint_metadata(BASE['relationReward'])},
                checkpoint_metadata(env['relationReward']))
    assert NORMALIZED['relationGraph'] == BASE['relationGraph']
    assert NORMALIZED['templateRsi'] == BASE['templateRsi']
    assert PRESERVED['relationGraph'] == CURRICULUM['relationGraph']
    assert PRESERVED['templateRsi'] == CURRICULUM['templateRsi']
    assert set(PRESERVED['relationReward']['hard_skill_training']) == {'near_start', 'climb_rsi'}
    assert PRESERVED['relationReward']['hard_skill_training'] == {
        key: CURRICULUM['relationReward']['hard_skill_training'][key]
        for key in ('near_start', 'climb_rsi')}
    with pytest.raises(ValueError, match='reward config differs'):
        check_checkpoint_metadata(
            {'relation_metadata': checkpoint_metadata(CURRICULUM['relationReward'])},
            checkpoint_metadata(PRESERVED['relationReward']))


@pytest.mark.parametrize('preset', ('holding', 'sit', 'climb', 'holding_at', 'holding_ontop'))
def test_reward_preserved_curriculum_uses_normalized_state_and_progress(preset):
    graph = sample_graph(4, PRESERVED['relationGraph'], preset=preset,
                         generator=torch.Generator().manual_seed(11))
    scene = _scene(4)
    scene[2][..., 0] = .8
    scene[1][..., 2] = .15
    scene[3][:, 0, 0] = 1.2
    scene[3][:, 1, 1] = 1.5
    scene[5][..., 0] = 2.5
    old_phi, old_diag = evaluate_interaction_edges(
        *scene, graph, NORMALIZED['relationReward'], .94)
    new_phi, new_diag = evaluate_interaction_edges(
        *scene, graph, PRESERVED['relationReward'], .94)
    torch.testing.assert_close(new_phi, old_phi)
    torch.testing.assert_close(new_diag['progress'], old_diag['progress'])
    torch.testing.assert_close(
        interaction_own_success(new_phi, new_diag['z_error'],
            new_diag['feet_height_error'], graph, PRESERVED['relationReward'],
            new_diag['region_error']),
        interaction_own_success(old_phi, old_diag['z_error'],
            old_diag['feet_height_error'], graph, NORMALIZED['relationReward'],
            old_diag['region_error']))


def test_sit_region_accepts_lateral_position_but_rejects_high_root():
    graph = sample_graph(1, NORMALIZED['relationGraph'], preset='sit',
                         generator=torch.Generator().manual_seed(4))
    edge = ((graph.edge_relation == SIT) & graph.edge_valid).nonzero()[0, 1]
    owner = graph.edge_owner[0, edge]
    scene = _scene(1)
    scene[2][0, owner] = torch.tensor([.19, 0., .52])
    phi, diag = evaluate_interaction_edges(*scene, graph, NORMALIZED['relationReward'], .94)
    assert phi[0, edge] < .9
    assert diag['region_error'][0, edge] == 0
    success = interaction_own_success(phi, diag['z_error'], diag['feet_height_error'],
        graph, NORMALIZED['relationReward'], diag['region_error'])
    assert success[0, edge]
    assert not interaction_own_success(phi, diag['z_error'], diag['feet_height_error'],
        graph, BASE['relationReward'], diag['region_error'])[0, edge]

    scene[2][0, owner] = torch.tensor([0., 0., .60])
    phi, diag = evaluate_interaction_edges(*scene, graph, NORMALIZED['relationReward'], .94)
    assert phi[0, edge] > .9
    assert not interaction_own_success(phi, diag['z_error'], diag['feet_height_error'],
        graph, NORMALIZED['relationReward'], diag['region_error'])[0, edge]


def test_two_edge_and_one_edge_success_have_same_local_scale():
    graph = sample_graph(64, NORMALIZED['relationGraph'],
                         generator=torch.Generator().manual_seed(9))
    count = owner_sum(graph.edge_valid.float(), graph)
    assert (count == 1).any() and (count == 2).any()
    reward = NORMALIZED['relationReward']
    runtime = Stage1ContextRuntime(64, graph, reward, 'cpu')
    values = torch.ones_like(graph.edge_valid, dtype=torch.float)
    zeros = torch.zeros_like(values)
    result = runtime.step(values, values, zeros, zeros, zeros)
    torch.testing.assert_close(result['local_task_reward'], torch.full_like(count, .6))
    torch.testing.assert_close(result['agent_task_reward'], torch.full_like(count, .6))


def test_curriculum_strengthens_far_progress_and_near_climb_height_signal():
    graph = sample_graph(1, CURRICULUM['relationGraph'], preset='holding_at',
                         generator=torch.Generator().manual_seed(3))
    edge = ((graph.edge_relation == 7) & graph.edge_valid).nonzero()[0, 1]
    source = graph.edge_src[0, edge] - 2
    goal = graph.edge_dst[0, edge] - 5
    scene = _scene(1)
    scene[3][0, source, 0] = 4.
    scene[5][0, goal, 2] = .2
    old_phi, old_diag = evaluate_interaction_edges(*scene, graph, NORMALIZED['relationReward'], .94)
    new_phi, new_diag = evaluate_interaction_edges(*scene, graph, CURRICULUM['relationReward'], .94)
    torch.testing.assert_close(old_phi[0, edge], new_phi[0, edge])
    assert new_diag['progress'][0, edge] > old_diag['progress'][0, edge]

    graph = sample_graph(1, CURRICULUM['relationGraph'], preset='climb',
                         generator=torch.Generator().manual_seed(5))
    edge = ((graph.edge_relation == CLIMB) & graph.edge_valid).nonzero()[0, 1]
    owner = graph.edge_owner[0, edge]
    scene = _scene(1)
    scene[2][0, owner] = torch.tensor([.7, 0., .94])
    scene[1][0, owner, :, 2] = .1
    old_phi, _ = evaluate_interaction_edges(*scene, graph, NORMALIZED['relationReward'], .94)
    new_phi, _ = evaluate_interaction_edges(*scene, graph, CURRICULUM['relationReward'], .94)
    assert new_phi[0, edge] > old_phi[0, edge]


def test_near_start_places_at_goal_and_ontop_support_in_configured_ranges():
    graph = sample_graph(64, CURRICULUM['relationGraph'],
                         generator=torch.Generator().manual_seed(7))
    config = deepcopy(CURRICULUM['relationReward'])
    config['hard_skill_training']['near_start']['probability_start'] = 1.
    task = SimpleNamespace(_relation_cfg=config, _edge_steps=0, _is_eval=False,
        relation_runtime=SimpleNamespace(graph=graph), device='cpu',
        num_agents=2, num_objects=3, _reset_ref_slots={})
    task._logical_box_order = torch.arange(3).expand(64, -1)
    task._box_states = _scene(64)[3]
    task._box_states[:, 0, 0] = 0.
    task._box_states[:, 1, 0] = 3.
    task._box_states[:, 2, 1] = 4.
    task._tar_pos = torch.full((64, 2, 3), 9.)
    task._logical_box_values = lambda values, ids: values[ids]
    SampledOnTopTaskMixin._configure_hard_skill_near_starts(task, torch.arange(64))
    at_seen = top_seen = 0
    for env in range(64):
        for edge in range(graph.edge_valid.shape[1]):
            if not graph.edge_valid[env, edge]:
                continue
            relation = graph.edge_relation[env, edge]
            source = graph.edge_src[env, edge] - 2
            if relation == 7:
                goal = graph.edge_dst[env, edge] - 5
                distance = (task._box_states[env, source, :2] - task._tar_pos[env, goal, :2]).norm()
                assert .75 <= distance <= 1.5
                at_seen += 1
            elif relation == ON_TOP:
                support = graph.edge_dst[env, edge] - 2
                distance = (task._box_states[env, source, :2] - task._box_states[env, support, :2]).norm()
                assert 1.1 <= distance <= 1.8
                top_seen += 1
    assert at_seen and top_seen


def test_climb_rsi_can_sample_the_final_frames_excluded_by_original_rsi():
    motion_lib = SimpleNamespace(get_motion_length=lambda ids: torch.full_like(ids.float(), 2.))
    ids = torch.zeros(100, dtype=torch.long)
    original = torch.full((100,), .3)
    late = mix_late_climb_rsi_times(motion_lib, ids, original.clone(),
        {'late_fraction': 1., 'phase_range': [.65, .98]})
    assert late.min() >= 1.3 and late.max() <= 1.96
    torch.testing.assert_close(mix_late_climb_rsi_times(motion_lib, ids, original.clone(),
        {'late_fraction': 0., 'phase_range': [.65, .98]}), original)


def test_near_start_preserves_put_down_reference_goal():
    graph = sample_graph(1, CURRICULUM['relationGraph'], preset='holding_at')
    reward = deepcopy(CURRICULUM['relationReward'])
    reward['hard_skill_training']['near_start']['probability_start'] = 1.
    task = SimpleNamespace(_relation_cfg=reward, _is_eval=False,
        relation_runtime=SimpleNamespace(graph=graph), device='cpu',
        num_agents=2, num_objects=3,
        _reset_ref_slots={'putDown': (torch.tensor([0]), torch.tensor([0]))})
    task._logical_box_order = torch.arange(3).expand(1, -1)
    task._box_states = _scene(1)[3]
    task._tar_pos = torch.full((1, 2, 3), 9.)
    task._logical_box_values = lambda values, ids: values[ids]
    SampledOnTopTaskMixin._configure_hard_skill_near_starts(task, torch.tensor([0]))
    torch.testing.assert_close(task._tar_pos, torch.full_like(task._tar_pos, 9.))
