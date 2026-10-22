from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges
from env.tasks.multi_agent.edge_ontop_reward import inner_xy_region_error
from env.tasks.multi_agent.edge_ontop_task import SampledOnTopTaskMixin
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from utils.edge_interaction_spec import CLIMB
from utils.edge_ontop_spec import ON_TOP
from utils.edge_scenario_spec import sample_graph
from utils.edge_stage1_spec import validate_sampler
from utils.relation_task_spec import (checkpoint_metadata, check_checkpoint_metadata,
    validate_relation_config)


ROOT = Path(__file__).resolve().parents[1]
BASE = yaml.safe_load((ROOT / 'data/cfg/multi_agent/'
    'approach_scenario_independent_with_climb.yaml').read_text())['env']
ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/'
    'approach_scenario_stage1_plane.yaml').read_text())['env']
CFG = ENV['relationReward']


def _scene():
    objects = torch.zeros(1, 3, 13)
    objects[..., 6] = 1
    objects[..., 2] = .2
    sizes = torch.full((1, 3, 3), .5)
    sizes[..., 2] = .4
    hands = torch.zeros(1, 2, 2, 3)
    feet = torch.zeros(1, 2, 2, 3)
    roots = torch.zeros(1, 2, 3)
    goals = torch.zeros(1, 2, 3)
    return hands, feet, roots, objects, sizes, goals


def _evaluate(scene, graph):
    return evaluate_interaction_edges(*scene, graph, CFG, .94)


def test_config_changes_only_plane_success_and_keeps_checkpoint_separate():
    validate_sampler(ENV['relationGraph'])
    validate_relation_config(CFG)
    for key, value in BASE.items():
        if key != 'relationReward':
            assert ENV[key] == value
    original = deepcopy(BASE['relationReward'])
    original['stage1_variant'] = CFG['stage1_variant']
    original['ontop'] = CFG['ontop']
    original['climb'] = CFG['climb']
    assert CFG == original
    assert CFG['task_sharing'] == {'self': .9, 'teammate': .1}
    with pytest.raises(ValueError, match='reward config differs'):
        check_checkpoint_metadata(
            {'relation_metadata': checkpoint_metadata(BASE['relationReward'])},
            checkpoint_metadata(CFG))


def test_yaw_aligned_inner_region():
    support = torch.tensor([[0., 0., .2, 0., 0., .70710678, .70710678]])
    size = torch.tensor([[.6, .4, .4]])
    point_inside = torch.tensor([[0., .23]])
    point_outside = torch.tensor([[0., .25]])
    assert inner_xy_region_error(point_inside, support, size, .1).item() == pytest.approx(0.)
    assert inner_xy_region_error(point_outside, support, size, .1).item() == pytest.approx(.01)


def test_ontop_region_success_changes_reset_and_saturates_with_team_mix():
    graph = sample_graph(1, ENV['relationGraph'], preset='holding_ontop',
        generator=torch.Generator().manual_seed(2))
    top = ((graph.edge_relation == ON_TOP) & graph.edge_valid).nonzero()[0, 1]
    source = graph.edge_src[0, top] - 2
    support = graph.edge_dst[0, top] - 2
    scene = _scene()
    scene[3][0, source, :3] = torch.tensor([.18, .18, .6])
    phi, diag = _evaluate(scene, graph)
    assert phi[0, top] < .9
    assert diag['region_error'][0, top] == 0
    runtime = Stage1ContextRuntime(1, graph, CFG, 'cpu')
    args = (phi, diag['z_error'], diag['feet_height_error'], diag['region_error'])
    runtime.reset(torch.tensor([0]), *args)
    assert runtime.own_success[0, top]
    out = runtime.step(phi, diag['progress'], *args[1:])
    assert out['own_success'][0, top]
    assert out['reward_saturated'][0, top]
    torch.testing.assert_close(out['agent_task_reward'],
        .9 * out['local_task_reward'] + .1 * out['local_task_reward'].flip(-1))

    scene[3][0, source, 0] = .21
    phi, diag = _evaluate(scene, graph)
    assert diag['region_error'][0, top] > 0
    assert not runtime.step(phi, diag['progress'], diag['z_error'],
        diag['feet_height_error'], diag['region_error'])['own_success'][0, top]
    scene[3][0, source, 0] = .18
    scene[3][0, source, 2] = .602
    phi, diag = _evaluate(scene, graph)
    assert not runtime.step(phi, diag['progress'], diag['z_error'],
        diag['feet_height_error'], diag['region_error'])['own_success'][0, top]


def test_climb_region_root_height_and_feet_conditions():
    graph = sample_graph(1, ENV['relationGraph'], preset='climb',
        generator=torch.Generator().manual_seed(3))
    climb = ((graph.edge_relation == CLIMB) & graph.edge_valid).nonzero()[0, 1]
    owner = graph.edge_owner[0, climb]
    scene = _scene()
    scene[2][0, owner] = torch.tensor([.2, .2, 1.34])
    scene[1][0, owner, :, 2] = .4
    runtime = Stage1ContextRuntime(1, graph, CFG, 'cpu')

    def success():
        phi, diag = _evaluate(scene, graph)
        out = runtime.step(phi, diag['progress'], diag['z_error'],
            diag['feet_height_error'], diag['region_error'])
        return bool(out['own_success'][0, climb]), phi[0, climb].item()

    passed, phi = success()
    assert passed and phi < .6
    scene[2][0, owner, 0] = .23
    assert not success()[0]
    scene[2][0, owner, 0] = .2
    scene[2][0, owner, 2] = 1.541
    assert not success()[0]
    scene[2][0, owner, 2] = 1.34
    scene[1][0, owner, :, 2] = .471
    assert not success()[0]


def test_trace_sharing_matches_actual_stage1_reward():
    graph = sample_graph(1, ENV['relationGraph'], preset='holding_ontop')

    class Task(SampledOnTopTaskMixin):
        pass

    task = Task()
    task.num_agents = 2
    task.num_objects = 3
    task._edge_stage1 = True
    task._edge_interaction = True
    task._relation_cfg = CFG
    task._logical_box_order = torch.tensor([[0, 1, 2]])
    task._ontop_last_diag = {
        'signed_gap': torch.zeros(1, 4), 'target': torch.zeros(1, 4, 3)}
    task.relation_runtime = SimpleNamespace(graph=graph,
        last_result={'local_task_reward': torch.tensor([[1., 2.]])})
    for row in task._ontop_trace_bindings(1)[0]:
        owner = row['binding_owner']
        assert row['self_task_contribution'] == pytest.approx((.9, 1.8)[owner])
        assert row['other_task_contribution'] == pytest.approx((.2, .1)[owner])
