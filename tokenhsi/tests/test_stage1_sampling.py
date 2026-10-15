"""Distance distributions, ON_TOP binding, legacy behavior and resume isolation."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from test_stage1 import CFG
import test_task_role_message as reference
from utils.stage1_sampling import (sample_source_xy, sample_goal_xy,
    reset_ontop_support_xy, check_sampling_resume)
from utils.task_role_spec import task_graph_packet
from utils.unified_training import validate_unified_env


def legacy_config():
    env = deepcopy(CFG)
    env['box']['reset'].pop('samplingMode')
    env['box']['reset']['ownerLocoDistanceRange'] = [1., 2.]
    for name in ('HOLDING_AT', 'HOLDING_ON_TOP'):
        env['templateRsi'][name] = [.4, 0., 0., 0., 0., .1, .4, .1]
    return env


def test_config_and_resume_both_generations():
    old = legacy_config()
    for env in (CFG, old):
        validate_unified_env(env)
        check_sampling_resume({'relation_experiment_config': {'env': env}}, env)
    for source, dest in ((old, CFG), (CFG, old)):
        with pytest.raises(ValueError, match='sampling/RSI mismatch'):
            check_sampling_resume({'relation_experiment_config': {'env': source}}, dest)
    bad = deepcopy(CFG)
    bad['templateRsi'] = old['templateRsi']
    with pytest.raises(ValueError, match='50/10/30/10'):
        validate_unified_env(bad)
    with pytest.raises(ValueError, match='saved experiment'):
        check_sampling_resume({}, CFG)


def test_source_matches_native_radial_law_not_area_uniform():
    torch.manual_seed(12)
    templates = torch.arange(1, 5).repeat_interleave(10000)
    root = torch.randn(len(templates), 2) * 20
    delta = sample_source_xy(root, templates) - root
    radius = delta.norm(dim=-1)
    for task in range(1, 5):
        limit = 5 if task == 1 else 10
        r = radius[templates == task]
        assert r.min() >= 1 - 1e-5 and r.max() <= limit + 1e-5
        assert abs(r.mean().item() - (1 + limit) / 2) < .08
        assert (delta[templates == task].mean(0).abs() < .15).all()


def test_goal_square_frame_and_native_exclusions():
    torch.manual_seed(13)
    center = torch.randn(12000, 2) * 100
    # Out-of-square roots remove rejection, permitting distribution checks.
    goal = sample_goal_xy(center + 20, center - 20, center) - center
    assert goal.abs().max() <= 4.5001
    assert (goal.abs() > 4.).any()  # Legacy +-2 m square cannot pass.
    assert (goal.mean(0).abs() < .06).all()
    assert ((goal.square().mean(0) - 6.75).abs() < .15).all()
    root, payload = center.clone(), center + torch.tensor([1.5, .5])
    goal = sample_goal_xy(root, payload, center)
    assert ((goal - root).norm(dim=-1) >= 1 - 1e-5).all()
    assert ((goal - payload).norm(dim=-1) >= 1 - 1e-5).all()


@pytest.mark.parametrize('skill', ['loco', 'pickUp', 'carryWith', 'putDown', 'sit'])
def test_ontop_only_target_xy_partial_reset_and_permuted_binding(skill):
    graph = reference.all_pairs()
    records = task_graph_packet(graph, 16).reshape(16, 2, 5).long()
    order = torch.tensor([3, 2, 1, 0]).expand(16, -1)
    ids = torch.tensor([1, 3, 6, 10, 11, 14, 15])
    env, owner = ids.repeat_interleave(2), torch.arange(2).repeat(len(ids))
    origins = torch.randn(16, 3) * 50
    task = SimpleNamespace(num_envs=16, num_agents=2, _logical_box_order=order,
        _box_states=torch.randn(16, 4, 13), _env_origins=origins,
        _humanoid_root_states=origins[:, None].expand(-1, 2, -1).clone(),
        relation_runtime=SimpleNamespace(graph=graph),
        _scenario_physical_object=lambda e, a: order[e, records[e, a, 3] - 2],
        _collect_random_slots=lambda skills: (env, owner) if skill in skills else None)
    before = task._box_states.clone()
    reset_ontop_support_xy(task)
    expected = before.clone()
    changed = 0
    for e, a in zip(env.tolist(), owner.tolist()):
        if records[e, a, 1] != 3 or skill not in ('loco', 'pickUp', 'carryWith'):
            continue
        support = order[e, records[e, a, 4] - 2]
        payload = task._scenario_physical_object(e, a)
        xy = task._box_states[e, support, :2]
        assert ((xy - origins[e, :2]).abs() <= 4.5001).all()
        assert (xy - task._humanoid_root_states[e, a, :2]).norm() >= .99999
        assert (xy - before[e, payload, :2]).norm() >= .99999
        expected[e, support, :2] = xy
        changed += 1
    assert changed > 0 if skill in ('loco', 'pickUp', 'carryWith') else changed == 0
    torch.testing.assert_close(task._box_states, expected, atol=0, rtol=0)
