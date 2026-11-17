from copy import deepcopy

import pytest
import torch

from utils import stage2_spec
from utils.coupled_push_spec import check_checkpoint, checkpoint_contract, group_sizes
from utils.experiments import native_config, validate_experiment
from utils.sensor_safety_spec import NAME, BASE, cliff_roles, room_roles
from utils.sensor_stage2_spec import paired_people, sample_surfaces, CLIFF
from utils.warehouse_sensor import sample_roles_graph


def test_config_changes_only_approved_sampling():
    cfg = native_config(NAME)
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    old = native_config(BASE)
    actual = deepcopy(cfg)
    actual['experiment']['name'] = BASE
    scene = actual['env']['sensorScene']
    scene['mixture'] = old['env']['sensorScene']['mixture']
    del scene['bridge']['curriculum']
    del scene['cliff']['sampling']
    del scene['room']['sampling']
    scene['room']['pickup_cross_probability'] = .5
    del scene['warehouse']['worker_pairing_probability']
    assert actual == old
    train = native_config(NAME, True)
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)


def test_role_exposure_and_token_budget():
    torch.manual_seed(42)
    env = native_config(NAME)['env']
    scene = env['sensorScene']
    n = 30000
    sizes = torch.tensor([.5, .5, .4]).expand(n, 4, 3)
    ranges = env['box']['build']['taskSizeRanges']
    probabilities = env['relationGraph']['template_probabilities']
    groups, pre, tasks = cliff_roles(sizes, probabilities, scene['cliff']['sampling']['all_edge_probability'])
    counts = (tasks != 5).sum(-1)
    assert set(counts.tolist()) == {2, 4}
    coupled = (tasks == 5) & (group_sizes(groups) == 2)
    assert torch.equal(coupled, tasks == 5)
    assert not paired_people(pre).any()
    _, start, unreachable = sample_surfaces(tasks, pre, groups, torch.full((n,), CLIFF), coupled)
    assert torch.equal(unreachable, tasks != 5)
    assert torch.equal(start > 0, unreachable)
    assert abs(float(unreachable.float().mean())-.7) < .008
    domain, groups, pre, tasks = room_roles(sizes, ranges, probabilities, scene['room']['sampling'], 'random_scenario')
    before = paired_people(pre)
    coupled = (tasks == 5) & (group_sizes(groups) > 1)
    independent = ~before & ~coupled
    assert before[domain == 0].all()
    assert not ((tasks == 5) & independent).any()
    assert set(group_sizes(groups)[coupled].tolist()) == {2, 3, 4}
    torch.testing.assert_close(torch.tensor([independent.float().mean(), before.float().mean(), coupled.float().mean()]),
        torch.tensor([.145833333, .45, .404166667]), atol=.008, rtol=0)
    assert (stage2_spec.required_tokens(groups, pre, tasks) <= 8).all()
    groups, pre, tasks, _ = sample_roles_graph(sizes, ranges, probabilities, 'random_scenario', .8)
    assert ((tasks == 5).sum(-1) == 1).all()
    assert ((pre >= 0).sum(-1) <= 1).all()
    assert abs(float(paired_people(pre).float().mean())-.48) < .008


def test_resume_only_allows_approved_old_to_new_contract():
    current = native_config(NAME)['env']
    old = dict(coupled_push_contract=checkpoint_contract(native_config(BASE)['env']))
    check_checkpoint(old, current)
    check_checkpoint(dict(coupled_push_contract=checkpoint_contract(current)), current)
    with pytest.raises(ValueError):
        check_checkpoint(old, current, inference=True)
    for key in ('relationReward', 'sensorScene'):
        changed = deepcopy(current)
        changed[key]['unapproved'] = True
        with pytest.raises(ValueError):
            check_checkpoint(old, changed)
    with pytest.raises(ValueError):
        check_checkpoint(dict(coupled_push_contract=checkpoint_contract(current)), native_config(BASE)['env'])
