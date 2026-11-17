from copy import deepcopy

import pytest
import torch

from utils import stage2_spec
from utils.coupled_push_spec import check_checkpoint, checkpoint_contract, group_sizes
from utils.experiments import native_config, validate_experiment
from utils.sensor_stage2_spec import ROOM, ROOM_SAMPLING, paired_people, room_roles, sample_surfaces


NAME = 'stage2_sensor_900_room_before'
BASE = 'stage2_sensor_900_bridge_curriculum'


def test_config_changes_only_room_sampling_and_name():
    config = native_config(NAME)
    original = deepcopy(config)
    assert original['env']['sensorScene']['room'].pop('sampling') == ROOM_SAMPLING
    original['experiment']['name'] = BASE
    assert original == native_config(BASE)
    train = native_config(NAME, True)
    original_train = deepcopy(train)
    original_train['params']['config']['name'] = 'Stage2Sensor900BridgeCurriculum'
    assert original_train == native_config(BASE, True)
    validate_experiment(NAME, config['env'], train)


def test_room_sampling_preserves_coupled_and_expected_exposure():
    torch.manual_seed(42)
    env = native_config(NAME)['env']
    n = 60000
    sizes = torch.full((n, 4, 3), .5)
    sizes[..., 2] = .4
    ranges = env['box']['build']['taskSizeRanges']
    probabilities = env['relationGraph']['template_probabilities']
    domain, groups, pre, tasks = stage2_spec.sample_episode(sizes, ranges, probabilities, num_objects=8)
    original = [x.clone() for x in (groups, pre, tasks)]
    groups, pre, tasks = room_roles(sizes, ranges, probabilities, domain, groups, pre, tasks)
    before = paired_people(pre)
    coupled = (tasks == 5) & (group_sizes(groups) > 1)
    independent = ~before & ~coupled
    assert not ((tasks == 5) & independent).any()
    assert before[domain == 0].all()
    assert torch.equal(groups[domain != 0], original[0][domain != 0])
    assert torch.equal(pre[domain != 0], original[1][domain != 0])
    assert torch.equal(tasks[coupled], original[2][coupled])
    assert set(group_sizes(groups)[coupled].tolist()) == {2, 3, 4}
    torch.testing.assert_close(torch.stack([independent.float().mean(), before.float().mean(), coupled.float().mean()]),
        torch.tensor([35/288, .5, 109/288]), rtol=0, atol=.008)
    roles = torch.arange(4).expand(n, -1)
    surface, start, unreachable = sample_surfaces(tasks, pre, roles, torch.full((n,), ROOM), torch.zeros_like(before))
    assert torch.equal(start == 0, coupled)
    assert torch.equal(surface, start)
    assert not unreachable.any()
    packet = stage2_spec.compose_graph(tasks, groups, pre, 8)
    stage2_spec.physical_mapping(tasks, groups, pre, packet)
    assert (stage2_spec.required_tokens(groups, pre, tasks) <= 8).all()


def test_resume_accepts_only_the_approved_room_change():
    old = native_config(BASE)['env']
    current = native_config(NAME)['env']
    weights = dict(coupled_push_contract=checkpoint_contract(old))
    check_checkpoint(weights, current)
    with pytest.raises(ValueError):
        check_checkpoint(weights, current, inference=True)
    for key in ('mixture', 'bridge', 'sensor'):
        changed = deepcopy(current)
        changed['sensorScene'][key]['unapproved'] = True
        with pytest.raises(ValueError):
            check_checkpoint(weights, changed)
    changed = deepcopy(current)
    changed['relationReward']['unapproved'] = True
    with pytest.raises(ValueError):
        check_checkpoint(weights, changed)
    check_checkpoint(dict(coupled_push_contract=checkpoint_contract(current)), current)
    with pytest.raises(ValueError):
        check_checkpoint(dict(coupled_push_contract=checkpoint_contract(current)), old)
