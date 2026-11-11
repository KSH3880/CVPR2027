from copy import deepcopy

import pytest
import torch

from utils.coupled_push_spec import checkpoint_contract, check_checkpoint
from utils.experiments import native_config, validate_experiment
from utils.premature_push import motion_penalty, check_contact_resume


BASE = 'stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg'
CONFIG = dict(weight=.5, speed_threshold=.05, force_epsilon=1e-6)


def sample():
    velocity = torch.tensor([[[.3, 0.]]]).expand(1, 4, 2).clone()
    forces = torch.zeros(1, 4, 2, 2)
    forces[0, :, 0, 0] = torch.tensor([-6., -3., -1., 0.])
    near = torch.ones(1, 4, 2, dtype=torch.bool)
    near[0, 3] = False
    groups = torch.zeros(1, 4, dtype=torch.long)
    eligible = torch.ones(1, 4, dtype=torch.bool)
    return velocity, forces, near, groups, eligible


def test_force_shares_group_cost_and_ignores_braking_and_distant_hands():
    args = sample()
    penalty, moving, force = motion_penalty(*args, CONFIG)
    torch.testing.assert_close(penalty, torch.tensor([[-.075, -.0375, -.0125, 0.]]))
    torch.testing.assert_close(penalty.sum(), torch.tensor(-.5*(.3-.05)))
    assert moving.all()
    args[1][0, 0, 1, 0] = 100.
    args[1][0, 3, 0, 0] = -100.
    torch.testing.assert_close(motion_penalty(*args, CONFIG)[2], force)


@pytest.mark.parametrize('case', ['ready', 'slow', 'no_force', 'tiny_force', 'solo'])
def test_no_penalty_cases(case):
    args = sample()
    if case == 'ready':
        args[2].fill_(True)
    elif case == 'slow':
        args[0].fill_(0.)
        args[0][..., 0] = .05
    elif case == 'no_force':
        args[1].zero_()
    elif case == 'tiny_force':
        args[1].mul_(1e-9)
    else:
        args[4].fill_(False)
    assert not motion_penalty(*args, CONFIG)[0].any()


def test_rotation_shuffle_and_separate_groups():
    args = sample()
    expected = motion_penalty(*args, CONFIG)[0]
    velocity, forces = args[:2]
    velocity.copy_(velocity[..., [1, 0]])
    forces.copy_(forces[..., [1, 0]])
    torch.testing.assert_close(motion_penalty(*args, CONFIG)[0], expected)
    order = torch.tensor([3, 1, 0, 2])
    torch.testing.assert_close(motion_penalty(*(x[:, order] for x in args), CONFIG)[0], expected[:, order])
    args[3][0] = torch.tensor([0, 0, 2, 3])
    args[4][0, 2:] = False
    assert not motion_penalty(*args, CONFIG)[0].any()


def test_contact_config_only_adds_penalty_and_identity():
    cfg = native_config('stage2_coupled_contact')
    validate_experiment('stage2_coupled_contact', cfg['env'], native_config('stage2_coupled_contact', True))
    assert cfg['env']['stage2JointPush'].pop('premature_motion_penalty') == CONFIG
    cfg['experiment']['name'] = BASE
    assert cfg == native_config(BASE)
    train = native_config('stage2_coupled_contact', True)
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)


def source():
    cfg = native_config(BASE)
    return dict(epoch=17000, relation_experiment_config=cfg,
                coupled_push_contract=checkpoint_contract(cfg['env']))


def test_resume_exception_is_one_way_and_only_for_matching_17k():
    env = native_config('stage2_coupled_contact')['env']
    original = source()
    check_contact_resume(original, env)
    with pytest.raises(ValueError):
        check_checkpoint(original, env)
    contact = dict(coupled_push_contract=checkpoint_contract(env))
    check_contact_resume(contact, env)
    check_checkpoint(contact, env)
    with pytest.raises(ValueError):
        check_checkpoint(contact, native_config(BASE)['env'])
    for change in ('epoch', 'fusion', 'distance', 'people', 'source_name'):
        wrong = deepcopy(original)
        if change == 'epoch':
            wrong['epoch'] = 18000
        elif change == 'fusion':
            wrong['coupled_push_contract']['graph']['human_relation_fusion'] = 'learnedbg'
        elif change == 'distance':
            wrong['coupled_push_contract']['coupled']['goal_distance'] = [2., 10.]
        elif change == 'people':
            wrong['coupled_push_contract']['people'] = 8
        else:
            wrong['relation_experiment_config']['experiment']['name'] = 'other'
        with pytest.raises(ValueError):
            check_contact_resume(wrong, env)


@pytest.mark.parametrize('key,value', [('weight', -1.), ('speed_threshold', float('nan')), ('force_epsilon', 0.)])
def test_invalid_penalty_config_rejected(key, value):
    cfg = native_config('stage2_coupled_contact')
    cfg['env']['stage2JointPush']['premature_motion_penalty'][key] = value
    with pytest.raises(ValueError):
        validate_experiment('stage2_coupled_contact', cfg['env'], native_config('stage2_coupled_contact', True))
