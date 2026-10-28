from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from test_before_adapter import config
from utils.before_spec import validate_env, checkpoint_contract
from utils.stage2_spec import check_checkpoint
from utils.before_pickup_carry_rsi import SPEC, select_skills
from env.tasks.multi_agent.before_pickup_carry_rsi import screen

NAME = 'stage2_before_ground_mix226_atprogress_nonemask_rsi433_1200'
BASE = 'stage2_before_ground_mix226_atprogress_nonemask_1200'


def test_config_changes_only_follower_rsi_and_separates_checkpoints():
    old, new = config(BASE), config(NAME)
    validate_env(new['env'])
    comparable = deepcopy(new['env'])
    assert comparable['before'].pop('follower_pickup_carry_rsi') == SPEC
    assert comparable == old['env'] and new['sim'] == old['sim']
    assert new['experiment']['transfer']['checkpoint'].endswith('00078000.pth')
    a, b = config(BASE, 'train/rlg'), config(NAME, 'train/rlg')
    b['params']['config']['name'] = a['params']['config']['name']
    assert a == b
    check_checkpoint({'before_contract': checkpoint_contract(new['env'])}, new['env'])
    for other in (BASE, 'stage2_before_ground_mix226_atprogress_carry50_1200',
                  'stage2_before_ground_mix226_atprogress_fullrsi_1200'):
        env = config(other)['env']
        for source, target in ((new['env'], env), (env, new['env'])):
            with pytest.raises(ValueError, match='BEFORE checkpoint'):
                check_checkpoint({'before_contract': checkpoint_contract(source)}, target)
    new['env']['before']['follower_pickup_carry_rsi']['probabilities'] = [.5, .25, .25]
    with pytest.raises(ValueError, match='40/30/30'):
        validate_env(new['env'])


def test_mixture_only_targets_dependent_ontop_and_falls_back_without_renormalizing():
    pre = torch.tensor([[-1, 0, 0, 0, -1]]).expand(100000, -1)
    templates = torch.tensor([[3, 1, 2, 4, 4]]).expand_as(pre)
    available = torch.ones(*pre.shape, 3, dtype=torch.bool)
    torch.manual_seed(33)
    eligible, requested, selected = select_skills(pre, templates, available)
    assert not selected[:, [0, 1, 2, 4]].any()
    assert torch.equal(requested, selected)
    frequencies = torch.bincount(selected[eligible], minlength=3)/eligible.sum()
    torch.testing.assert_close(frequencies, torch.tensor([.4, .3, .3]), atol=.005, rtol=0)
    available[..., 1] = False
    torch.manual_seed(33)
    _, requested2, selected2 = select_skills(pre, templates, available)
    assert torch.equal(requested2, requested)
    assert not (selected2 == 1).any()
    assert torch.equal(selected2 == 2, selected == 2)
    assert (selected2[requested == 1] == 0).all()


@pytest.mark.parametrize('mode,expected', [('loco', 0), ('pickup', 1), ('carry', 2)])
def test_explicit_evaluation_modes_and_no_eligible(mode, expected):
    pre = torch.tensor([[-1, 0]])
    templates = torch.tensor([[3, 4]])
    available = torch.ones(1, 2, 3, dtype=torch.bool)
    _, _, selected = select_skills(pre, templates, available, mode)
    assert selected.tolist() == [[0, expected]]
    _, _, selected = select_skills(pre, templates*0, available)
    assert not selected.any()


def test_placement_exhaustion_falls_back_to_loco_and_preserves_other_envs():
    t = SimpleNamespace(
        _before_predecessor=torch.tensor([[-1, 0], [-1, 0]]),
        _templates=torch.tensor([[3, 4], [3, 4]]),
        _before_rsi_skill=torch.tensor([[0, 1], [0, 2]]),
        _before_rsi_attempts=torch.zeros(2, 2, dtype=torch.long),
        _before_carry_mask=torch.tensor([[False, True], [False, True]]),
        _before_rsi_counts=torch.zeros(3, 3),
        cfg={'env': {'before': {'follower_pickup_carry_rsi': SPEC}}})
    ids = torch.tensor([0])
    for _ in range(8):
        screen(t, ids, torch.tensor([True]))
    assert t._before_rsi_skill.tolist() == [[0, 0], [0, 2]]
    assert t._before_carry_mask.tolist() == [[False, False], [False, True]]
    assert t._before_rsi_counts[2].tolist() == [0, 1, 0]
    screen(t, ids, torch.tensor([False]))
    assert t._before_rsi_counts[1].tolist() == [1, 0, 0]
