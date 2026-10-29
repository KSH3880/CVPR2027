from copy import deepcopy
import pytest
import torch

from before_test_utils import config
from utils.before_spec import validate_env, checkpoint_contract
from utils.before_spec import check_checkpoint
from utils.before_sampling import sample_distance_modes
from utils.before_relative_sampling import DISTANCE_MIXTURE, pair_distances, sample_layout

NAME = 'stage2_before_nonemask_rsi433_near50'




def test_half_pair_modes_and_rsi_are_independent():
    from utils.before_pickup_carry_rsi import select_skills
    torch.manual_seed(50)
    pre = torch.tensor([[-1, 0]]).expand(100000, -1)
    templates = torch.tensor([[3, 4]]).expand_as(pre)
    modes = sample_distance_modes(pre, DISTANCE_MIXTURE)
    assert not modes[:, 0].any()
    assert .49 < modes[:, 1].float().mean() < .51
    assert torch.equal(sample_distance_modes(pre, DISTANCE_MIXTURE, force_near=True), (pre >= 0).long())
    _, _, skills = select_skills(pre, templates, torch.ones(100000, 2, 3, dtype=torch.bool))
    for mode in (0, 1):
        selected = skills[:, 1][modes[:, 1] == mode]
        torch.testing.assert_close(torch.bincount(selected, minlength=3)/len(selected),
                                   torch.tensor([.4, .3, .3]), atol=.01, rtol=0)


@pytest.mark.parametrize('template,advanced', [(1, False), (2, False), (4, False), (4, True)])
def test_rejection_enforces_sum_and_retains_reference_offset(template, advanced):
    torch.manual_seed(42)
    n = 10000
    leader = torch.zeros(n, 2)
    support = torch.full((n, 2), 2.)
    follower = torch.ones(n, 2)
    payload = follower+torch.tensor([.4, .2])
    templates = torch.full((n,), template)
    top = torch.full((n,), advanced)
    near = torch.ones(n, dtype=torch.bool)
    support, actual, box = sample_layout(leader, support, follower, payload, templates, near, top)
    first, second = pair_distances(leader, support, actual, box, templates)
    assert (second < first).all()
    assert ((first >= 1.-1e-5) & (first <= 10.+1e-5)).all()
    if advanced:
        torch.testing.assert_close(box-actual, payload-follower, atol=1e-6, rtol=1e-5)
    if template == 4:
        assert ((actual-support).norm(dim=-1)+(actual-box).norm(dim=-1) < first).all()


def test_original_mode_keeps_support_and_native_ranges_without_ordering():
    torch.manual_seed(11)
    n = 10000
    leader = torch.zeros(n, 2); support = torch.ones(n, 2)
    templates = torch.full((n,), 4)
    result, follower, payload = sample_layout(leader, support, leader, leader,
        templates, torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool))
    assert torch.equal(result, support)
    for distance in ((follower-support).norm(dim=-1), (payload-follower).norm(dim=-1)):
        assert ((distance >= 1.-1e-5) & (distance <= 10.+1e-5)).all()
        assert distance.max() > 9.9
    first, second = pair_distances(leader, support, follower, payload, templates)
    assert (second >= first).any()


def test_ontop_requires_sum_not_each_distance_individually():
    first, second = pair_distances(torch.tensor([[5., 0.]]), torch.zeros(1, 2),
        torch.tensor([[3., 0.]]), torch.tensor([[6., 0.]]), torch.tensor([4]))
    assert first.item() == 5. and second.item() == 6.
