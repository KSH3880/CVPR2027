from unittest.mock import patch

import pytest
import torch

from learning.multi_agent.human_coordination import CoordinationBlock, HumanCoordination, stage2_human_relations
from utils.coupled_push_peer_selection import CoupledPeerSelection, validate_mode


def scene(people=8, envs=2):
    return torch.zeros(envs, people, dtype=torch.long), torch.zeros(envs, people, dtype=torch.bool)


@pytest.mark.parametrize('mode', ['keep3', 'shrink3'])
def test_random_selection_is_stable_and_does_not_consume_global_rng(mode):
    groups, ready = scene()
    selector = CoupledPeerSelection(mode, 'cpu', 42)
    before = torch.get_rng_state().clone()
    selected = selector.update(groups, ready).clone()
    assert torch.equal(before, torch.get_rng_state())
    assert (selected.sum(-1) == 3).all() and not selected.diagonal(dim1=1, dim2=2).any()
    rng = selector.generator.get_state().clone()
    for _ in range(5):
        assert torch.equal(selector.update(groups, ready), selected)
    assert torch.equal(rng, selector.generator.get_state())
    duplicate = CoupledPeerSelection(mode, 'cpu', 42)
    assert torch.equal(duplicate.update(groups, ready), selected)


@pytest.mark.parametrize('mode', ['keep3', 'shrink3'])
def test_success_replaces_only_the_completed_slot_when_other_unready_people_exist(mode):
    groups, ready = scene(envs=1)
    selector = CoupledPeerSelection(mode, 'cpu', 43)
    before = selector.update(groups, ready).clone()
    members = before[0, 0].nonzero().flatten()
    ready[0, members[0]] = True
    after = selector.update(groups, ready)
    assert not after[0, 0, members[0]]
    assert after[0, 0, members[1:]].all()
    assert after[0, 0].sum() == 3
    assert (after[0, 0] & ~before[0, 0]).sum() == 1


@pytest.mark.parametrize('unready', [0, 1, 2, 3, 5, 7])
def test_keep_and_shrink_cover_every_readiness_count(unready):
    groups, ready = scene(envs=1)
    ready[:] = True
    ready[:, 1:1+unready] = False
    for mode in ('keep3', 'shrink3'):
        selector = CoupledPeerSelection(mode, 'cpu', 44)
        selected = selector.update(groups, ready)[0, 0]
        assert (selected & ~ready[0]).sum() == min(3, unready)
        assert selected.sum() == (3 if mode == 'keep3' else min(3, unready))


def test_keep_retains_the_same_people_as_the_last_unready_member_arrives():
    groups, ready = scene(envs=1)
    selector = CoupledPeerSelection('keep3', 'cpu', 45)
    selected = selector.update(groups, ready).clone()
    ready[:] = True
    assert torch.equal(selector.update(groups, ready), selected)
    lost = (~selected[0, 0]).nonzero().flatten()
    lost = lost[lost != 0][0]
    ready[0, lost] = False
    after = selector.update(groups, ready)
    assert after[0, 0, lost] and after[0, 0].sum() == 3
    assert (after[0, 0] != selected[0, 0]).sum() == 2


def test_shrink_reaches_zero_and_reactivates_after_an_unselected_person_loses_contact():
    groups, ready = scene(envs=1)
    selector = CoupledPeerSelection('shrink3', 'cpu', 46)
    before = selector.update(groups, ready).clone()
    ready[:] = True
    assert not selector.update(groups, ready).any()
    lost = (~before[0, 0]).nonzero().flatten()
    lost = lost[lost != 0][0]
    ready[0, lost] = False
    after = selector.update(groups, ready)
    assert after[0, 0, lost] and after[0, 0].sum() == 1


@pytest.mark.parametrize('mode', ['keep3', 'shrink3'])
def test_groups_of_four_or_fewer_and_other_groups_are_preserved(mode):
    groups, ready = scene()
    groups[1] = torch.tensor([0, 0, 0, 0, 4, 4, 6, 7])
    selector = CoupledPeerSelection(mode, 'cpu', 47)
    for state in (False, True):
        ready[:] = state
        selected = selector.update(groups, ready)
        peers = (groups[:, :, None] == groups[:, None, :]) & ~torch.eye(8, dtype=torch.bool)
        assert not (selected & ~peers).any()
        assert torch.equal(selected[1], peers[1])


@pytest.mark.parametrize('mode', ['keep3', 'shrink3'])
def test_partial_reset_does_not_change_other_scenes(mode):
    groups, ready = scene(envs=3)
    selector = CoupledPeerSelection(mode, 'cpu', 48)
    before = selector.update(groups, ready).clone()
    selector.reset(torch.tensor([1]))
    assert not selector.selected[1].any()
    assert torch.equal(selector.selected[[0, 2]], before[[0, 2]])
    after = selector.update(groups, ready)
    assert torch.equal(after[[0, 2]], before[[0, 2]])
    selector.reset()
    assert selector.selected is None


def test_unselected_coupled_keys_are_masked_while_self_none_and_before_remain():
    torch.manual_seed(49)
    block = CoordinationBlock(8, 2, 16, 4, count_normalization='coupled_none')
    humans = torch.randn(1, 5, 8, requires_grad=True)
    relation = torch.tensor([[[1, 3, 3, 0, 2]]]).expand(1, 5, 5)
    padding = torch.tensor([[[[False, False, True, False, False]]]]).expand(1, 1, 5, 5)
    captured = []

    def audit(module, args, kwargs):
        mask = kwargs['attn_mask'].reshape(1, 2, 5, 5)
        assert torch.isneginf(mask[..., 2]).all()
        assert torch.isfinite(mask[..., [0, 1, 3, 4]]).all()
        captured.append(True)

    handle = block.attention.register_forward_pre_hook(audit, with_kwargs=True)
    block(humans, None, relation, padding)[:, 0].square().mean().backward()
    handle.remove()
    assert captured and not humans.grad[:, 2].any()
    assert humans.grad[:, [0, 1, 3, 4]].abs().sum() > 0


@pytest.mark.parametrize('mode', ['keep3', 'shrink3'])
def test_all_four_layers_use_the_same_mask_after_human_shuffling(mode):
    torch.manual_seed(50)
    groups, ready = scene(envs=1)
    ready[:, 1:] = True
    selector = CoupledPeerSelection(mode, 'cpu', 51)
    selected = selector.update(groups, ready)
    packet = torch.zeros(1, 8, 5)
    packet[..., :2] = torch.tensor([1, 4])
    packet[..., 2] = torch.arange(8)
    humans = torch.randn(1, 8, 8)
    net = HumanCoordination(width=8, heads=2, feedforward_dim=16, relations=4)
    with torch.no_grad():
        for block in net.blocks:
            block.relation_bias.copy_(torch.tensor([[.1, .2, .3, .4], [.5, .6, .7, .8]]))
    net.coupled_peer_mask = selected
    expected = net(humans, None, packet, None, groups=groups)
    order = torch.tensor([7, 2, 4, 0, 3, 1, 6, 5])
    relation = stage2_human_relations(packet, groups)
    relation = torch.where((relation == 3) & ~selected, 0, relation)[:, order][:, :, order]
    seen = []

    def audit(module, args, kwargs):
        actual = kwargs['attn_mask'].reshape(1, 2, 8, 8)
        bias = torch.tensor([[.1, .2, .3, .4], [.5, .6, .7, .8]])
        expected = bias[:, relation].permute(1, 0, 2, 3)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        seen.append(True)

    handles = [b.attention.register_forward_pre_hook(audit, with_kwargs=True) for b in net.blocks]
    net.shuffle_human_order = True
    with patch('torch.randperm', return_value=order):
        actual = net(humans, None, packet, None, groups=groups)
    for handle in handles:
        handle.remove()
    assert len(seen) == 4 and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    assert not any('peer_mask' in key for key in net.state_dict())


@pytest.mark.parametrize('mode', ['keep3', 'shrink3'])
def test_viewer_option_cannot_be_used_for_training_resume_or_formal_eval(mode):
    validate_mode(mode, 'HumanoidMACoupledPush', True)
    for task, evaluation, protocol, resume in [('HumanoidMACoupledPush', False, '', 0),
                                              ('HumanoidMACoupledPush', True, '', 1),
                                              ('HumanoidMACoupledPush', True, 'general', 0),
                                              ('HumanoidMABeforePush', True, '', 0)]:
        with pytest.raises(ValueError, match='viewer/test'):
            validate_mode(mode, task, evaluation, protocol, resume)
