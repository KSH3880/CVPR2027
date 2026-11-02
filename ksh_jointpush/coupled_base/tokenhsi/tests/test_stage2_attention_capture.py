"""Canonical head maps must match real shuffled coordination and PyTorch MHA."""
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ksh_jointpush.ksh_torch_compat import install_attention_compat
install_attention_compat()
from ksh_jointpush.ksh_stage2_attention import canonical_coordination_maps
from learning.multi_agent.human_coordination import HumanCoordination


def test_canonical_maps_preserve_shuffled_outputs_and_agent_labels():
    torch.manual_seed(13)
    branch = HumanCoordination(shuffle_human_order=True, relations=4).eval()
    for layer in branch.blocks:
        layer.relation_bias.data.copy_(torch.tensor([[0., .2, .3, .6], [-.1, .4, .2, .7]]))
    humans = torch.randn(3, 4, 64)
    groups = torch.tensor([[0, 1, 1, 1], [0, 0, 2, 2], [0, 0, 0, 0]])
    packet = torch.tensor([[1., 4., i, i, 12+i] for i in range(4)]).flatten().expand(3, -1)
    with torch.no_grad():
        original = branch(humans, None, packet, None, groups=groups)
    before = {key: value.clone() for key, value in branch.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = canonical_coordination_maps(branch, humans, packet, groups)
    torch.testing.assert_close(result['output'], original, atol=2e-6, rtol=2e-6)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, branch.state_dict()[key]) for key, value in before.items())
    assert branch.shuffle_human_order
    assert result['attention'].shape == (3, 4, 2, 4, 4)
    torch.testing.assert_close(result['attention'].sum(-1), torch.ones(3, 4, 2, 4))
    expected = torch.tensor([[.2, 0., 0., 0.], [0., .2, .6, .6],
                             [0., .6, .2, .6], [0., .6, .6, .2]])
    torch.testing.assert_close(result['relation_bias'][0, 0, 0], expected)
    torch.testing.assert_close(result['raw_qk_scores'] / (32 ** .5), result['qk_scores'], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close((result['raw_qk_scores'] / (32 ** .5) + result['relation_bias']).softmax(-1), result['attention'], atol=2e-6, rtol=2e-6)
    for index, layout in enumerate(groups):
        expected = torch.where(torch.eye(4, dtype=torch.bool), .2,
                               torch.where(layout[:, None] == layout[None, :], .6, 0.))
        torch.testing.assert_close(result['relation_bias'][index, 0, 0], expected)
    assert result['mha_reference_max_error'] < 2e-6


def test_zero_projection_identity_still_has_nontrivial_attention():
    torch.manual_seed(14)
    branch = HumanCoordination(shuffle_human_order=True, relations=4).eval()
    branch.init_identity()
    humans = torch.randn(1, 4, 64)
    groups = torch.tensor([[0, 1, 1, 1]])
    packet = torch.tensor([[1., 4., i, i, 12+i] for i in range(4)]).flatten()[None]
    result = canonical_coordination_maps(branch, humans, packet, groups)
    assert torch.equal(result['output'], humans)
    assert (result['attention']-.25).abs().max() > .001
