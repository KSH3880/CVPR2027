import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tokenhsi'))
from learning.multi_agent.human_coordination import HumanCoordination
from depbias_ablation import DependencyBiasAblation
from bias_ablation import BiasAblation, state_digest


def fixture():
    torch.manual_seed(12)
    network = nn.Module()
    for name in ('coordination', 'critic_coordination'):
        module = HumanCoordination(shuffle_human_order=True, relations=4, relation_fusion='stage2_zerobg')
        for block in module.blocks:
            block.bias_projection.data.normal_()
        module.relation_fusion.mlp[-1].weight.data.normal_(std=.1)
        setattr(network, name, module)
    humans = torch.randn(3, 4, 64)
    packet = torch.tensor([[[1, 4, 0, 0, 0], [1, 4, 1, 0, 0],
                            [1, 2, 2, 2, 3], [1, 0, 3, 3, 2]]]).expand(3, -1, -1)
    groups = torch.tensor([[0, 0, 2, 3]]).expand(3, -1)
    return network, humans, packet, groups


def test_real_blocks_restore_rng_and_fusion():
    for mode in ('off', 'shuffled'):
        network, humans, packet, groups = fixture()
        weights = state_digest(network.state_dict())
        rng = torch.get_rng_state()
        expected = network.coordination(humans, None, packet, None, groups=groups)
        expected_rng = torch.get_rng_state()
        handler = DependencyBiasAblation(network, mode, 100001)
        torch.set_rng_state(rng)
        actual = network.coordination(humans, None, packet, None, groups=groups)
        assert torch.equal(expected, actual)
        assert torch.equal(expected_rng, torch.get_rng_state())
        active = []
        hook = network.coordination.relation_fusion.register_forward_pre_hook(
            lambda module, args: active.append(args[1].clone()))
        torch.set_rng_state(rng)
        network.coordination(humans, None, packet, None, groups=groups)
        handler.enabled = True
        torch.set_rng_state(rng)
        changed = network.coordination(humans, None, packet, None, groups=groups)
        assert torch.equal(expected_rng, torch.get_rng_state())
        assert not torch.allclose(expected, changed)
        assert all(torch.equal(a, b) for a, b in zip(active[:4], active[4:]))
        hook.remove()
        network.critic_coordination(humans, None, packet, None, groups=groups)
        handler.finish(1)
        torch.set_rng_state(rng)
        assert torch.equal(expected, network.coordination(humans, None, packet, None, groups=groups))
        assert state_digest(network.state_dict()) == weights


def test_fixed_person_mapping_despite_internal_shuffle():
    network, humans, packet, groups = fixture()
    handler = DependencyBiasAblation(network, 'shuffled', 10)
    handler.enabled = True
    outputs = [network.coordination(humans, None, packet, None, groups=groups) for _ in range(4)]
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], rtol=1e-5, atol=1e-5)
    permutation = handler.permutation.clone()
    for _ in range(4):
        network.critic_coordination(humans, None, packet, None, groups=groups)
    assert torch.equal(permutation, handler.permutation)
    handler.finish(4)


def test_padding_diagonal_histogram_and_independent_rng():
    network, *_ = fixture()
    handler = DependencyBiasAblation(network, 'shuffled', 22)
    bias = torch.arange(32).float().reshape(1, 2, 4, 4)
    bias[:, :, :, 3] = float('-inf')
    order = torch.tensor([2, 0, 3, 1])
    rng = torch.get_rng_state()
    result = handler.transform(bias[:, :, order][:, :, :, order], order, 'first')
    assert torch.equal(rng, torch.get_rng_state())
    restored = result[:, :, order.argsort()][:, :, :, order.argsort()]
    assert torch.equal(restored, bias.gather(-1, handler.permutation[:, None].expand_as(bias)))
    assert torch.equal(bias.sort(-1).values, restored.sort(-1).values)
    assert torch.equal(bias.diagonal(dim1=-2, dim2=-1), restored.diagonal(dim1=-2, dim2=-1))


def test_off_exactly_matches_previous_bias_removal():
    network, humans, packet, groups = fixture()
    previous = BiasAblation(network, 'dependency')
    previous.enabled = True
    rng = torch.get_rng_state()
    expected = [module(humans, None, packet, None, groups=groups)
                for module in (network.coordination, network.critic_coordination)]
    previous.finish(1)
    current = DependencyBiasAblation(network, 'off', 100001)
    current.enabled = True
    torch.set_rng_state(rng)
    actual = [module(humans, None, packet, None, groups=groups)
              for module in (network.coordination, network.critic_coordination)]
    current.finish(1)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
