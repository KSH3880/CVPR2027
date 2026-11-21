from copy import deepcopy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tokenhsi'))
from learning.multi_agent.task_role_encoder import TaskTypeEmbeddingBias, TaskTypeEmbeddingFusion, TaskActionFusion, task_action_condition
from taskbias_ablation import TaskBiasAblation
from bias_ablation import BiasAblation, state_digest


def fixture():
    network = nn.Module()
    for branch in ('actor', 'critic'):
        encoder = nn.Module()
        encoder.edge_encoder = TaskTypeEmbeddingBias(4, 2)
        encoder.edge_encoder.bias_projection.data.normal_()
        encoder.context_fusion = TaskTypeEmbeddingFusion()
        encoder.action_fusion = TaskActionFusion()
        encoder.action_fusion.mlp[-1].weight.data.normal_()
        setattr(network, branch+'_encoder', encoder)
    packet = torch.tensor([[1, 2, 0, 2, 6, 1, 3, 1, 3, 7]]).expand(3, -1).clone()
    types = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2])
    return network, packet, types


def forward(encoder, packet, types):
    background = encoder.edge_encoder(types, None)
    return encoder.context_fusion(packet, encoder.edge_encoder, types, background)[0]


@pytest.mark.parametrize('mode', ['off', 'shuffled'])
def test_real_fusion_and_preservation(mode):
    network, packet, types = fixture()
    legacy = deepcopy(network)
    before = {branch: forward(getattr(network, branch+'_encoder'), packet, types) for branch in ('actor', 'critic')}
    tokens = torch.randn(3, 8, 64)
    encoder = network.actor_encoder
    condition = task_action_condition(packet, encoder.edge_encoder.category_embed)
    features = encoder.action_fusion(tokens, condition)
    weights, original_packet = state_digest(network.state_dict()), packet.clone()
    handler = TaskBiasAblation(network, mode, 100001)
    assert torch.equal(before['actor'], forward(encoder, packet, types))
    handler.enabled = True
    rng = torch.get_rng_state()
    for branch in ('actor', 'critic'):
        after = forward(getattr(network, branch+'_encoder'), packet, types)
        if mode == 'off':
            old = BiasAblation(legacy, 'task')
            old.enabled = True
            assert torch.equal(after, forward(getattr(legacy, branch+'_encoder'), packet, types))
            for module, name, original in old.bindings:
                setattr(module, name, original)
        else:
            assert torch.equal(before[branch].sort(-1).values, after.sort(-1).values)
            assert torch.equal(before[branch].diagonal(dim1=-2, dim2=-1), after.diagonal(dim1=-2, dim2=-1))
            indices = handler.permutation[None, :, None].expand_as(before[branch])
            assert torch.equal(after, before[branch].gather(-1, indices))
            assert not torch.equal(before[branch], after)
        assert torch.equal(after, forward(getattr(network, branch+'_encoder'), packet, types))
    assert torch.equal(rng, torch.get_rng_state())
    handler.finish(2)
    assert torch.equal(before['actor'], forward(encoder, packet, types))
    assert torch.equal(packet, original_packet)
    assert torch.equal(features, encoder.action_fusion(tokens, task_action_condition(packet, encoder.edge_encoder.category_embed)))
    assert state_digest(network.state_dict()) == weights


def test_shuffled_padding_and_internal_token_order():
    network, _, _ = fixture()
    handler = TaskBiasAblation(network, 'shuffled', 100007)
    bias = torch.randn(3, 8, 8, 8)
    bias[:, :, :, 5] = float('-inf')
    natural = torch.arange(8)
    expected = handler.transform(bias, natural, 'canonical')
    order = torch.randperm(8)
    actual = handler.transform(bias[:, :, order][:, :, :, order], order, 'permuted')
    assert torch.equal(actual, expected[:, :, order][:, :, :, order])
    assert torch.equal(torch.isneginf(expected), torch.isneginf(bias))
