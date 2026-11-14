import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tokenhsi'))
from bias_ablation import BiasAblation, state_digest
from learning.multi_agent.task_role_encoder import TaskTypeEmbeddingBias, TaskActionFusion
from learning.multi_agent.stage2_relation_fusion import Stage2RelationFusion, Stage2FusionCoordinationBlock


def test_task_bias_only_and_restore():
    network = nn.Module()
    for branch in ('actor', 'critic'):
        encoder = nn.Module()
        encoder.edge_encoder = TaskTypeEmbeddingBias(4, 2)
        encoder.edge_encoder.bias_projection.data.fill_(.3)
        encoder.task_action_fusion = TaskActionFusion()
        encoder.task_action_fusion.mlp[-1].weight.data.fill_(.2)
        setattr(network, branch+'_encoder', encoder)
    encoder = network.actor_encoder
    tokens = torch.randn(1, 4, 64)
    condition = (torch.tensor([[0, 1]]), encoder.edge_encoder.category_embed(torch.tensor([[0, 1]])),
                 torch.ones(1, 2, dtype=torch.bool))
    features = encoder.task_action_fusion(tokens, condition)
    bias = encoder.edge_encoder.bias_table().clone()
    weights = state_digest(network.state_dict())
    ablation = BiasAblation(network, 'task')
    assert torch.equal(bias, encoder.edge_encoder.bias_table())
    ablation.enabled = True
    assert not encoder.edge_encoder.bias_table().any()
    assert not network.critic_encoder.edge_encoder.bias_table().any()
    assert torch.equal(features, encoder.task_action_fusion(tokens, condition))
    ablation.finish(1)
    assert state_digest(network.state_dict()) == weights
    assert torch.equal(bias, encoder.edge_encoder.bias_table())


def test_dependency_bias_only_and_restore():
    network = nn.Module()
    for name in ('coordination', 'critic_coordination'):
        coordination = nn.Module()
        coordination.relation_fusion = Stage2RelationFusion(64)
        coordination.relation_fusion.mlp[-1].weight.data.fill_(.2)
        coordination.blocks = nn.ModuleList([Stage2FusionCoordinationBlock(64, 2, 128) for _ in range(4)])
        for block in coordination.blocks:
            block.bias_projection.data.fill_(.3)
        setattr(network, name, coordination)
    coordination = network.coordination
    context = torch.randn(1, 2, 64)
    active = torch.tensor([[[True, False], [False, True]]])
    features = coordination.relation_fusion(context, active)
    bias = coordination.blocks[0].effective_relation_bias(coordination.relation_fusion.embedding).clone()
    weights = state_digest(network.state_dict())
    ablation = BiasAblation(network, 'dependency')
    assert torch.equal(bias, coordination.blocks[0].effective_relation_bias(coordination.relation_fusion.embedding))
    ablation.enabled = True
    for name in ('coordination', 'critic_coordination'):
        branch = getattr(network, name)
        for block in branch.blocks:
            assert not block.effective_relation_bias(branch.relation_fusion.embedding).any()
    assert torch.equal(features, coordination.relation_fusion(context, active))
    ablation.finish(1)
    assert state_digest(network.state_dict()) == weights
    assert torch.equal(bias, coordination.blocks[0].effective_relation_bias(coordination.relation_fusion.embedding))
