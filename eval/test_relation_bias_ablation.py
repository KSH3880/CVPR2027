import torch
from torch.nn import functional as F

from test_depbias_ablation import fixture
from relation_bias_ablation import RelationBiasAblation
from bias_ablation import state_digest
from learning.multi_agent.human_coordination import stage2_human_relations


def test_relation_ids_and_direction():
    _, _, packet, groups = fixture()
    relation = stage2_human_relations(packet, groups)
    assert (relation[:, 3, 2] == 2).all()
    assert (relation[:, 2, 3] == 0).all()
    assert (relation[:, 0, 1] == 3).all() and (relation[:, 1, 0] == 3).all()
    assert (relation.diagonal(dim1=-2, dim2=-1) == 1).all()


def test_selective_removal_real_attention_and_fusion(monkeypatch):
    for mode, kind in (('enabling_off', 2), ('coupled_off', 3)):
        network, humans, packet, groups = fixture()
        digest = state_digest(network.state_dict())
        module = network.coordination
        module.shuffle_human_order = False
        network.critic_coordination.shuffle_human_order = False
        relations = stage2_human_relations(packet, groups)
        attention = F.scaled_dot_product_attention
        captured, active = [], []

        def capture(q, k, v, **kwargs):
            captured.append(kwargs['attn_mask'].clone())
            return attention(q, k, v, **kwargs)

        monkeypatch.setattr(F, 'scaled_dot_product_attention', capture)
        hook = module.relation_fusion.register_forward_pre_hook(
            lambda instance, args: active.append(args[1].clone()))
        original = module(humans, None, packet, None, groups=groups)
        original_masks = captured.copy()
        captured.clear()
        handler = RelationBiasAblation(network, mode)
        assert torch.equal(original, module(humans, None, packet, None, groups=groups))
        captured.clear()
        handler.enabled = True
        rng = torch.get_rng_state()
        changed = module(humans, None, packet, None, groups=groups)
        assert torch.equal(rng, torch.get_rng_state())
        assert not torch.equal(original, changed)
        for before, after in zip(original_masks, captured):
            expected = before.masked_fill((relations == kind)[:, None], 0)
            assert torch.equal(after, expected)
        assert all(torch.equal(a, b) for a, b in zip(active[:4], active[-4:]))
        network.critic_coordination(humans, None, packet, None, groups=groups)
        handler.finish(1)
        assert torch.equal(original, module(humans, None, packet, None, groups=groups))
        assert state_digest(network.state_dict()) == digest
        assert torch.equal(relations, stage2_human_relations(packet, groups))
        hook.remove()
        monkeypatch.setattr(F, 'scaled_dot_product_attention', attention)


def test_padding_preserved(monkeypatch):
    for mode in ('enabling_off', 'coupled_off'):
        network, humans, packet, groups = fixture()
        coordination = network.coordination
        relations = stage2_human_relations(packet, groups)
        padding = torch.zeros(3, 1, 4, 4, dtype=torch.bool)
        padding[:, :, 0, 1] = True
        padding[:, :, 3, 2] = True
        masks = []
        attention = F.scaled_dot_product_attention

        def capture(q, k, v, **kwargs):
            masks.append(kwargs['attn_mask'].clone())
            return attention(q, k, v, **kwargs)

        monkeypatch.setattr(F, 'scaled_dot_product_attention', capture)
        handler = RelationBiasAblation(network, mode)
        handler.enabled = True
        for module in (network.coordination, network.critic_coordination):
            for block in module.blocks:
                block(humans, None, relations, padding, module.relation_fusion)
        handler.finish(1)
        assert all(torch.equal(torch.isneginf(mask), padding.expand_as(mask)) for mask in masks)
        monkeypatch.setattr(F, 'scaled_dot_product_attention', attention)
