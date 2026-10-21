"""Mask physical subscenes through token permutations and every actor layer."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/multi_agent/native4_zeroshot/attention_probe.py'
spec = importlib.util.spec_from_file_location('attention_probe', path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_masks_follow_physical_bindings_and_token_permutations_in_all_layers():
    task = SimpleNamespace(
        _push_binding=torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]]),
        _agent_lanes=torch.tensor([[3, 1, 2, 0], [2, 0, 3, 1]]))

    class Layer:
        def forward(self, x, rel_bias=None, *args, **kwargs):
            self.received = rel_bias
            return x

    class Encoder:
        entity_counts = [4, 8, 4]
        relation_graph_spec = {'shuffle_token_order': True}

        def __init__(self):
            self.layers = [Layer() for _ in range(4)]
            self.context_fusion = SimpleNamespace(forward=lambda *args: None)

        def forward(self, obs, return_all=False, token_order=None):
            x = torch.zeros(len(obs), 16, 64)
            for layer in self.layers:
                x = layer.forward(x, torch.zeros(len(obs), 2, 16, 16))
            return x

    encoder = Encoder()
    control = {'scope': 'pair'}
    probe.install_attention_scope(encoder, task, control)
    order = torch.tensor([9, 4, 15, 0, 10, 6, 2, 13, 7, 5, 3, 1, 11, 14, 8, 12])
    encoder.forward(torch.zeros(2, 1), token_order=order)
    groups = probe.scene_groups(task).index_select(1, order)
    expected = groups[:, :, None] == groups[:, None, :]
    assert expected.sum(-1).eq(8).all()
    for layer in encoder.layers:
        assert torch.equal(torch.isfinite(layer.received), expected[:, None].expand(-1, 2, -1, -1))
    control['scope'] = 'full'
    encoder.forward(torch.zeros(2, 1), token_order=order)
    assert all(torch.isfinite(layer.received).all() for layer in encoder.layers)


def test_none_zero_preserves_task_edges_self_and_invalid_edge_background():
    # Source edge schema: valid, src, dst, relation, owner.
    packet = torch.tensor([[1, 0, 2, 12, 0, 1, 2, 3, 7, 0, 0, 1, 0, 12, 0]], dtype=torch.float)
    bias = torch.arange(2 * 1 * 2 * 4 * 4, dtype=torch.float).reshape(2, 1, 2, 4, 4) + 1
    result = probe.zero_none_bias(bias, packet)
    keep = torch.eye(4, dtype=torch.bool)
    keep[0, 2] = keep[2, 3] = True
    assert torch.equal(result[..., keep], bias[..., keep])
    assert result[..., ~keep].eq(0).all()


def test_selected_key_type_masks_keep_other_keys_and_same_scene_after_shuffle():
    task = SimpleNamespace(
        _push_binding=torch.tensor([[2, 0, 3, 1]]),
        _agent_lanes=torch.tensor([[3, 1, 2, 0]]))
    order = torch.tensor([9, 4, 15, 0, 10, 6, 2, 13, 7, 5, 3, 1, 11, 14, 8, 12])
    groups = probe.scene_groups(task)[0, order]
    for scope, selected in [('hide_humans', range(4)), ('hide_doors', range(4, 8)),
                            ('hide_boxes', range(8, 12)),
                            ('hide_goals', range(12, 16)),
                            ('hide_boxes_goals', range(8, 16)),
                            ('hide_objects_goals', range(4, 16))]:
        actual = probe.attention_allowed(task, order, scope)[0]
        for i in range(16):
            for j in range(16):
                blocked = groups[i] != groups[j] and int(order[j]) in selected
                assert bool(actual[i, j]) == (not blocked)


def test_one_foreign_box_hidden_for_every_query_without_hiding_own_scene():
    task = SimpleNamespace(
        _push_binding=torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]]),
        _agent_lanes=torch.tensor([[3, 1, 2, 0], [2, 0, 3, 1]]))
    order = torch.tensor([9, 4, 15, 0, 10, 6, 2, 13, 7, 5, 3, 1, 11, 14, 8, 12])
    groups = probe.scene_groups(task).index_select(1, order)
    for rank, scope in enumerate(('hide_box_first', 'hide_box_second')):
        actual = probe.attention_allowed(task, order, scope)
        assert actual.sum(-1).eq(15).all()
        for env in range(2):
            for i in range(16):
                for j in range(16):
                    token = int(order[j])
                    selected = 8 <= token < 12 and int(task._push_binding[env, token - 8]) % 2 == rank
                    blocked = selected and groups[env, i] != groups[env, j]
                    assert bool(actual[env, i, j]) == (not blocked)


@pytest.mark.parametrize('humans', [2, 4])
def test_remove_doors_reduces_every_actor_layer_with_aligned_gta(humans):
    task = SimpleNamespace(_push_binding=torch.tensor([[2, 0, 3, 1]]),
                           _agent_lanes=torch.tensor([[3, 1, 2, 0]]))

    class Layer:
        def forward(self, x, rel_bias, geo_score, geo_message, gta_g, gta_ginv):
            assert x.shape == (1, 3 * humans, 64)
            assert rel_bias.shape == (1, 2, 3 * humans, 3 * humans)
            assert gta_g.shape == gta_ginv.shape == (1, 3 * humans, 4, 4)
            assert torch.equal(x[0, :, 0].long() % 100, gta_g[0, :, 0, 0].long())
            return x + 100

    class Encoder:
        entity_counts = [humans, 2 * humans, humans]
        relation_graph_spec = {'shuffle_token_order': True}

        def __init__(self):
            self.layers = [Layer() for _ in range(4)]
            self.context_fusion = SimpleNamespace(forward=lambda *args: None)

        def forward(self, obs, return_all=False, token_order=None):
            x = obs.index_select(1, token_order)
            gta = token_order.float()[None, :, None, None].expand(1, 4 * humans, 4, 4)
            for layer in self.layers:
                x = layer.forward(x, torch.zeros(1, 2, 4 * humans, 4 * humans), None, None, gta, gta)
            return x.index_select(1, token_order.argsort())

    encoder = Encoder()
    probe.install_attention_scope(encoder, task, {'scope': 'remove_doors'})
    obs = torch.arange(4 * humans).float()[None, :, None].expand(1, 4 * humans, 64)
    order = ([9, 4, 15, 0, 10, 6, 2, 13, 7, 5, 3, 1, 11, 14, 8, 12] if humans == 4 else
             [5, 2, 7, 0, 4, 3, 1, 6])
    out = encoder.forward(obs, token_order=torch.tensor(order))
    assert torch.equal(out[:, humans:2 * humans], obs[:, humans:2 * humans])
    keep = torch.tensor(list(range(humans)) + list(range(2 * humans, 4 * humans)))
    assert torch.equal(out[:, keep], obs[:, keep] + 400)


@pytest.mark.parametrize('ordinary', [0, 1, 2, 3])
def test_partition_probe_keeps_all_task_regions_and_goals_when_dropping_ordinary(ordinary):
    class Layer:
        def forward(self, x, rel_bias, *args, **kwargs):
            assert x.shape[1] == 12 + ordinary
            ids = x[0, :, 0].long().remainder(100)
            assert not ((ids >= 8 + ordinary) & (ids < 12)).any()
            assert rel_bias.shape[-2:] == (12 + ordinary, 12 + ordinary)
            return x + 100

    class Encoder:
        entity_counts = [4, 8, 4]
        relation_graph_spec = {'shuffle_token_order': True}

        def __init__(self):
            self.layers = [Layer() for _ in range(4)]
            self.context_fusion = SimpleNamespace(forward=lambda *args: None)

        def forward(self, obs, return_all=False, token_order=None):
            x = obs.index_select(1, token_order)
            for layer in self.layers:
                x = layer.forward(x, torch.zeros(1, 2, 16, 16), None, None, None, None)
            return x.index_select(1, token_order.argsort())

    encoder = Encoder()
    probe.install_attention_scope(encoder, SimpleNamespace(), {'scope': f'keep_ordinary_{ordinary}'})
    obs = torch.arange(16).float()[None, :, None].expand(1, 16, 64)
    out = encoder.forward(obs, token_order=torch.tensor([9, 4, 15, 0, 10, 6, 2, 13, 7, 5, 3, 1, 11, 14, 8, 12]))
    assert torch.equal(out[:, 4:8], obs[:, 4:8] + 400)
    assert torch.equal(out[:, 12:16], obs[:, 12:16] + 400)
    assert torch.equal(out[:, 8 + ordinary:12], obs[:, 8 + ordinary:12])
