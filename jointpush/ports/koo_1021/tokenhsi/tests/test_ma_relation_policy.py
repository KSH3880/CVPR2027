from pathlib import Path
import pytest
import torch
import torch.nn as nn
import yaml

from learning.multi_agent.amp_network_builder_ma import RelationEncoder, RelationTransformerLayer, AMPMultiAgentBuilder
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from utils.relation_task_spec import STATE_MODE, checkpoint_metadata, check_checkpoint_metadata


def encoder(m=2, o=3):
    return RelationEncoder([223, 30, 1], m, o, 16, 2, 2, 32,
        lambda size: nn.Sequential(nn.Linear(size, 16), nn.ReLU()),
        observation_mode='clean_scene', kinematic_size=7, relation_bias_mode='edge_mlp',
        gta_cfg={'enable': True}, relation_reward_mode=STATE_MODE, diagnostics_interval=2)


def scene(m=2, o=3, batch=4):
    width, L = 224 * m + 30 * o, 2 * m + o
    x = torch.randn(batch, 247 * m + 37 * o)
    x[:, 223 * m + 30 * o:width] = 1
    x[:, width:width + 7 * L].reshape(batch, L, 7)[..., 3:7] = torch.tensor([0., 0., 0., 1.])
    x[:, -9 * m:] = torch.rand(batch, 9 * m)
    return x


@pytest.mark.parametrize('m,o,width', [(1, 1, 284), (2, 2, 568), (2, 3, 605), (3, 4, 889)])
def test_shapes_rms_bypass_and_count_changes(m, o, width):
    e = encoder()
    count = sum(p.numel() for p in e.parameters())
    e.set_entity_counts(m, o)
    assert sum(p.numel() for p in e.parameters()) == count
    x = scene(m, o)
    assert x.shape == (4, width)
    rms = SceneRunningMeanStd([223, 30, 1], [m, o, m], [223, 30, 0], 7, 9 * m)
    out = rms(x)
    untouched = 223 * m + 30 * o
    torch.testing.assert_close(out[:, untouched:], x[:, untouched:], rtol=0, atol=0)
    assert not torch.equal(out[:, :223 * m], x[:, :223 * m])
    assert e(out).shape == (4, m, 16)
    assert e.build_dynamic_relation_bias(x[:, -9 * m:]).shape == (2, 4, 2, 2 * m + o, 2 * m + o)
    assert not any('state_edge' in key or 'rel_matrix' in key for key in e.state_dict())


def test_dynamic_zero_init_gradient_and_batch_isolation():
    torch.manual_seed(5)
    e, x = encoder(), scene()
    assert torch.count_nonzero(e.dynamic_bias_projection) == 0
    opt = torch.optim.Adam(e.parameters(), lr=.01)
    for step in range(2):
        opt.zero_grad()
        loss = (e(x) * torch.randn(4, 2, 16)).sum()
        loss.backward()
        assert e.dynamic_bias_projection.grad.abs().sum() > 0
        if step == 0:
            assert e.dynamic_edge_mlp[0].weight.grad.abs().sum() == 0
        else:
            assert e.dynamic_edge_mlp[0].weight.grad.abs().sum() > 0
        opt.step()
    before = e(x).detach()
    changed = x.clone()
    changed[0, -18:] = 1 - changed[0, -18:]
    after = e(changed).detach()
    torch.testing.assert_close(before[1:], after[1:])
    assert not torch.allclose(before[0], after[0])
    dense = e.build_dynamic_relation_bias(x[:, -18:])
    pairs = torch.zeros(7, 7, dtype=torch.bool)
    pairs[e.state_edge_src, e.state_edge_dst] = True
    assert torch.count_nonzero(dense[..., ~pairs]) == 0
    assert e.last_diagnostics['dynamic_row_rms'] > 0
    assert e.last_diagnostics['layer0/entropy'] > 0


def test_history_permutation_equivariance():
    torch.manual_seed(7)
    e, x = encoder(), scene()
    with torch.no_grad():
        e.dynamic_bias_projection.normal_(std=.1)
        e.edge_encoder.bias_projection.normal_(std=.1)
    perm = torch.tensor([1, 0, 3, 2, 4, 6, 5])
    m, o, B = 2, 3, x.shape[0]
    node_width = 224 * m + 30 * o
    y = torch.cat([x[:, :446].reshape(B, 2, 223)[:, [1, 0]].flatten(1),
                   x[:, 446:536].reshape(B, 3, 30)[:, [1, 0, 2]].flatten(1),
                   x[:, 536:538][:, [1, 0]],
                   x[:, node_width:node_width + 49].reshape(B, 7, 7)[:, perm].flatten(1),
                   x[:, -18:-2].reshape(B, 2, 2, 4)[:, [1, 0]].flatten(1),
                   x[:, -2:][:, [1, 0]]], -1)
    torch.testing.assert_close(e(y), e(x)[:, [1, 0]], atol=2e-6, rtol=2e-6)


def test_attention_static_and_batched_broadcast_agree():
    layer = RelationTransformerLayer(16, 2, 32)
    x, bias = torch.randn(4, 7, 16), torch.randn(2, 7, 7)
    torch.testing.assert_close(layer(x, bias), layer(x, bias.unsqueeze(0).expand(4, -1, -1, -1)))


def build_full_network(m=2, o=3, new=True):
    path = Path(__file__).parents[1] / 'data/cfg/train/rlg/amp_ma_carry_relation.yaml'
    params = yaml.safe_load(path.read_text())['params']['network']
    builder = AMPMultiAgentBuilder()
    builder.load(params)
    return builder.build('amp', actions_num=28, input_shape=((247 if new else 238) * m + 37 * o,),
        amp_input_shape=(64,), value_size=1, num_agents=m, num_objects=o,
        humanoid_obs_size=230, object_obs_size=39, goal_obs_size=6,
        observation_mode='clean_scene', scene_entity_sizes=[223, 30, 1],
        scene_kinematic_size=7, scene_arena_scale=5.,
        relation_reward_mode=STATE_MODE if new else 'legacy_tokenhsi', device='cpu')


def test_full_builder_zero_init_separate_actor_critic_and_strict_load():
    network = build_full_network()
    assert torch.count_nonzero(network.actor_encoder.dynamic_bias_projection) == 0
    assert network.actor_encoder.dynamic_bias_projection is not network.critic_encoder.dynamic_bias_projection
    x = scene(batch=2)
    assert network.eval_actor(x)[0].shape == (4, 28)
    assert network.eval_critic(x).shape == (4, 1)
    changed = build_full_network(3, 4)
    changed.load_state_dict(network.state_dict(), strict=True)
    legacy = build_full_network(new=False)
    assert legacy.actor_encoder.edge_encoder.relation_embed.weight.shape[0] == 6
    with pytest.raises(RuntimeError):
        network.load_state_dict(legacy.state_dict(), strict=True)
    assert sum(p.numel() for p in network.parameters()) == sum(p.numel() for p in changed.parameters())


def test_checkpoint_schema_and_legacy_contract():
    cfg = {'mode': STATE_MODE}
    meta = checkpoint_metadata(cfg)
    check_checkpoint_metadata({'relation_metadata': meta}, meta)
    check_checkpoint_metadata({}, checkpoint_metadata({}))
    with pytest.raises(ValueError, match='Legacy checkpoint'):
        check_checkpoint_metadata({}, meta)
    with pytest.raises(ValueError, match='schema mismatch'):
        check_checkpoint_metadata({'relation_metadata': meta}, checkpoint_metadata({}))
    changed = checkpoint_metadata(dict(cfg, subgoal_success_bonus=7))
    with pytest.raises(ValueError, match='config differs'):
        check_checkpoint_metadata({'relation_metadata': meta}, changed)
    changed = checkpoint_metadata(dict(cfg, diagnostics={'enable': False}))
    check_checkpoint_metadata({'relation_metadata': meta}, changed)
