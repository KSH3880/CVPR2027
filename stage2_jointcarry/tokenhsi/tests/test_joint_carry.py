from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from learning.multi_agent.task_coordination import TaskCoordination, task_relations
from learning.multi_agent.task_role_encoder import TaskTypeEmbeddingBias
from utils.joint_carry_spec import sample_graph, JOINT_SAMPLER
from utils.task_role_spec import task_graph_packet

ROOT = Path(__file__).resolve().parents[1]


SPEC = dict(mode='edge_composition', sampler=JOINT_SAMPLER, semantic_only=True,
    policy_task_roles=True, max_edges_per_agent=2, edge_capacity=4,
    family_probabilities={'joint_carry_at': .5, 'joint_carry_ontop': .5},
    shuffle_edge_order=True, shuffle_token_order=True,
    coupling='same_carry_task_payload_target')


def test_joint_sampler_binding_and_scene_distribution():
    graph = sample_graph(2048, SPEC, generator=torch.Generator().manual_seed(7))
    records = task_graph_packet(graph, 2048).reshape(2048, 2, 5).long()
    assert records[:, :, 0].all()
    assert (records[:, 0, 1] == records[:, 1, 1]).all()
    assert (records[:, :, 2] == torch.tensor([0, 1])).all()
    assert (records[:, :, 3] == 2).all()
    assert (records[:, 0, 4] == records[:, 1, 4]).all()
    assert abs((records[:, 0, 1] == 2).float().mean() - .5) < .04
    assert not graph.prereq_mask.any()
    for preset, category, target in [('joint_carry_at', 2, 6), ('joint_carry_ontop', 3, 4)]:
        packet = task_graph_packet(sample_graph(32, SPEC, preset=preset), 32).reshape(32, 2, 5)
        assert (packet[..., 1] == category).all()
        assert (packet[..., 4] == target).all()


def test_ca_relation_bias_manual_oracle_padding_and_permutation():
    torch.manual_seed(3)
    module = TaskCoordination()
    embed = TaskTypeEmbeddingBias(4, 2).requires_grad_(False)
    packet = torch.tensor([[1, 2, 0, 3, 7, 1, 2, 1, 3, 7, 1, 1, 2, 0, 4, 0, 0, 0, 0, 0]]).float()
    nodes = torch.randn(1, 9, 64)
    humans = nodes[:, :3]
    expected = torch.tensor([[[1, 2, 0, 0], [2, 1, 0, 0], [0, 0, 1, 0]]])
    torch.testing.assert_close(task_relations(packet, 3), expected)
    with torch.no_grad():
        module.relation_bias.copy_(torch.tensor([[.1, .3, -.2], [-.1, .2, .5]]))
    tokens, _ = module.task_tokens(nodes, packet, embed)
    wq, wk, wv = module.attention.in_proj_weight.chunk(3)
    bq, bk, bv = module.attention.in_proj_bias.chunk(3)
    q = torch.nn.functional.linear(humans, wq, bq).view(1, 3, 2, 32).transpose(1, 2)
    k = torch.nn.functional.linear(tokens, wk, bk).view(1, 4, 2, 32).transpose(1, 2)
    v = torch.nn.functional.linear(tokens, wv, bv).view(1, 4, 2, 32).transpose(1, 2)
    logits = q @ k.transpose(-2, -1) / (32 ** .5)
    logits += module.relation_bias[:, expected].permute(1, 0, 2, 3)
    logits[..., -1] = float('-inf')
    oracle = (logits.softmax(-1) @ v).transpose(1, 2).reshape(1, 3, 64)
    oracle = module.attention.out_proj(oracle)
    output = module(humans, nodes, packet, embed)
    torch.testing.assert_close(output, oracle)
    shuffled = packet.reshape(1, 4, 5)[:, [2, 0, 3, 1]].flatten(1)
    torch.testing.assert_close(output, module(humans, nodes, shuffled, embed))
    assert module(humans, nodes, torch.zeros_like(packet), embed).count_nonzero() == 0
    output.square().mean().backward()
    assert (module.relation_bias.grad.abs().sum(0) > 0).all()
    assert all(p.grad is None for p in embed.parameters())


def test_sit_climb_payload_is_zero_and_role_slots_remain_distinct():
    module = TaskCoordination()
    embed = TaskTypeEmbeddingBias(4, 2)
    nodes = torch.randn(1, 8, 64)
    packet = torch.tensor([[1, 1, 0, 0, 4, 1, 3, 1, 2, 4]]).float()
    inputs = []
    handle = module.grounding[0].register_forward_pre_hook(lambda _, args: inputs.append(args[0]))
    module.task_tokens(nodes, packet, embed)
    handle.remove()
    packed = inputs[0]
    assert packed[0, 0, 128:192].count_nonzero() == 0
    torch.testing.assert_close(packed[0, 1, 128:192], nodes[0, 2])
    torch.testing.assert_close(packed[0, 1, 192:], nodes[0, 4])


def build_network(stage2=True):
    from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
    torch.set_num_threads(1)
    train = yaml.safe_load((ROOT / 'data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_task_embedding.yaml').read_text())
    env = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding.yaml').read_text())['env']
    if stage2:
        train['params']['network']['coordination'] = dict(enabled=True, mode='task_coupled', grounding_hidden=128, num_heads=2)
    builder = AMPMultiAgentBuilder()
    builder.load(train['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(634,), amp_input_shape=(1320,),
        value_size=1, num_agents=2, num_objects=4, humanoid_obs_size=230,
        object_obs_size=39, goal_obs_size=6, observation_mode='clean_scene',
        scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7, scene_arena_scale=5.,
        relation_reward_mode='state_relation_edge_stage2_v1' if stage2 else env['relationReward']['mode'],
        relation_graph_spec=SPEC if stage2 else env['relationGraph'], device='cpu')
    for enc in (net.actor_encoder, net.critic_encoder):
        enc.gta_diagnostics_first_forward = False
    return net


def joint_observations(n=8):
    from test_task_role_message import observations
    return observations(sample_graph(n, SPEC))


def test_transfer_initial_action_freeze_updates_and_joint_dedup():
    from learning.multi_agent.stage2_transfer import transfer_stage1_weights
    from utils.relation_task_spec import checkpoint_metadata
    from test_stage2_coordination import Wrapper
    source, target = Wrapper(build_network(False)), Wrapper(build_network())
    reward = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding.yaml').read_text())['env']['relationReward']
    checkpoint = dict(model=source.state_dict(), relation_metadata=checkpoint_metadata(reward))
    transfer_stage1_weights(target, checkpoint)
    net = target.a2c_network
    obs = joint_observations()
    with torch.no_grad():
        humans = net.actor_encoder(obs)
        expected = source.a2c_network.action_head(humans).flatten(0, 1)
        actual, _ = net.eval_actor(obs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    frozen = {k: v.clone() for k, v in net.actor_encoder.state_dict().items()}
    initial = {k: v.clone() for k, v in net.coordination.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad()
        mu, _ = net.eval_actor(obs)
        (mu.square().mean() + net.eval_critic(obs).square().mean()).backward()
        optimizer.step()
    assert all(p.grad is None and not p.requires_grad for p in net.actor_encoder.parameters())
    for k, v in frozen.items():
        torch.testing.assert_close(net.actor_encoder.state_dict()[k], v, rtol=0, atol=0)
    for k in ('grounding.0.weight', 'attention.in_proj_weight', 'relation_bias'):
        assert not torch.equal(initial[k], net.coordination.state_dict()[k])
    assert net.coordination.relation_bias[:, 0].count_nonzero() == 0
    enc = net.actor_encoder
    with torch.no_grad():
        enc.edge_encoder.bias_projection.normal_()
    graph = sample_graph(4, SPEC, preset='joint_carry_at')
    packet = task_graph_packet(graph, 4)
    bias, _ = enc.context_fusion(packet, enc.edge_encoder, enc.entity_types, enc.build_relation_bias())
    expected_bias = enc.edge_encoder.bias_table()[2, 1, 2]
    torch.testing.assert_close(bias[:, :, :, 2, 6], expected_bias[:, None].expand(4, 4, 2))


def test_transfer_rejects_wrong_variant_and_architecture():
    from learning.multi_agent.stage2_transfer import transfer_stage1_weights
    from utils.relation_task_spec import checkpoint_metadata
    from test_stage2_coordination import Wrapper
    source, target = Wrapper(build_network(False)), Wrapper(build_network())
    reward = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill.yaml').read_text())['env']['relationReward']
    checkpoint = dict(model=source.state_dict(), relation_metadata=checkpoint_metadata(reward))
    assert transfer_stage1_weights(target, checkpoint)['temporary_distill_source']
    carry = deepcopy(checkpoint)
    from utils.task_role_spec import CARRY_DISTILL_VARIANT
    carry['relation_metadata']['relation_reward_config']['stage1_variant'] = CARRY_DISTILL_VARIANT
    assert transfer_stage1_weights(target, carry)['source_variant'] == CARRY_DISTILL_VARIANT
    for key, value in [('packet_version', 3), ('context_fusion', 'semantic_only')]:
        bad = deepcopy(checkpoint)
        bad['relation_metadata'][key] = value
        with pytest.raises(ValueError):
            transfer_stage1_weights(target, bad)
