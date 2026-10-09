from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from env.tasks.multi_agent.joint_carry_reward import (
    JointCarryRuntime, anchor_scores, opposite_gate, cpa_collision_penalty)
from learning.multi_agent.task_coordination import TaskCoordination
from learning.multi_agent.task_role_encoder import TaskTypeEmbeddingBias
from utils.edge_context_spec import HOLDING
from utils.edge_ontop_spec import permute_graph
from utils.joint_carry_spec import sample_graph, validate_env
from utils.joint_carry_rsi import load_paired_snapshots, transform_states
from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata
from utils.task_role_spec import task_graph_packet

ROOT = Path(__file__).resolve().parents[2]
ENV = yaml.safe_load((ROOT / 'tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_task_embedding.yaml').read_text())['env']


@pytest.mark.parametrize('position,velocity,expected', [
    ([2., 0., 0.], [-1., 0., 0.], .99**60),
    ([2., 0., 0.], [1., 0., 0.], 0.),
    ([2., .35, 0.], [-1., 0., 0.], .5 * 2 / (4 + .35**2)**.5 * .99**60),
    ([2., .8, 0.], [-1., 0., 0.], 0.),
    ([.35, 0., 0.], [0., 0., 0.], 0.),
    ([2., 0., 0.], [0., 0., 0.], 0.),
    ([0., 0., 0.], [0., 0., 0.], 0.),
])
def test_cpa_approach_departure_stationary_and_passing(position, velocity, expected):
    roots = torch.tensor([[[0., 0., 0.], position]])
    speeds = torch.tensor([[[0., 0., 0.], velocity]])
    penalty = cpa_collision_penalty(roots, speeds, .7, .99, 1/30)
    torch.testing.assert_close(penalty, torch.full((1, 2), expected))
    torch.testing.assert_close(cpa_collision_penalty(roots + 10., speeds + 3., .7, .99, 1/30), penalty)
    torch.testing.assert_close(cpa_collision_penalty(roots.flip(1), speeds.flip(1), .7, .99, 1/30), penalty.flip(1))
    assert torch.isfinite(penalty).all()


def test_cpa_urgency_uses_control_dt_and_ignores_self():
    roots = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
    speeds = torch.tensor([[[0., 0., 0.], [-1., 0., 0.]]])
    for dt, expected in [(1/30, .99**30), (1/60, .99**60)]:
        torch.testing.assert_close(cpa_collision_penalty(roots, speeds, .7, .99, dt),
                                  torch.full((1, 2), expected))
    assert not cpa_collision_penalty(roots[:, :1], speeds[:, :1], .7, .99, 1/30).any()


def test_config_and_checkpoint_separation():
    validate_env(ENV)
    validate_relation_config(ENV['relationReward'])
    bad = deepcopy(ENV)
    bad['box']['build']['randomSize'] = True
    with pytest.raises(ValueError):
        validate_env(bad)
    bad = deepcopy(ENV)
    bad['templateRsi']['HOLDING_ON_TOP'] = bad['templateRsi']['HOLDING_AT']
    with pytest.raises(ValueError):
        validate_env(bad)
    stage1 = yaml.safe_load((ROOT / 'tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding.yaml').read_text())['env']['relationReward']
    with pytest.raises(ValueError):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(stage1)},
                                  checkpoint_metadata(ENV['relationReward']))
    old = deepcopy(ENV['relationReward'])
    old['joint_carry'].pop('collision')
    with pytest.raises(ValueError):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(old)},
                                  checkpoint_metadata(ENV['relationReward']))


def test_opposite_anchor_assignment_rotation_and_snapshot_geometry():
    box = torch.zeros(4, 13); box[:, 6] = 1.
    size = torch.tensor([[.52, .8, .4]]).expand(4, -1)
    hands = torch.zeros(4, 2, 2, 3)
    hands[:, 0, :, 1] = -.25
    hands[:, 1, :, 1] = .25
    hands[1, 1, :, 1] = -.25
    hands[2, 0, :, 1] = .25; hands[2, 1, :, 1] = -.25
    hands[3, 1, :, 1] = .5
    scores, anchors = anchor_scores(hands, box, size, .15)
    assert opposite_gate(scores, .9).tolist() == [True, False, True, False]
    torch.testing.assert_close(anchors[0, :, 1], torch.tensor([-.25, .25]))
    angle = torch.tensor([torch.pi/2]*4)
    shift = torch.tensor([[3., 4., 1.]]).expand(4, -1)
    hstate = torch.zeros(4, 2, 2, 13); hstate[..., :3] = hands; hstate[..., 6] = 1.
    rotated, _ = anchor_scores(transform_states(hstate, angle, shift)[..., :3],
        transform_states(box, angle, shift), size, .15)
    torch.testing.assert_close(scores, rotated)
    pools = load_paired_snapshots(ROOT / 'joint_carry/carry_joint_corrected', 'cpu')
    assert [len(pools[k]['root_state']) for k in ('pickUp', 'carryWith', 'putDown')] == [55, 129, 135]
    pool = pools['carryWith']
    with np.load(ROOT / 'joint_carry/carry_joint_corrected/carryWith/cooperative_rsi_candidate.npz') as data:
        names = data['joint_names'].tolist()
    hand_ids = [names.index('right_hand'), names.index('left_hand')]
    actual, _ = anchor_scores(pool['body_state'][:, :, hand_ids, :3], pool['box_state'][:, 0],
        size[:1].expand(129, -1), .15)
    assert opposite_gate(actual, .9).all()
    root = pool['root_state'][:4]
    transformed = transform_states(root, angle, shift)
    torch.testing.assert_close(transformed[..., 7], -root[..., 8])
    torch.testing.assert_close(transformed[..., 8], root[..., 7])
    torch.testing.assert_close(transformed[..., 10], -root[..., 11])


@pytest.mark.parametrize('preset', ['joint_carry_at', 'joint_carry_ontop'])
def test_reward_gate_no_latch_no_saturation_and_edge_shuffle(preset):
    graph = sample_graph(2, ENV['relationGraph'], preset=preset)
    config = ENV['relationReward']
    phi = torch.full((2, 4), .97)
    progress = torch.full_like(phi, .7)
    zero = torch.zeros_like(phi)
    runtime = JointCarryRuntime(2, graph, config, 'cpu')
    runtime.coupled_holding[:] = torch.tensor([True, False])
    result = runtime.step(phi, progress, zero, zero, zero)
    holding = graph.edge_relation == HOLDING
    assert result['total'][1, ~holding[1]].count_nonzero() == 0
    torch.testing.assert_close(result['agent_task_reward'][0], torch.full((2,), 2*(.2*.97+.2*.7+.2)))
    torch.testing.assert_close(result['agent_task_reward'][1], torch.full((2,), .2*.97+.2*.7+.2))
    assert not result['reward_saturated'].any()
    assert runtime.done[0].all() and not runtime.done[1].any()
    runtime.coupled_holding.zero_()
    phi[holding] = .1
    dropped = runtime.step(phi, progress, zero, zero, zero)
    assert not runtime.done.any()
    assert not dropped['own_success'].any()
    assert dropped['total'][~holding].count_nonzero() == 0
    order = torch.tensor([[2, 0, 3, 1], [3, 2, 1, 0]])
    other = JointCarryRuntime(2, permute_graph(graph, order), config, 'cpu')
    permuted = other.step(phi.gather(1, order), progress.gather(1, order), zero, zero, zero)
    torch.testing.assert_close(dropped['agent_task_reward'], permuted['agent_task_reward'])


def test_ca_random_shuffle_preserves_output_and_gradient():
    graph = sample_graph(8, ENV['relationGraph'])
    packet = task_graph_packet(graph, 8)
    module = TaskCoordination(shuffle_task_order=False)
    embed = TaskTypeEmbeddingBias(4, 2).requires_grad_(False)
    nodes = torch.randn(8, 8, 64)
    with torch.no_grad():
        module.relation_bias.copy_(torch.tensor([[0., .7, -.3], [0., -.2, .4]]))
    output = module(nodes[:, :2], nodes, packet, embed)
    output.square().sum().backward()
    gradients = [p.grad.clone() for p in module.parameters()]
    module.zero_grad(set_to_none=True)
    module.shuffle_task_order = True
    shuffled = module(nodes[:, :2], nodes, packet, embed)
    shuffled.square().sum().backward()
    torch.testing.assert_close(output, shuffled)
    for old, parameter in zip(gradients, module.parameters()):
        torch.testing.assert_close(old, parameter.grad, atol=2e-5, rtol=1e-4)


def test_full_actor_sa_and_ca_permutation_and_reload():
    from test_joint_carry import build_network, joint_observations
    net = build_network()
    obs = joint_observations()
    def action(observation, order):
        humans, nodes = net.actor_encoder(observation, return_all=True, token_order=order)
        context = net.coordination(humans, nodes, observation[:, -10:], net.actor_encoder.edge_encoder)
        return net.action_head(torch.cat((humans, context), -1))
    identity = torch.arange(8)
    shuffled_obs = obs.clone()
    shuffled_obs[:, -10:] = obs[:, -10:].reshape(-1, 2, 5).flip(1).flatten(1)
    original = action(obs, identity)
    shuffled = action(shuffled_obs, torch.tensor([7, 2, 0, 4, 1, 6, 3, 5]))
    torch.testing.assert_close(original, shuffled, atol=1e-6, rtol=1e-5)
    original.square().mean().backward()
    grads = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
    net.zero_grad(set_to_none=True)
    shuffled.square().mean().backward()
    for n, p in net.named_parameters():
        if n in grads:
            torch.testing.assert_close(grads[n], p.grad, atol=2e-6, rtol=2e-4)
    restored = build_network()
    restored.load_state_dict(net.state_dict(), strict=True)
    torch.testing.assert_close(net.eval_actor(obs)[0], restored.eval_actor(obs)[0], atol=1e-6, rtol=1e-5)
