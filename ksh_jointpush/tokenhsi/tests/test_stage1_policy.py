
"""Stage1 policy conditioning, permutation invariance and optimiser updates."""
from dataclasses import replace
from io import BytesIO
import pytest
import torch
import stage1_test_utils as reference
from stage1_test_utils import CFG, fused
from learning.multi_agent.task_role_encoder import (
    TaskActionFusion,
    task_action_condition,
)
from learning.multi_agent.amp_network_builder_ma import (
    RelationTransformerLayer,
    build_gta_transforms,
)
from utils.task_role_spec import task_graph_packet
from utils.edge_ontop_spec import permute_graph
from utils.edge_scenario_spec import (
    compose_canonical_graph,
    sample_graph,
    sample_size_conditioned_graph,
    classify_templates,
    agent_object_indices,
)
from utils.size_rsi import sample_sizes, task_probabilities
from utils.task_role_spec import task_size_probabilities, TASK_PROBS
from utils.unified_training import family_from_templates
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from env.tasks.multi_agent.edge_ontop_reward import evaluate_ontop_edges


def test_human_only_conditioning_manual_oracle_padding_and_task_record_order():
    torch.manual_seed(15)
    fusion = TaskActionFusion()
    torch.nn.init.normal_(fusion.mlp[-1].weight, std=0.05)
    embedding = torch.nn.Embedding(4, 64)
    packet = torch.tensor(
        [[1, 2, 1, 2, 6, 1, 0, 0, 0, 3], [1, 3, 0, 2, 4, 0, 0, 0, 0, 0]],
        dtype=torch.float,
    )
    x = torch.randn(2, 8, 64)
    condition = task_action_condition(packet, embedding)
    actual = fusion(x, condition, True)
    expected = x.clone()
    for b, records in enumerate(packet.reshape(2, 2, 5).long()):
        for valid, task, actor, payload, target in records:
            if valid:
                expected[b, actor] += fusion.mlp(
                    torch.cat((x[b, actor], embedding.weight[task]))
                )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:, 2:], x[:, 2:], atol=0, rtol=0)
    torch.testing.assert_close(actual[1, 1], x[1, 1], atol=0, rtol=0)
    flipped = packet.reshape(2, 2, 5).flip(1).flatten(1)
    torch.testing.assert_close(
        fusion(x, task_action_condition(flipped, embedding)), actual
    )
    assert all((torch.isfinite(v) for v in fusion.last_diagnostics.values()))


def test_post_gta_pre_output_projection_placement_and_attention_independence():
    torch.manual_seed(18)
    layer = RelationTransformerLayer(64, 2, 128)
    fusion = TaskActionFusion()
    torch.nn.init.normal_(fusion.mlp[-1].weight, std=0.05)
    x = torch.randn(2, 8, 64)
    pose = torch.randn(2, 8, 7)
    pose[..., 3:] = torch.nn.functional.normalize(pose[..., 3:], dim=-1)
    g, ginv = build_gta_transforms(pose)
    condition = (
        torch.tensor([[1, 0], [0, 1]]),
        torch.randn(2, 2, 64),
        torch.ones(2, 2, dtype=torch.bool),
    )
    captured = []
    hook = layer.proj.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    base_out = layer(x, gta_g=g, gta_ginv=ginv)
    result = layer(
        x, gta_g=g, gta_ginv=ginv, task_condition=condition, task_action_fusion=fusion
    )
    hook.remove()
    torch.testing.assert_close(captured[1], fusion(captured[0], condition))
    assert not torch.equal(base_out, result)
    swapped = (condition[0], condition[1].flip(1), condition[2])
    assert not torch.equal(
        result,
        layer(
            x, gta_g=g, gta_ginv=ginv, task_condition=swapped, task_action_fusion=fusion
        ),
    )


@pytest.mark.parametrize("branch", ["actor", "critic"])
def test_token_edge_task_shuffle_outputs_and_all_gradients(branch):
    torch.manual_seed(82)
    net = reference.network()
    enc = getattr(net, branch + "_encoder")
    head = net.action_head if branch == "actor" else net.value_head
    with torch.no_grad():
        enc.edge_encoder.bias_projection.uniform_(-0.1, 0.1)
        enc.task_action_fusion.mlp[-1].weight.normal_(std=0.02)
    obs = reference.observations(reference.all_pairs())
    original = head(enc(obs, token_order=torch.arange(8)))
    original.square().mean().backward()
    grads = {n: p.grad.clone() for n, p in enc.named_parameters() if p.grad is not None}
    assert (enc.edge_encoder.category_embed.weight.grad.abs().sum(-1) > 0).all()
    assert all((p.grad.abs().sum() > 0 for p in enc.task_action_fusion.parameters()))
    net.zero_grad(set_to_none=True)
    graph = permute_graph(reference.all_pairs(), torch.rand(16, 4).argsort(-1))
    packet = task_graph_packet(graph, 16).reshape(16, 2, 5).flip(1).flatten(1)
    changed = torch.cat((obs[:, :-10], packet), -1)
    result = head(enc(changed, token_order=torch.tensor([6, 4, 1, 3, 7, 0, 5, 2])))
    torch.testing.assert_close(original, result, atol=1e-05, rtol=1e-05)
    result.square().mean().backward()
    for name, p in enc.named_parameters():
        if name in grads:
            torch.testing.assert_close(grads[name], p.grad, atol=5e-05, rtol=0.001)


def test_zero_start_two_step_learning_shared_layers_and_independent_branches():
    torch.manual_seed(26)
    net = reference.network()
    initial = {}
    for branch in ("actor", "critic"):
        enc = getattr(net, branch + "_encoder")
        assert enc.task_action_fusion.mlp[-1].weight.count_nonzero() == 0
        assert enc.task_action_fusion.mlp[-1].bias.count_nonzero() == 0
        assert enc.edge_encoder.bias_projection.count_nonzero() == 0
        assert sum((p.numel() for p in enc.task_action_fusion.parameters())) == 12416
        assert not any(
            (
                "task_action" in n
                for layer in enc.layers
                for n, _ in layer.named_parameters()
            )
        )
        initial[branch] = {
            n: p.detach().clone()
            for n, p in enc.named_parameters()
            if "task_action" in n or "category_embed" in n
        }
        calls = []
        hook = enc.task_action_fusion.register_forward_hook(
            lambda m, a, r: calls.append(m)
        )
        obs = reference.observations(reference.all_pairs())
        enc(obs)
        hook.remove()
        assert len(calls) == 4 and all((m is enc.task_action_fusion for m in calls))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (
            net.eval_actor(obs)[0].square().mean()
            + net.eval_critic(obs).square().mean()
        ).backward()
        optimizer.step()
    for branch in ("actor", "critic"):
        enc = getattr(net, branch + "_encoder")
        for n, p in enc.named_parameters():
            if n in initial[branch]:
                assert not torch.equal(initial[branch][n], p), n
        assert (
            (
                initial[branch]["edge_encoder.category_embed.weight"]
                - enc.edge_encoder.category_embed.weight
            )
            .abs()
            .sum(-1)
            > 0
        ).all()
        assert enc.build_relation_bias().count_nonzero() == 0
    assert (
        net.actor_encoder.task_action_fusion
        is not net.critic_encoder.task_action_fusion
    )
    assert net.actor_encoder.edge_encoder is not net.critic_encoder.edge_encoder


def test_all_pairs_packet_binding_owner_and_edge_shuffle():
    graph = reference.all_pairs()
    ids = torch.tensor([1, 0, 5, 4, 3, 2, 7, 6])
    graph = replace(
        graph,
        edge_src=ids[graph.edge_src],
        edge_dst=ids[graph.edge_dst],
        edge_owner=1 - graph.edge_owner,
    )
    packet = task_graph_packet(graph, 16).reshape(16, 2, 5)
    for b in range(16):
        for owner in range(2):
            edges = graph.edge_valid[b] & (graph.edge_owner[b] == owner)
            primary = edges & (graph.edge_src[b] < 2)
            p = primary.nonzero()[0, 0]
            secondary = edges & ~primary
            actor, target = (int(graph.edge_src[b, p]), int(graph.edge_dst[b, p]))
            relation = int(graph.edge_relation[b, p])
            if secondary.any():
                q = secondary.nonzero()[0, 0]
                expected = [
                    1,
                    int(graph.edge_relation[b, q]) - 5,
                    actor,
                    target,
                    int(graph.edge_dst[b, q]),
                ]
            else:
                expected = [1, relation - 9, actor, 0, target]
            assert packet[b, owner].tolist() == expected
    shuffled = permute_graph(graph, torch.rand(16, 4).argsort(-1))
    torch.testing.assert_close(
        task_graph_packet(shuffled, 16), packet.flatten(1), atol=0, rtol=0
    )


def test_size_sampling_no_standalone_holding_preserves_family_rsi_bindings():
    torch.manual_seed(172)
    sizes = sample_sizes(12000, "cpu")
    weights = task_size_probabilities(sizes[:, :2])
    old = task_probabilities(sizes[:, :2])
    assert weights[..., 0].count_nonzero() == 0
    torch.testing.assert_close(
        weights[..., [0, 3, 4]].sum(-1), old[..., [0, 3, 4]].sum(-1)
    )
    torch.testing.assert_close(weights[..., 1:3], old[..., 1:3], atol=0, rtol=0)
    torch.testing.assert_close(
        weights.mean((0, 1)), torch.tensor(TASK_PROBS), atol=0.008, rtol=0
    )
    graph = sample_size_conditioned_graph(
        weights, CFG["relationGraph"], "random_scenario"
    )
    packet = task_graph_packet(graph, len(sizes)).reshape(-1, 5).long()
    rows = torch.arange(len(sizes)).repeat_interleave(2)
    actors = torch.arange(2).repeat(len(sizes))
    templates = classify_templates(graph, rows, actors, True)
    assert torch.equal(packet[:, 1], templates - 1)
    assert torch.equal(packet[:, 2], actors)
    assert torch.equal(agent_object_indices(graph, rows, actors), actors)
    assert torch.equal(
        family_from_templates(templates),
        torch.where(packet[:, 1] == 0, 1, torch.where(packet[:, 1] == 1, 2, 0)),
    )
    for preset, task in (("sit", 0), ("climb", 1), ("carry_at", 2), ("carry_ontop", 3)):
        fixed = sample_size_conditioned_graph(weights[:2], CFG["relationGraph"], preset)
        assert (task_graph_packet(fixed, 2).reshape(2, 2, 5)[..., 1] == task).all()
    for sampler in (
        lambda: sample_graph(2, CFG["relationGraph"], preset="holding"),
        lambda: sample_size_conditioned_graph(
            weights[:2], CFG["relationGraph"], "holding"
        ),
    ):
        with pytest.raises(ValueError, match="no standalone holding"):
            sampler()


def test_local_edge_sum_no_teammate_dependency_and_partial_suffix():
    graph = reference.all_pairs()
    new = Stage1ContextRuntime(16, graph, CFG["relationReward"], "cpu")
    phi, progress, error = (
        torch.rand(16, 4) * 0.8,
        torch.rand(16, 4),
        torch.ones(16, 4),
    )
    result = new.step(phi, progress, error, error, error)
    expected = torch.zeros(16, 2).scatter_add(
        1, graph.edge_owner, (0.2 * phi + 0.2 * progress) * graph.edge_valid
    )
    torch.testing.assert_close(result["agent_task_reward"], expected)
    assert result["teammate_task_reward"].count_nonzero() == 0
    changed = torch.where(graph.edge_owner == 1, 0.0, phi)
    again = new.step(changed, progress, error, error, error)
    torch.testing.assert_close(
        again["agent_task_reward"][:, 0], result["agent_task_reward"][:, 0]
    )
    ids = torch.tensor([15, 0, 8, 2])
    torch.testing.assert_close(new.suffix(ids), task_graph_packet(graph, 16)[ids])
    full = new.step(
        torch.ones(16, 4),
        torch.ones(16, 4),
        torch.zeros(16, 4),
        torch.zeros(16, 4),
        torch.zeros(16, 4),
    )
    assert full["agent_task_reward"][15].tolist() == pytest.approx([1.2, 1.2])
    assert full["agent_task_reward"][0].tolist() == pytest.approx([0.6, 0.6])


def test_carry_ontop_payload_above_support_and_at_goal_match_packet():
    graph = compose_canonical_graph(torch.tensor([[3, 4]]))
    packet = task_graph_packet(graph, 1).reshape(1, 2, 5).long()
    boxes = torch.zeros(1, 4, 13)
    boxes[..., 6] = 1
    boxes[0, 1, :3] = torch.tensor([2.0, 3.0, 0.7])
    boxes[0, 3, :3] = torch.tensor([2.0, 3.0, 0.25])
    sizes = torch.full((1, 4, 3), 0.4)
    sizes[0, 3, 2] = 0.5
    goals = torch.tensor([[[5.0, 6.0, 0.2], [-5.0, -6.0, 0.2]]])
    _, diag = evaluate_ontop_edges(
        torch.zeros(1, 2, 2, 3),
        torch.zeros(1, 2, 3),
        boxes,
        sizes,
        goals,
        graph,
        CFG["relationReward"],
    )
    at = (graph.edge_relation[0] == 7).nonzero()[0, 0]
    top = (graph.edge_relation[0] == 8).nonzero()[0, 0]
    assert packet[0, 1].tolist() == [1, 3, 1, 3, 5]
    torch.testing.assert_close(diag["target"][0, at], goals[0, packet[0, 0, 4] - 6])
    torch.testing.assert_close(
        diag["target"][0, top], boxes[0, packet[0, 1, 3] - 2, :3]
    )
    assert abs(float(diag["signed_gap"][0, top])) < 1e-06


def test_checkpoint_roundtrip_rebuild_and_actor_critic_separation():
    net = reference.network().eval()
    assert net.actor_encoder.context_fusion is not net.critic_encoder.context_fusion
    obs = reference.observations(reference.all_pairs())
    expected = net.eval_actor(obs)[0]
    value = net.eval_critic(obs)
    stream = BytesIO()
    torch.save(net.state_dict(), stream)
    stream.seek(0)
    restored = reference.network().eval()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    restored.set_entity_counts(2, 4)
    torch.testing.assert_close(
        restored.eval_actor(obs)[0], expected, atol=1e-06, rtol=1e-05
    )
    torch.testing.assert_close(restored.eval_critic(obs), value, atol=1e-06, rtol=1e-05)
    with pytest.raises(ValueError, match="packet architecture"):
        restored.actor_encoder.set_task_graph(
            {**CFG["relationGraph"], "policy_task_roles": False}
        )


def test_arbitrary_endpoint_and_type_permutation_remaps_bias():
    enc = reference.network().actor_encoder
    with torch.no_grad():
        enc.edge_encoder.bias_projection.normal_()
    packet = task_graph_packet(reference.all_pairs(), 16)
    base = fused(enc, packet)
    order = torch.tensor([6, 4, 1, 3, 7, 0, 5, 2])
    changed = packet.reshape(16, 2, 5).clone()
    changed[:, :, 2:] = order.argsort()[changed[:, :, 2:].long()].float()
    actual = fused(enc, changed.flatten(1), enc.entity_types[order])
    torch.testing.assert_close(
        actual, base.index_select(-2, order).index_select(-1, order)
    )
