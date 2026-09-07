import unittest

import torch
import torch.nn as nn

from learning.multi_agent.amp_network_builder_ma import (
    ENTITY_GOAL,
    ENTITY_HUMANOID,
    ENTITY_OBJECT,
    RELATION_BIAS_EDGE_MLP,
    RELATION_BIAS_LOOKUP,
    RelationEncoder,
    build_entity_type_ids,
)


ENTITY_SIZES = [5, 3, 2]
D_MODEL = 8
NUM_HEADS = 2
NUM_LAYERS = 2


def tokenizer(input_size):
    return nn.Sequential(
        nn.Linear(input_size, D_MODEL),
        nn.ReLU(),
        nn.Linear(D_MODEL, D_MODEL),
    )


def make_encoder(mode, num_agents=2, num_objects=3):
    return RelationEncoder(
        entity_sizes=ENTITY_SIZES,
        num_agents=num_agents,
        num_objects=num_objects,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=16,
        tokenizer_builder=tokenizer,
        relation_bias_mode=mode,
    )


def obs_width(num_agents, num_objects):
    return (num_agents * ENTITY_SIZES[0]
            + num_objects * ENTITY_SIZES[1]
            + num_agents * ENTITY_SIZES[2])


def legacy_a1_forward(encoder, obs):
    """The pre-A2 RelationEncoder.forward, retained here as a regression oracle."""
    batch_size = obs.shape[0]
    tokens = []
    offset = 0
    for i, (tok, size, count) in enumerate(zip(
            encoder.tokenizers, encoder.entity_sizes, encoder.entity_counts)):
        width = count * size
        block = obs[:, offset:offset + width].view(batch_size, count, size)
        tokens.append(tok(block) + encoder.type_embed[i])
        offset += width

    x = torch.cat(tokens, dim=1)
    for i, layer in enumerate(encoder.layers):
        x = layer(x, encoder.rel_embed[i][:, encoder.rel_matrix])
    return x[:, 0]


def split_obs(obs, num_agents, num_objects):
    humanoid_end = num_agents * ENTITY_SIZES[0]
    object_end = humanoid_end + num_objects * ENTITY_SIZES[1]
    humanoids = obs[:, :humanoid_end].view(-1, num_agents, ENTITY_SIZES[0])
    objects = obs[:, humanoid_end:object_end].view(-1, num_objects, ENTITY_SIZES[1])
    goals = obs[:, object_end:].view(-1, num_agents, ENTITY_SIZES[2])
    return humanoids, objects, goals


def join_obs(humanoids, objects, goals):
    return torch.cat([
        humanoids.flatten(1),
        objects.flatten(1),
        goals.flatten(1),
    ], dim=-1)


def full_encoder_output(encoder, obs):
    """Return every contextual token instead of RelationEncoder's ego-only token."""
    batch_size = obs.shape[0]
    tokens = []
    offset = 0
    for i, (tok, size, count) in enumerate(zip(
            encoder.tokenizers, encoder.entity_sizes, encoder.entity_counts)):
        width = count * size
        block = obs[:, offset:offset + width].view(batch_size, count, size)
        tokens.append(tok(block) + encoder.type_embed[i])
        offset += width

    x = torch.cat(tokens, dim=1)
    relation_bias = encoder.build_relation_bias()
    for i, layer in enumerate(encoder.layers):
        x = layer(x, relation_bias[i])
    return x


class RelationBiasTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(17)

    def test_entity_type_layout(self):
        ids = build_entity_type_ids(2, 3)
        expected = torch.tensor([
            ENTITY_HUMANOID, ENTITY_HUMANOID,
            ENTITY_OBJECT, ENTITY_OBJECT, ENTITY_OBJECT,
            ENTITY_GOAL, ENTITY_GOAL,
        ])
        self.assertTrue(torch.equal(ids, expected))

    def test_a1_matches_legacy_forward_exactly(self):
        encoder = make_encoder(RELATION_BIAS_LOOKUP)
        obs = torch.randn(4, obs_width(2, 3))
        expected = legacy_a1_forward(encoder, obs)
        actual = encoder(obs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_a2_shapes_and_zero_initial_bias(self):
        encoder = make_encoder(RELATION_BIAS_EDGE_MLP)
        edges = encoder.build_edge_embeddings()
        bias = encoder.build_relation_bias()

        self.assertEqual(edges.shape, (7, 7, 64))
        self.assertEqual(bias.shape, (NUM_LAYERS, NUM_HEADS, 7, 7))
        torch.testing.assert_close(bias, torch.zeros_like(bias), rtol=0, atol=0)

        obs = torch.randn(4, obs_width(2, 3))
        self.assertEqual(encoder(obs).shape, (4, D_MODEL))

    def test_a2_actor_and_critic_gradients(self):
        actor = make_encoder(RELATION_BIAS_EDGE_MLP)
        critic = make_encoder(RELATION_BIAS_EDGE_MLP)
        action_head = nn.Linear(D_MODEL, 4)
        value_head = nn.Linear(D_MODEL, 1)
        obs = torch.randn(8, obs_width(2, 3))

        loss = action_head(actor(obs)).square().mean() + value_head(critic(obs)).square().mean()
        loss.backward()

        for encoder in (actor, critic):
            grad = encoder.edge_encoder.bias_projection.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0.0)

            # Exact zero projection intentionally gates the edge MLP on the first step.
            # Once the projection has moved, gradients must reach the whole edge encoder.
            with torch.no_grad():
                encoder.edge_encoder.bias_projection.add_(grad)
            encoder.zero_grad(set_to_none=True)

        action_head.zero_grad(set_to_none=True)
        value_head.zero_grad(set_to_none=True)
        second_loss = (action_head(actor(obs)).square().mean()
                       + value_head(critic(obs)).square().mean())
        second_loss.backward()

        for encoder in (actor, critic):
            edge_encoder = encoder.edge_encoder
            for parameter in (
                    edge_encoder.source_type_embed.weight,
                    edge_encoder.relation_embed.weight,
                    edge_encoder.target_type_embed.weight,
                    edge_encoder.edge_mlp[0].weight,
                    edge_encoder.edge_mlp[2].weight):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_both_modes_are_permutation_equivariant(self):
        num_agents, num_objects = 3, 4
        obs = torch.randn(2, obs_width(num_agents, num_objects))
        humanoids, objects, goals = split_obs(obs, num_agents, num_objects)

        for mode in (RELATION_BIAS_LOOKUP, RELATION_BIAS_EDGE_MLP):
            encoder = make_encoder(mode, num_agents, num_objects).eval()
            with torch.no_grad():
                if mode == RELATION_BIAS_LOOKUP:
                    encoder.rel_embed.normal_()
                else:
                    encoder.edge_encoder.bias_projection.normal_()

            canonical = full_encoder_output(encoder, obs)
            for shift in range(num_agents):
                permutation = (torch.arange(num_agents) + shift) % num_agents
                permuted_obs = join_obs(
                    humanoids[:, permutation],
                    torch.cat([objects[:, :num_agents][:, permutation],
                               objects[:, num_agents:]], dim=1),
                    goals[:, permutation],
                )
                actual = encoder(permuted_obs)
                expected = canonical[:, permutation[0]]
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_entity_count_change_and_checkpoint_loading(self):
        for mode in (RELATION_BIAS_LOOKUP, RELATION_BIAS_EDGE_MLP):
            source = make_encoder(mode, 1, 1)
            target = make_encoder(mode, 4, 6)
            target.load_state_dict(source.state_dict(), strict=True)

            source.set_entity_counts(4, 6)
            obs = torch.randn(3, obs_width(4, 6))
            self.assertEqual(source(obs).shape, (3, D_MODEL))
            self.assertEqual(source.rel_matrix.shape, (14, 14))
            self.assertEqual(source.entity_type_ids.shape, (14,))
            self.assertEqual(source.build_relation_bias().shape,
                             (NUM_LAYERS, NUM_HEADS, 14, 14))


if __name__ == "__main__":
    unittest.main()
