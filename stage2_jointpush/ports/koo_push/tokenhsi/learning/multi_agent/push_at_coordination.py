"""Model-only PUSH+AT coordination; Stage-1 graph/observation stays unchanged."""
import torch
from torch import nn

PUSH, AT = 12, 7


class CoordinationBlock(nn.Module):
    def __init__(self, width=64, heads=2, feedforward=128):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.query_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, feedforward), nn.ReLU(), nn.Linear(feedforward, width))
        self.relation_bias = nn.Parameter(torch.zeros(heads, 3))

    def forward(self, humans, tokens, relations, valid):
        bias = self.relation_bias[:, relations].permute(1, 0, 2, 3)
        bias = bias.masked_fill(~valid[:, None, None], float('-inf')).flatten(0, 1)
        context, _ = self.attention(self.query_norm(humans), tokens, tokens,
                                    attn_mask=bias, need_weights=False)
        humans = humans + context
        return humans + self.ffn(self.ffn_norm(humans))


class PushAtCoordination(nn.Module):
    def __init__(self, width=64, hidden=128, heads=2, num_layers=4, feedforward=128):
        super().__init__()
        self.width = width
        self.grounding = nn.Sequential(nn.Linear(5 * width, hidden), nn.ReLU(), nn.Linear(hidden, width))
        self.blocks = nn.ModuleList([CoordinationBlock(width, heads, feedforward) for _ in range(num_layers)])

    def bundles(self, packet, num_agents):
        if packet.ndim != 2 or packet.shape[1] % 5:
            raise ValueError('Expected semantic edge packet [N, 5*E]')
        raw = packet.reshape(packet.shape[0], -1, 5)
        if not torch.isfinite(raw).all() or not torch.equal(raw, raw.round()):
            raise ValueError('Graph fields must be finite integers')
        valid, src, dst, relation, owner = raw.long().unbind(-1)
        if ((valid != 0) & (valid != 1)).any():
            raise ValueError('Invalid edge validity')
        active = valid.bool()
        if (active & ((owner < 0) | (owner >= num_agents))).any():
            raise ValueError('Invalid edge owner')
        bundles = []
        for agent in range(num_agents):
            push = active & (relation == PUSH) & (owner == agent)
            if (push.sum(-1) > 1).any():
                raise ValueError('Each agent supports one PUSH bundle')
            index = push.long().argmax(-1)
            actor = src.gather(1, index[:, None]).squeeze(1)
            box = dst.gather(1, index[:, None]).squeeze(1)
            at = active & (relation == AT) & (owner == agent) & (src == box[:, None])
            if (push.any(-1) & (at.sum(-1) != 1)).any():
                raise ValueError('PUSH requires exactly one matching owner/payload AT edge')
            goal = dst.gather(1, at.long().argmax(-1)[:, None]).squeeze(1)
            has = push.any(-1)
            if (has & (actor != agent)).any():
                raise ValueError('PUSH actor must match edge owner')
            bundles.append(torch.stack((has, actor, box, goal), -1))
        return torch.stack(bundles, 1)

    def task_tokens(self, nodes, packet, edge_encoder, entity_types):
        num_agents = int((entity_types == entity_types[0]).sum().item())
        bundles = self.bundles(packet, num_agents)
        valid, actor, box, goal = bundles.unbind(-1)
        valid = valid.bool()
        refs = [x.masked_fill(~valid, 0) for x in (actor, box, goal)]
        if any(((x < 0) | (x >= nodes.shape[1])).any() for x in refs):
            raise ValueError('Task endpoint outside scene')
        actor, box, goal = refs
        if (valid & ((entity_types[box] != 1) | (entity_types[goal] != 2))).any():
            raise ValueError('PUSH+AT needs human/object/goal endpoints')
        row = torch.arange(nodes.shape[0], device=nodes.device)[:, None]
        def semantic(source, relation, target):
            return edge_encoder.edge_mlp(torch.cat((
                edge_encoder.source_type_embed(entity_types[source]),
                edge_encoder.relation_embed(torch.full_like(source, relation)),
                edge_encoder.target_type_embed(entity_types[target])), -1))
        with torch.no_grad():
            push = semantic(actor, PUSH, box)
            at = semantic(box, AT, goal)
        features = torch.cat((nodes[row, actor], nodes[row, box], nodes[row, goal], push, at), -1)
        tokens = self.grounding(features).masked_fill(~valid[..., None], 0)
        return tokens, valid, bundles

    def forward(self, humans, nodes, packet, edge_encoder, entity_types):
        tokens, valid, bundles = self.task_tokens(nodes, packet, edge_encoder, entity_types)
        order = torch.randperm(tokens.shape[1], device=tokens.device)
        tokens, valid, bundles = tokens[:, order], valid[:, order], bundles[:, order]
        _, actor, box, goal = bundles.unbind(-1)
        own = actor[:, None] == torch.arange(humans.shape[1], device=nodes.device)[None, :, None]
        physical = (box - humans.shape[1]) // getattr(self, 'partition_count', 1)
        same_goal = goal[:, :, None] == goal[:, None, :]
        same_payload = same_goal if getattr(self, 'group_by_goal', False) else (physical[:, :, None] == physical[:, None, :])
        shared = valid[:, :, None] & valid[:, None, :] & same_payload & same_goal
        partner = (own[..., None] & shared[:, None]).any(-2) & ~own
        relations = torch.where(own, 1, torch.where(partner, 2, 0))
        has_task = valid.any(-1)
        safe = valid.clone(); safe[~has_task, 0] = True
        context = humans
        for block in self.blocks:
            context = block(context, tokens, relations, safe)
        return context.masked_fill(~has_task[:, None, None], 0)


def attach_push_at_coordination(network):
    """Call AFTER loading the Stage-1 model. Leaves original outputs unchanged initially."""
    if hasattr(network, 'push_at_coordination') or network.stage2:
        raise ValueError('Coordination is already attached')
    encoder = network.actor_encoder
    if network.observation_mode != 'clean_scene' or not encoder.semantic_only:
        raise ValueError('Requires koo semantic clean-scene encoder')
    first = network.action_head[0][0]
    if not isinstance(first, nn.Linear) or first.in_features != 64:
        raise ValueError('Expected 64-D Stage-1 action head')
    expanded = nn.Linear(128, first.out_features, bias=first.bias is not None).to(first.weight)
    with torch.no_grad():
        expanded.weight.zero_(); expanded.weight[:, :64].copy_(first.weight)
        if first.bias is not None: expanded.bias.copy_(first.bias)
    network.action_head[0][0] = expanded
    network.push_at_coordination = PushAtCoordination().to(first.weight)
    encoder.requires_grad_(False)
    return network
