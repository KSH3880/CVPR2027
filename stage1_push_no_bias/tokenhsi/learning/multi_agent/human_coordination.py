import torch
import torch.nn as nn


class CoordinationBlock(nn.Module):
    def __init__(self, width, heads, feedforward_dim, relations, bias_mode="all"):
        super().__init__()
        if bias_mode not in ("all", "before_only_mask_none"):
            raise ValueError("Unknown human attention bias mode")
        if bias_mode != "all" and relations != 3:
            raise ValueError("Masked attention requires NONE/SELF/BEFORE relations")
        self.bias_mode = bias_mode
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.query_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, width),
        )
        # Restricted modes store only BEFORE; NONE/SELF have no parameters.
        self.relation_bias = nn.Parameter(torch.zeros(heads, relations if bias_mode == "all" else 1))

    def effective_relation_bias(self):
        if self.bias_mode != "all":
            return torch.cat((self.relation_bias.new_zeros(self.relation_bias.shape[0], 2),
                              self.relation_bias), dim=-1)
        return self.relation_bias

    def forward(self, humans, tokens, relations, padding):
        bias = self.effective_relation_bias()[:, relations].permute(1, 0, 2, 3)
        if self.bias_mode == "before_only_mask_none":
            padding = padding | (relations == 0)[:, None]
        bias = bias.masked_fill(padding, float("-inf")).flatten(0, 1)
        query = self.query_norm(humans)
        if tokens is None:
            tokens = query
        context, _ = self.attention(
            query, tokens, tokens, attn_mask=bias, need_weights=False
        )
        humans = humans + context
        return humans + self.ffn(self.ffn_norm(humans))


def human_relations(packet, num_agents):
    packet = packet.reshape(packet.shape[0], -1, 5).long()
    ids = torch.arange(num_agents, device=packet.device)
    owners = packet[..., 0].bool()[..., None] & (packet[..., 2, None] == ids)
    valid, task, actor, payload, target = packet.unbind(-1)
    dependency = (valid.bool()[:, :, None] & valid.bool()[:, None, :]
        & (task[:, :, None] == 2)
        & ((task[:, None, :] == 0) | (task[:, None, :] == 1) | (task[:, None, :] == 3))
        & (payload[:, :, None] == target[:, None, :])
        & (actor[:, :, None] != actor[:, None, :]))
    predecessor = torch.einsum('nqi,nji,njk->nqk', owners.transpose(1, 2).float(),
                               dependency.float(), owners.float()).bool()
    return torch.where(torch.eye(num_agents, device=packet.device, dtype=torch.bool),
                       1, torch.where(predecessor, 2, 0))


class HumanCoordination(nn.Module):
    def __init__(
        self,
        width=64,
        heads=2,
        num_layers=4,
        feedforward_dim=128,
        shuffle_human_order=False,
        bias_mode="all",
        relations=3,
    ):
        super().__init__()
        self.shuffle_human_order = shuffle_human_order
        self.blocks = nn.ModuleList(
            [
                CoordinationBlock(width, heads, feedforward_dim, relations, bias_mode=bias_mode)
                for _ in range(num_layers)
            ]
        )

    def init_identity(self):
        for block in self.blocks:
            nn.init.zeros_(block.attention.out_proj.weight)
            nn.init.zeros_(block.attention.out_proj.bias)
            nn.init.zeros_(block.ffn[-1].weight)
            nn.init.zeros_(block.ffn[-1].bias)

    def forward(
        self, humans, nodes, packet, edge_encoder, entity_types=None, entity_poses=None, groups=None
    ):
        relations = human_relations(packet, humans.shape[1]) if groups is None else stage2_human_relations(packet, groups)
        context = humans
        if self.shuffle_human_order:
            order = torch.randperm(humans.shape[1], device=humans.device)
            context = context[:, order]
            relations = relations[:, order][:, :, order]
        padding = torch.zeros((), device=humans.device, dtype=torch.bool)
        for block in self.blocks:
            context = block(context, None, relations, padding)
        if self.shuffle_human_order:
            context = context[:, order.argsort()]
        has_task = packet.reshape(packet.shape[0], -1, 5)[..., 0].bool().any(-1)
        return torch.where(has_task[:, None, None], context, humans)


def stage2_human_relations(packet,groups):
    """Directed BEFORE dependencies and symmetric coupled PUSH membership."""
    m=groups.shape[1]
    relation=human_relations(packet,m)
    records=packet.reshape(len(packet),m,5).long()
    push=torch.zeros_like(groups,dtype=torch.bool)
    push.scatter_(1,records[...,2],records[...,0].bool()&(records[...,1]==4))
    coupled=(groups[:,:,None]==groups[:,None])&push[:,:,None]&push[:,None]
    coupled &= ~torch.eye(m,device=packet.device,dtype=torch.bool)
    return torch.where(coupled,3,relation)
