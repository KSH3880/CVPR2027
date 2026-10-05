"""Variable-size allocation features; no entity IDs or positional embeddings."""
from __future__ import annotations

import math
import torch
from torch import nn


class TaskAllocationModel(nn.Module):
    """Encode matched box/goal pairs, then let agents read task values.

    Inputs: agent_obs [B,A,Da], box_obs [B,N,Db], goal_obs [B,N,Dg].
    Box j and goal j must describe the same delivery task. task_valid [B,N]
    masks padding/completed tasks; selectable [B,A,N] masks agent-specific
    restrictions. Output probabilities include a final idle column. They are
    independent agent distributions, NOT a collision-free joint assignment.
    """

    def __init__(self, agent_dim: int, box_dim: int, goal_dim: int,
                 d_model: int = 128, nhead: int = 4, layers: int = 1):
        super().__init__()
        if min(agent_dim, box_dim, goal_dim, d_model, nhead, layers) <= 0:
            raise ValueError("dimensions and layers must be positive")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.agent_proj = nn.Linear(agent_dim, d_model)
        self.box_proj = nn.Linear(box_dim, d_model)
        self.goal_proj = nn.Linear(goal_dim, d_model)

        def encoder():
            block = nn.TransformerEncoderLayer(
                d_model, nhead, 4 * d_model, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True)
            return nn.TransformerEncoder(
                block, layers, norm=nn.LayerNorm(d_model),
                enable_nested_tensor=False)

        self.agent_encoder = encoder()
        self.task_encoder = encoder()
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=0.0, batch_first=True)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.idle_head = nn.Linear(d_model, 1)
        self.d_model = d_model

    @staticmethod
    def pair_attention_mask(tasks: int, device=None):
        """True blocks attention; order is box0, goal0, box1, goal1, ..."""
        pair = torch.arange(tasks, device=device).repeat_interleave(2)
        return pair[:, None] != pair[None, :]

    def forward(self, agent_obs, box_obs, goal_obs,
                task_valid=None, selectable=None):
        if any(x.ndim != 3 for x in (agent_obs, box_obs, goal_obs)):
            raise ValueError("observations must have shape [batch, entities, features]")
        b, a, _ = agent_obs.shape
        n = box_obs.shape[1]
        if a == 0 or box_obs.shape[0] != b or goal_obs.shape[:2] != (b, n):
            raise ValueError("nonempty agents and matching box/goal batch sizes required")
        if task_valid is None:
            task_valid = torch.ones(b, n, dtype=torch.bool, device=box_obs.device)
        if task_valid.shape != (b, n) or task_valid.dtype != torch.bool:
            raise ValueError("task_valid must be bool [B,N]")
        allowed = task_valid[:, None, :].expand(b, a, n)
        if selectable is not None:
            if selectable.shape != (b, a, n) or selectable.dtype != torch.bool:
                raise ValueError("selectable must be bool [B,A,N]")
            allowed = allowed & selectable

        agents = self.agent_encoder(self.agent_proj(agent_obs))
        if n:
            # Sanitize invalid observations before projection (even NaN padding).
            boxes = box_obs.masked_fill(~task_valid[..., None], 0)
            goals = goal_obs.masked_fill(~task_valid[..., None], 0)
            pairs = torch.stack((self.box_proj(boxes), self.goal_proj(goals)), dim=2)
            tokens = pairs.flatten(1, 2)
            encoded = self.task_encoder(
                tokens, mask=self.pair_attention_mask(n, tokens.device))
            tasks = encoded.reshape(b, n, 2, self.d_model).mean(dim=2)
            tasks = tasks.masked_fill(~task_valid[..., None], 0)
        else:
            tasks = agents.new_empty(b, 0, self.d_model)

        # Fixed zero null value keeps all-masked/no-task scenes finite.
        values = torch.cat((tasks, agents.new_zeros(b, 1, self.d_model)), dim=1)
        cross_allowed = torch.cat((allowed, torch.ones(
            b, a, 1, dtype=torch.bool, device=agents.device)), dim=-1)
        mask = (~cross_allowed).repeat_interleave(
            self.cross_attention.num_heads, dim=0)
        context, weights = self.cross_attention(
            agents, values, values, attn_mask=mask,
            need_weights=True, average_attn_weights=False)
        logits = torch.matmul(self.query(agents + context), self.key(tasks).transpose(1, 2))
        logits = logits / math.sqrt(self.d_model)
        logits = logits.masked_fill(~allowed, -torch.inf)
        logits = torch.cat((logits, self.idle_head(agents + context)), dim=-1)
        return {"logits": logits, "probabilities": logits.softmax(dim=-1),
                "agent_features": agents, "task_features": tasks,
                "task_context": context, "cross_attention": weights}
