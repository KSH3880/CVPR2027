"""Non-preemptive two-agent scheduling of four shared delivery jobs."""
import itertools
import torch
from torch import nn
from torch.distributions import Categorical
from coordinator.task_allocation import TaskAllocationModel

SCHEMA_FOUR = 'ms18-four-box-scheduling-v1'


def joint_choices(boxes=4, device=None):
    # -1 is idle. Duplicate non-idle ownership is never a candidate.
    return torch.tensor([p for p in itertools.product(range(-1, boxes), repeat=2)
                         if p[0] < 0 or p[1] < 0 or p[0] != p[1]],
                        dtype=torch.long, device=device)


def legal_choices(obs, choices):
    current, delivered = obs['assignment'], obs['delivered']
    busy = current >= 0
    candidate = choices[None].expand(len(current), -1, -1)
    keep = (~busy[:, None] | (candidate == current[:, None])).all(-1)
    available = ~delivered
    selected_done = delivered.gather(1, choices.clamp(min=0).flatten()[None].expand(len(current), -1))
    selected_done = selected_done.reshape_as(candidate) & (candidate >= 0)
    # Fill all available workers, except when fewer jobs remain than workers.
    target_workers = available.sum(-1).clamp(max=2)
    work_conserving = (candidate >= 0).sum(-1) == target_workers[:, None]
    return keep & ~selected_done.any(-1) & work_conserving


class FourBoxPolicy(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.model = TaskAllocationModel(16, 20, 3, d_model=d_model)
        self.value = nn.Sequential(nn.Linear(2*d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        self.register_buffer('choices', joint_choices())

    def forward(self, obs):
        out = self.model(obs['agent'], obs['box'], obs['goal'], task_valid=~obs['delivered'])
        # Idle is the final model column, so map -1 to the task count.
        index = self.choices.clone()
        index[index < 0] = obs['box'].shape[1]
        score = out['logits'][:, 0, index[:, 0]] + out['logits'][:, 1, index[:, 1]]
        legal = legal_choices(obs, self.choices)
        if not legal.any(-1).all():
            raise ValueError('no legal assignment: busy/delivered bookkeeping is inconsistent')
        score = score.masked_fill(~legal, -torch.inf)
        features = torch.cat((out['agent_features'].mean(1), out['task_features'].mean(1)), -1)
        return Categorical(logits=score), self.value(features).squeeze(-1)

    def assignment_from_action(self, action):
        return self.choices[action]

    def nearest_action(self, obs):
        distance = torch.cdist(obs['agent'][..., :2], obs['box'][..., :2])
        carry = (obs['box'][..., :2]-obs['goal'][..., :2]).norm(dim=-1)
        distance = torch.where(obs['agent'][..., 13:14] >= .5, torch.zeros_like(distance), distance)
        remaining = distance + carry[:, None]
        parts=[]
        for a in range(2):
            selected=remaining[:, a, self.choices[:, a].clamp(min=0)]
            parts.append(torch.where(self.choices[None, :, a] < 0, 0., selected))
        lengths=torch.stack(parts, -1)
        cost=lengths.amax(-1)+.01*lengths.sum(-1)
        cost=cost.masked_fill(~legal_choices(obs, self.choices), torch.inf)
        return cost.argmin(-1)
