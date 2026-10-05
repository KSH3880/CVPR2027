"""Simulator-independent joint policy, reward and semi-Markov returns."""
import torch
from torch import nn
from torch.distributions import Categorical
from coordinator.task_allocation import TaskAllocationModel

SCHEMA = 'ms18-task-allocation-v1'


class AllocationPolicy(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        # root pos/quat/velocity, held, phase, remaining time; task box pose,
        # velocities, size, current owner one-hot, held and delivered.
        self.model = TaskAllocationModel(16, 20, 3, d_model=d_model)
        self.value = nn.Sequential(nn.Linear(2*d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))

    def forward(self, obs):
        out = self.model(obs['agent'], obs['box'], obs['goal'])
        score = out['logits'][..., :2]
        # Two valid permutations, not independent per-agent categorical draws.
        logits = torch.stack((score[:, 0, 0]+score[:, 1, 1],
                              score[:, 0, 1]+score[:, 1, 0]), -1)
        locked = obs['locked'].any(-1) | obs['delivered'].any(-1)
        current = obs['assignment'][:, 0]
        legal = ~locked[:, None] | (torch.arange(2, device=logits.device)[None] == current[:, None])
        logits = logits.masked_fill(~legal, -torch.inf)
        features = torch.cat((out['agent_features'].mean(1), out['task_features'].mean(1)), -1)
        return Categorical(logits=logits), self.value(features).squeeze(-1)


def assignment_from_action(action):
    return torch.stack((action, 1-action), -1).long()


def switch_cost(previous, selected, initialized, coefficient):
    changed = (previous != selected).sum(-1).float()
    return -coefficient * changed * initialized.float()


def step_reward(new_deliveries, failure, dt, time_coef=1., delivery_coef=10., failure_coef=40.):
    return -time_coef * dt + delivery_coef * new_deliveries.float() - failure_coef * failure.float()


def gae(reward, value, next_value, done, duration, gamma=.99, lam=.95):
    """gamma/lambda are per simulator step; terminal transitions never bootstrap."""
    discount = gamma ** duration
    trace_discount = (gamma * lam) ** duration
    adv = torch.zeros_like(reward)
    carry = torch.zeros_like(reward[0])
    for t in reversed(range(len(reward))):
        live = (~done[t]).float()
        delta = reward[t] + discount[t]*next_value[t]*live-value[t]
        carry = delta + trace_discount[t]*live*carry
        adv[t] = carry
    return adv, adv+value


def sample_layout(count, device, extent=3., clearance=1.2, max_tries=200):
    """Six mutually separated points: agents, boxes, then goals."""
    if extent <= 0 or clearance <= 0:
        raise ValueError('extent/clearance must be positive')
    result = torch.empty(count, 6, 2, device=device)
    for j in range(6):
        pending = torch.ones(count, dtype=torch.bool, device=device)
        for _ in range(max_tries):
            ids = pending.nonzero(as_tuple=False).flatten()
            if not len(ids):
                break
            candidate = (torch.rand(len(ids), 2, device=device)*2-1)*extent
            valid = torch.ones(len(ids), dtype=torch.bool, device=device) if j == 0 else (
                (candidate[:, None]-result[ids, :j]).norm(dim=-1) >= clearance).all(-1)
            result[ids[valid], j] = candidate[valid]
            pending[ids[valid]] = False
        if pending.any():
            raise ValueError('layout rejection budget exhausted; increase extent or reduce clearance')
    return result


def nearest_initial_action(obs, initialized):
    """Evaluation baseline: minimize nominal joint makespan, then keep ownership."""
    distance=torch.cdist(obs['agent'][..., :2],obs['box'][..., :2])
    carry=(obs['box'][..., :2]-obs['goal'][..., :2]).norm(dim=-1)
    total=distance+carry[:,None]
    direct=torch.stack((total[:,0,0],total[:,1,1]),-1).amax(-1)
    swapped=torch.stack((total[:,0,1],total[:,1,0]),-1).amax(-1)
    initial_action=(swapped<direct).long()
    return torch.where(initialized,obs['assignment'][:,0],initial_action)
