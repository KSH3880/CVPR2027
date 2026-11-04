"""First-placement, one-shot approach deadline (no IsaacGym dependency)."""
import math
import torch
from utils.edge_ontop_spec import batched


def validate_approach_timeout(config):
    if not isinstance(config, dict) or set(config) != {'mode', 'seconds', 'min_improvement_m'}:
        raise ValueError('approach_timeout needs mode, seconds and min_improvement_m')
    if config['mode'] != 'first_approach_v1':
        raise ValueError('approach_timeout mode must be first_approach_v1')
    for value in (config['seconds'], config['min_improvement_m']):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('approach_timeout values must be finite and positive')


def follower_goal_distances(roots, goals, graph):
    n = len(roots)
    parent = graph.prereq_mask.unsqueeze(0).expand(n, -1, -1) if graph.prereq_mask.dim() == 2 else graph.prereq_mask
    predecessor = parent.long().argmax(-1)
    dst = batched(graph.edge_dst, n).gather(1, predecessor)
    goal = (dst - graph.num_agents - graph.num_objects).clamp(0, goals.shape[1]-1)
    actor = batched(graph.edge_owner, n)
    rows = torch.arange(n, device=roots.device)[:, None]
    return (roots[rows, actor, :2] - goals[rows, goal, :2]).norm(dim=-1)


class ApproachTimeout:
    def __init__(self, shape, device, dt, config):
        validate_approach_timeout(config)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('control dt must be positive')
        # dt may be a float32 value; tolerate roundoff at an integral step count.
        self.limit = math.ceil(config['seconds'] / dt - 1e-6)
        self.min_improvement = config['min_improvement_m']
        self.started = torch.zeros(shape, dtype=torch.bool, device=device)
        self.completed = torch.zeros_like(self.started)
        self.start_distance = torch.zeros(shape, device=device)
        self.idle = torch.zeros(shape, dtype=torch.long, device=device)

    def reset(self, ids):
        self.started[ids] = False
        self.completed[ids] = False
        self.start_distance[ids] = 0
        self.idle[ids] = 0

    def step(self, distance, gate, success, dependent):
        opening = dependent & gate & ~self.started
        self.start_distance.copy_(torch.where(opening, distance, self.start_distance))
        self.started |= opening
        self.completed |= dependent & success
        active = dependent & self.started & ~self.completed
        improved = active & (self.start_distance - distance >= self.min_improvement)
        self.completed |= improved
        active &= ~self.completed
        self.idle.copy_(torch.where(active & ~opening & ~improved,
                                    self.idle + 1, torch.zeros_like(self.idle)))
        return active & (self.idle >= self.limit)


def apply_timeout_reset(expired, reset_buf, terminate_buf):
    """A scene reset terminates all agents, avoiding bootstrap across the reset."""
    scene = expired.any(-1)
    reset_buf[scene] = 1
    terminate_buf.view(len(scene), -1)[scene] = 1
    return scene


TERMINATION_REASONS = ('fall_only', 'approach_timeout_only', 'fall_and_approach_timeout', 'horizon_only', 'all')


def termination_reason_counts(fall, timeout, reset):
    """Disjoint scene reasons; coincident horizon is assigned to failure."""
    ended = reset.bool()
    return torch.stack(((fall & ~timeout & ended).sum(),
                        (timeout & ~fall & ended).sum(),
                        (fall & timeout & ended).sum(),
                        (ended & ~fall & ~timeout).sum(), ended.sum()))
