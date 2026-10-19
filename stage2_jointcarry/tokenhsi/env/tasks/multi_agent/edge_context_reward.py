"""Current edge success, scalar context and ungated reward kernels."""
import torch
from utils.edge_context_spec import HOLDING, AT
from utils.edge_ontop_spec import batched, select_graph


def edge_context(phi, graph):
    pre_mask = graph.prereq_mask
    if pre_mask.ndim == 2:
        pre_mask = pre_mask[None]
    pre = torch.where(pre_mask, phi[:, None, :], torch.ones_like(phi[:, None, :])).amin(-1)
    term_index = batched(graph.term_index, phi.shape[0])
    term = torch.where(term_index >= 0, phi.gather(1, term_index.clamp_min(0)), 0.)
    return torch.stack([pre, term], -1) * batched(graph.edge_valid, phi.shape[0])[..., None]


def own_success(phi, z_error, graph, threshold=.9, z_tolerance=.001):
    return ((phi >= threshold) & ((batched(graph.edge_relation, phi.shape[0]) == HOLDING) |
            (z_error.abs() <= z_tolerance))) & batched(graph.edge_valid, phi.shape[0])


def owner_sum(values, graph):
    return values.new_zeros(values.shape[0], graph.num_agents).scatter_add(
        1, batched(graph.edge_owner, values.shape[0]), values)


def goal_success(success, graph):
    mask = batched(graph.required_goal & graph.edge_valid, success.shape[0])
    failed = owner_sum((~success & mask).float(), graph)
    count = owner_sum(mask.expand_as(success).float(), graph)
    return (failed == 0) & (count > 0)


def scene_success(success, graph):
    mask = batched(graph.required_goal & graph.edge_valid, success.shape[0])
    return (success | ~mask).all(-1) & mask.any(-1)


def edge_context_reward(phi, progress, success, graph, config):
    valid = batched(graph.edge_valid, phi.shape[0])
    success = success & valid
    term_index = batched(graph.term_index, phi.shape[0])
    term_success = (term_index >= 0) & success.gather(1, term_index.clamp_min(0))
    saturated = (success | term_success) & valid
    state = torch.where(saturated, 1., phi) * config['state_reward_weight'] * valid
    prog = torch.where(saturated, 1., progress) * config['progress_reward_weight'] * valid
    bonus = saturated.to(phi.dtype) * config['success_reward_weight']
    total = state + prog + bonus
    return dict(phi_raw=phi, progress_raw=progress, own_success=success, term_success=term_success & valid,
                reward_saturated=saturated, state_component=state, progress_component=prog,
                success_component=bonus, total=total, agent_task_reward=owner_sum(total, graph))


def evaluate_edge_geometry(hands, roots, objects, goals, graph, config):
    """Positions in one world frame; objects already reordered into logical slots."""
    N, E = roots.shape[0], graph.edge_src.numel()
    phi = roots.new_zeros(N, E); progress = torch.zeros_like(phi); z = torch.zeros_like(phi)
    distance = torch.zeros_like(phi); xy = torch.zeros_like(phi)
    for relation in (HOLDING, AT):
        ids = (graph.edge_relation == relation).nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        if relation == HOLDING:
            src = hands[:, graph.edge_src[ids]].mean(-2)
            dst = objects[:, graph.edge_dst[ids] - graph.num_agents]
            approach_src = roots[:, graph.edge_src[ids]]
            scale = config['holding']['hand_distance_scale']
        else:
            src = objects[:, graph.edge_src[ids] - graph.num_agents]
            dst = goals[:, graph.edge_dst[ids] - graph.num_agents - graph.num_objects]
            approach_src = src
            scale = config['at']['near_distance_scale']
        delta = src - dst
        distance[:, ids] = delta.norm(dim=-1)
        phi[:, ids] = torch.exp(-scale * delta.square().sum(-1))
        z[:, ids] = delta[..., 2].abs()
        dxy = (approach_src[..., :2] - dst[..., :2]).norm(dim=-1)
        xy[:, ids] = dxy
        p = config['progress']
        progress[:, ids] = 1 / (1 + (dxy - p['delta']).clamp_min(0) / p['sigma'])
    return phi, dict(progress=progress, z_error=z, distance=distance, distance_xy=xy)


class EdgeContextRuntime:
    def __init__(self, num_envs, graph, config, device):
        self.graph, self.config = graph, config
        self.phi = torch.zeros(num_envs, len(graph.ids), device=device)
        self.own_success = torch.zeros_like(self.phi, dtype=torch.bool)
        self.achieved = torch.zeros_like(self.own_success)  # diagnostics only
        self.done = torch.zeros(num_envs, graph.num_agents, dtype=torch.bool, device=device)  # current goals
        self.last_result = None

    def reset(self, ids, phi, z_error):
        self.phi[ids] = phi
        graph = select_graph(self.graph, ids)
        self.own_success[ids] = own_success(phi, z_error, graph,
            self.config['satisfaction_threshold'], self.config['success']['at_z_tolerance'])
        self.achieved[ids] = self.own_success[ids]
        self.done[ids] = goal_success(self.own_success[ids], graph)

    def step(self, phi, progress, z_error):
        success = own_success(phi, z_error, self.graph,
            self.config['satisfaction_threshold'], self.config['success']['at_z_tolerance'])
        result = edge_context_reward(phi, progress, success, self.graph, self.config)
        self.phi.copy_(phi); self.own_success.copy_(success); self.achieved |= success
        self.done.copy_(goal_success(success, self.graph))
        self.last_result = result
        return result

    def suffix(self, ids=None):
        phi = self.phi if ids is None else self.phi[ids]
        return edge_context(phi, self.graph if ids is None else select_graph(self.graph, ids)).flatten(1)
