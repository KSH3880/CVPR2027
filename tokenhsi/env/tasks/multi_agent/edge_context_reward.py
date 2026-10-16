import torch


from utils.edge_context_spec import HOLDING


from utils.edge_ontop_spec import batched


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


class EdgeContextRuntime:
    def __init__(self, num_envs, graph, config, device):
        self.graph, self.config = graph, config
        self.phi = torch.zeros(num_envs, len(graph.ids), device=device)
        self.own_success = torch.zeros_like(self.phi, dtype=torch.bool)
        self.achieved = torch.zeros_like(self.own_success)  # diagnostics only
        self.done = torch.zeros(num_envs, graph.num_agents, dtype=torch.bool, device=device)  # current goals
        self.last_result = None
