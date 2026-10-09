import torch

from utils.edge_context_spec import HOLDING, AT
from utils.edge_ontop_spec import ON_TOP, batched, select_graph
from utils.mixed_carry_spec import joint_scenes
from env.tasks.multi_agent.edge_context_reward import goal_success, owner_sum
from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success, quat_rotate
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime


def cpa_collision_penalty(root_pos, root_vel, min_distance, discount, dt):
    relative_pos = root_pos[:, :, None, :2] - root_pos[:, None, :, :2]
    relative_vel = root_vel[:, :, None, :2] - root_vel[:, None, :, :2]
    dot = (relative_pos * relative_vel).sum(-1)
    speed_squared = relative_vel.square().sum(-1)
    denominator = (relative_pos.norm(dim=-1) * relative_vel.norm(dim=-1)).clamp_min(1e-6)
    closing = (-dot / denominator).clamp(0., 1.)
    time = (-dot / speed_squared.clamp_min(1e-6)).clamp_min(0.)
    distance = (relative_pos + time[..., None] * relative_vel).norm(dim=-1)
    risk = ((min_distance - distance) / min_distance).clamp(0., 1.)
    urgency = torch.pow(torch.full_like(time, discount), time / dt)
    violation = closing * risk * urgency
    diagonal = torch.eye(root_pos.shape[1], device=root_pos.device, dtype=torch.bool)
    return violation.masked_fill(diagonal, 0.).amax(-1)


def anchor_scores(hands, box, size, inset, scale=10.):
    local = torch.zeros(box.shape[0], 2, 3, device=box.device, dtype=box.dtype)
    local[:, :, 1] = (size[:, 1:2] / 2 - inset) * box.new_tensor([-1., 1.])
    anchors = box[:, None, :3] + quat_rotate(box[:, None, 3:7].expand(-1, 2, -1), local)
    delta = hands.mean(-2)[:, :, None] - anchors[:, None]
    return torch.exp(-scale * delta.square().sum(-1)), anchors


def opposite_gate(scores, threshold):
    close = scores >= threshold
    return (close[:, 0, 0] & close[:, 1, 1]) | (close[:, 0, 1] & close[:, 1, 0])


def apply_joint_geometry(phi, diag, hands, objects, sizes, graph, config):
    scores, anchors = anchor_scores(hands, objects[:, 0], sizes[:, 0],
        config['joint_carry']['anchor_inset'], config['holding']['hand_distance_scale'])
    relation = batched(graph.edge_relation, len(phi))
    owner = batched(graph.edge_owner, len(phi))
    rows = torch.arange(len(phi), device=phi.device)[:, None]
    joint = joint_scenes(graph)
    holding = (relation == HOLDING) & batched(graph.edge_valid, len(phi)) & joint[:, None]
    best, choice = scores.max(-1)
    phi = torch.where(holding, best[rows, owner], phi)
    target = anchors[rows, choice]
    diag['target'] = torch.where(holding[..., None], target[rows, owner], diag['target'])
    distance = (hands.mean(-2) - target).norm(dim=-1)
    diag['distance'] = torch.where(holding, distance[rows, owner], diag['distance'])
    return phi, opposite_gate(scores, config['satisfaction_threshold']) & joint


def shared_alignment_reward(roots, objects, sizes, goals, graph, gate, config):
    n = len(roots)
    row = torch.arange(n, device=roots.device)
    relation = batched(graph.edge_relation, n)
    placement = batched(graph.edge_valid, n) & ((relation == AT) | (relation == ON_TOP))
    edge = placement.long().argmax(-1)
    source = batched(graph.edge_src, n)[row, edge] - graph.num_agents
    target = batched(graph.edge_dst, n)[row, edge]
    ontop = relation[row, edge] == ON_TOP
    support = (target - graph.num_agents).clamp(0, graph.num_objects - 1)
    goal = (target - graph.num_agents - graph.num_objects).clamp(0, graph.num_agents - 1)
    target_xy = torch.where(ontop[:, None], objects[row, support, :2], goals[row, goal, :2])
    delta = target_xy - objects[row, source.clamp(0, graph.num_objects - 1), :2]
    distance = delta.norm(dim=-1)
    leader = (roots[..., :2] - target_xy[:, None]).norm(dim=-1).argmax(-1)
    x, y, z, w = roots[row, leader, 3:7].unbind(-1)
    yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    facing = torch.stack((yaw.cos(), yaw.sin()), -1)
    score = (facing * delta / distance.clamp_min(1e-6)[:, None]).sum(-1).clamp(0., 1.)
    radius = torch.where(ontop, sizes[row, support, :2].norm(dim=-1) / 2, 0.)
    radius = radius + config['buffer']
    saturated = distance <= radius
    score = torch.where(saturated, 1., score)
    joint = joint_scenes(graph)
    reward = score * (gate & joint) * config['weight']
    return reward, dict(score=score, leader=leader, distance=distance,
                        radius=radius, saturated=saturated, joint=joint)


class JointCarryRuntime(Stage1ContextRuntime):
    def __init__(self, n, graph, config, device):
        super().__init__(n, graph, config, device)
        self.coupled_holding = torch.zeros(n, dtype=torch.bool, device=device)

    def current_success(self, phi, z_error, feet_error, graph, region_error, gate):
        success = interaction_own_success(phi, z_error, feet_error, graph, self.config, region_error)
        return success & self.active_edges(phi, graph, gate)

    def active_edges(self, phi, graph, gate):
        holding = batched(graph.edge_relation, len(phi)) == HOLDING
        own_holding = owner_sum((holding & graph.edge_valid &
            (phi >= self.config['satisfaction_threshold'])).float(), graph) > 0
        owner = batched(graph.edge_owner, len(phi))
        placement = torch.where(joint_scenes(graph)[:, None], gate[:, None],
                                own_holding.gather(1, owner))
        return graph.edge_valid & (holding | placement)

    def reset(self, ids, phi, z_error, feet_error=None, region_error=None):
        graph = select_graph(self.graph, ids)
        success = self.current_success(phi, z_error, feet_error, graph, region_error,
                                       self.coupled_holding[ids])
        self.phi[ids] = phi
        self.own_success[ids] = success
        self.achieved[ids] = success
        self.done[ids] = goal_success(success, graph)

    def step(self, phi, progress, z_error, feet_error=None, region_error=None):
        success = self.current_success(phi, z_error, feet_error, self.graph, region_error,
                                       self.coupled_holding)
        active = self.active_edges(phi, self.graph, self.coupled_holding)
        state = phi * active * self.config['state_reward_weight']
        prog = progress * active * self.config['progress_reward_weight']
        bonus = success * self.config['success_reward_weight']
        total = state + prog + bonus
        local = owner_sum(total, self.graph)
        zeros = torch.zeros_like(success)
        result = dict(phi_raw=phi, progress_raw=progress, own_success=success,
            term_success=zeros, reward_saturated=zeros, paired_placement_success=zeros,
            state_component=state, progress_component=prog, success_component=bonus,
            total=total, local_task_reward=local, agent_task_reward=local,
            teammate_task_reward=torch.zeros_like(local))
        self.phi.copy_(phi)
        self.own_success.copy_(success)
        self.achieved.copy_(success)
        self.done.copy_(goal_success(success, self.graph))
        self.last_result = result
        return result
