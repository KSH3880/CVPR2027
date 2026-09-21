"""SIT/CLIMB geometry and current-success rules on the shared edge runtime."""
import torch

from utils.edge_context_spec import HOLDING
from utils.edge_ontop_spec import batched, graph_packet, select_graph
from utils.edge_interaction_spec import SIT, CLIMB
from env.tasks.multi_agent.edge_context_reward import edge_context, edge_context_reward, goal_success
from env.tasks.multi_agent.edge_ontop_reward import (OnTopContextRuntime, evaluate_ontop_edges,
    mix_task_reward, vertical_extent)


def quat_rotate(q, v):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    xyz = q[..., :3]
    uv = torch.cross(xyz, v, dim=-1)
    return v + 2 * (q[..., 3:4] * uv + torch.cross(xyz, uv, dim=-1))


def evaluate_interaction_edges(hands, feet, roots, objects, sizes, goals, graph, config, char_h):
    phi, diag = evaluate_ontop_edges(hands, roots, objects, sizes, goals, graph, config)
    n, m, o = roots.shape[0], graph.num_agents, graph.num_objects
    src = batched(graph.edge_src, n)
    dst = batched(graph.edge_dst, n)
    relation = batched(graph.edge_relation, n)
    valid = batched(graph.edge_valid, n)
    batch = torch.arange(n, device=roots.device)[:, None]
    human = roots[batch, src.clamp(0, m - 1)]
    support = objects[batch, (dst - m).clamp(0, o - 1)]
    support_size = sizes[batch, (dst - m).clamp(0, o - 1)]

    sit = (relation == SIT) & valid
    climb = (relation == CLIMB) & valid
    top_z = support[..., 2] + vertical_extent(support[..., 3:7], support_size / 2)
    if config['sit']['state_definition'] == 'box_top_plus_pelvis_clearance':
        sit_target = support[..., :3].clone()
        sit_target[..., 2] = top_z + float(config['sit']['pelvis_clearance'])
    else:
        local = torch.tensor(config['sit']['target_local_offset'], device=roots.device,
                             dtype=roots.dtype).view(1, 1, 3).expand_as(human)
        sit_target = support[..., :3] + quat_rotate(support[..., 3:7], local)
    climb_target = support[..., :3].clone()
    climb_target[..., 2] = top_z + float(char_h)
    target = torch.where(sit[..., None], sit_target, climb_target)
    delta = human - target
    scale = torch.where(sit, float(config['sit']['near_distance_scale']),
                        float(config['climb']['near_distance_scale']))
    interaction_phi = torch.exp(-scale * delta.square().sum(-1))

    # SIT approaches the transformed tarSitPos; CLIMB approaches the object XY.
    progress_target_xy = torch.where(sit[..., None], sit_target[..., :2], support[..., :2])
    distance_xy = (human[..., :2] - progress_target_xy).norm(dim=-1)
    p = config['progress']
    progress = 1 / (1 + (distance_xy - p['delta']).clamp_min(0) / p['sigma'])
    mask = sit | climb
    phi = torch.where(mask, interaction_phi, phi)
    diag['progress'] = torch.where(mask, progress, diag['progress'])
    diag['z_error'] = torch.where(mask, delta[..., 2], diag['z_error'])
    diag['distance'] = torch.where(mask, delta.norm(dim=-1), diag['distance'])
    diag['distance_xy'] = torch.where(mask, distance_xy, diag['distance_xy'])
    diag['target'] = torch.where(mask[..., None], target, diag['target'])

    feet_z = feet.mean(-2)[..., 2]
    feet_z = feet_z[batch, src.clamp(0, m - 1)]
    diag['z_feet'] = feet_z * climb
    diag['z_surface'] = top_z * climb
    diag['feet_height_error'] = (feet_z - top_z).abs() * climb
    return phi, diag


def interaction_own_success(phi, z_error, feet_error, graph, config):
    relation = batched(graph.edge_relation, phi.shape[0])
    valid = batched(graph.edge_valid, phi.shape[0])
    threshold = config['satisfaction_threshold']
    success = (phi >= threshold) & ((relation == HOLDING) |
        (z_error.abs() <= config['success']['at_z_tolerance']))
    success = torch.where(relation == SIT, phi >= threshold, success)
    success = torch.where(relation == CLIMB,
        (phi >= threshold) & (feet_error <= config['climb']['feet_height_tolerance']), success)
    return success & valid


class InteractionContextRuntime(OnTopContextRuntime):
    def reset(self, ids, phi, z_error, feet_error=None):
        if feet_error is None:
            feet_error = torch.zeros_like(phi)
        graph = select_graph(self.graph, ids)
        self.phi[ids] = phi
        self.own_success[ids] = interaction_own_success(phi, z_error, feet_error, graph, self.config)
        self.achieved[ids] = self.own_success[ids]
        self.done[ids] = goal_success(self.own_success[ids], graph)

    def step(self, phi, progress, z_error, feet_error=None):
        if feet_error is None:
            feet_error = torch.zeros_like(phi)
        success = interaction_own_success(phi, z_error, feet_error, self.graph, self.config)
        result = edge_context_reward(phi, progress, success, self.graph, self.config)
        result['local_task_reward'] = result['agent_task_reward']
        result['agent_task_reward'] = mix_task_reward(result['local_task_reward'])
        self.phi.copy_(phi)
        self.own_success.copy_(success)
        self.achieved |= success
        self.done.copy_(goal_success(success, self.graph))
        self.last_result = result
        return result

    def suffix(self, ids=None):
        graph = self.graph if ids is None else select_graph(self.graph, ids)
        phi = self.phi if ids is None else self.phi[ids]
        return graph_packet(graph, edge_context(phi, graph))
