import torch

from env.tasks.multi_agent.before_reward import BeforeRuntime
from env.tasks.multi_agent.edge_context_reward import goal_success, owner_sum
from utils.edge_ontop_spec import select_graph
from utils.shared_support import rotate
from utils.stage1_push import PUSH


def face_xy_distances(hands, box, size):
    corners = size[..., None, :]*size.new_tensor([[-1,-1,-1], [1,-1,-1], [1,-1,1], [-1,-1,1]])/2
    q = box[..., None, 3:7].expand(*corners.shape[:-1], 4)
    vertices = rotate(q, corners)[..., :2]
    edges = vertices.roll(-1, -2)-vertices
    delta = (hands[..., :2]-box[..., None, :2])[..., None, :]-vertices[..., None, :, :]
    edge = edges[..., None, :, :]
    fraction = (delta*edge).sum(-1)/edge.square().sum(-1).clamp_min(1e-12)
    distance = (delta-fraction.clamp(0, 1)[..., None]*edge).norm(dim=-1).amin(-1)
    cross = edge[..., 0]*delta[..., 1]-edge[..., 1]*delta[..., 0]
    area = (edges[..., 0, 0]*edges[..., 1, 1]-edges[..., 0, 1]*edges[..., 1, 0]).abs()
    inside = ((cross >= 0).all(-1) | (cross <= 0).all(-1)) & (area[..., None] > 1e-10)
    return torch.where(inside, 0., distance)


def contact_geometry(hands, box, size, hand_force, box_force):
    # Both hands on the box's local -Y face (projected to XY) with pushing forces.
    q = box[..., 3:7]
    normal = rotate(q, torch.tensor([0., 1., 0.], device=q.device).expand_as(box[..., :3]))
    local_point = torch.zeros_like(box[..., :3])
    local_point[..., 1] = -size[..., 1]/2
    point = box[..., :3] + rotate(q, local_point)
    distances = face_xy_distances(hands, box, size)
    phi = torch.exp(-10*distances.amax(-1).square())
    hit = ((distances <= .18).all(-1)
           & ((hand_force * -normal[..., None, :]).sum(-1) >= .5).all(-1)
           & ((box_force*normal).sum(-1) >= .5))
    return phi, hit, point, normal


class BridgePushRuntime(BeforeRuntime):
    """BEFORE gate for the crosser; PUSH contact/AT success for each pushing human."""

    def __init__(self, task, original):
        self.__dict__.update(original.__dict__)
        self.task = task

    def _success(self, ids):
        graph = select_graph(self.graph, ids)
        data = self.task._push_state(ids)
        contact = self.task._push_hits[ids]
        at = (data['phi_at'] >= .9) & (data['z_error'].abs() <= .001)
        owner = graph.edge_owner
        mask = self.task._pushing[ids].gather(1, owner) & graph.edge_valid
        success = torch.where(graph.edge_relation == PUSH, contact.gather(1, owner), at.gather(1, owner))
        return mask, success, at

    def reset(self, ids, phi, z_error, feet_error=None, region_error=None):
        super().reset(ids, phi, z_error, feet_error, region_error)
        mask, success, _ = self._success(ids)
        self.own_success[ids] = torch.where(mask, success, self.own_success[ids])
        self.achieved[ids] = self.own_success[ids]
        self.done[ids] = goal_success(self.own_success[ids], select_graph(self.graph, ids))

    def step(self, phi, progress, z_error, feet_error=None, region_error=None):
        result = super().step(phi, progress, z_error, feet_error, region_error)
        graph = self.graph
        mask, success, at = self._success(torch.arange(len(phi), device=phi.device))
        paired = at.gather(1, graph.edge_owner) & mask & (graph.edge_relation == PUSH)
        saturated = success | paired
        weights = self.config
        result['state_component'] = torch.where(mask, weights['state_reward_weight']*torch.where(saturated, 1., phi), result['state_component'])
        result['progress_component'] = torch.where(mask, weights['progress_reward_weight']*torch.where(saturated, 1., progress), result['progress_component'])
        result['success_component'] = torch.where(mask, weights['success_reward_weight']*saturated.float(), result['success_component'])
        result['own_success'] = torch.where(mask, success, result['own_success'])
        result['reward_saturated'] = torch.where(mask, saturated, result['reward_saturated'])
        result['total'] = result['state_component']+result['progress_component']+result['success_component']
        local = owner_sum(result['total'], graph)
        result['local_task_reward'] = result['agent_task_reward'] = local
        self.own_success.copy_(result['own_success'])
        self.achieved |= self.own_success
        self.done.copy_(goal_success(self.own_success, graph))
        return result
