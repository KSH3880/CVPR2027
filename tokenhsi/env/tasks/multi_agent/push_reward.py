import torch
from isaacgym.torch_utils import quat_rotate, quat_conjugate

from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from env.tasks.multi_agent.edge_context_reward import goal_success, owner_sum
from utils.stage1_push import PUSH
from utils.edge_ontop_spec import select_graph


def face_xy_distances(hands, box, size):
    corners = size[..., None, :]*size.new_tensor([[-1,-1,-1], [1,-1,-1], [1,-1,1], [-1,-1,1]])/2
    q = box[..., None, 3:7].expand(*corners.shape[:-1], 4)
    vertices = quat_rotate(q.reshape(-1, 4), corners.reshape(-1, 3)).reshape_as(corners)[..., :2]
    edges = vertices.roll(-1, -2)-vertices
    delta = (hands[..., :2]-box[..., None, :2])[..., None, :]-vertices[..., None, :, :]
    edge = edges[..., None, :, :]
    fraction = (delta*edge).sum(-1)/edge.square().sum(-1).clamp_min(1e-12)
    distance = (delta-fraction.clamp(0, 1)[..., None]*edge).norm(dim=-1).amin(-1)
    cross = edge[..., 0]*delta[..., 1]-edge[..., 1]*delta[..., 0]
    area = (edges[..., 0, 0]*edges[..., 1, 1]-edges[..., 0, 1]*edges[..., 1, 0]).abs()
    inside = ((cross >= 0).all(-1) | (cross <= 0).all(-1)) & (area[..., None] > 1e-10)
    return torch.where(inside, 0., distance)


def contact_geometry(hands, box, size, hand_force, box_force, midpoint_only=False, contact_height=.6175, face_xy=False):
    q = box[..., 3:7]
    normal = quat_rotate(q.reshape(-1, 4), torch.tensor([0., 1., 0.], device=q.device).expand(q.numel()//4, -1)).reshape_as(box[..., :3])
    local_point = torch.zeros_like(box[..., :3])
    local_point[..., 1] = -size[..., 1]/2
    local_point[..., 2] = 0 if face_xy else contact_height - size[..., 2]/2
    point = box[..., :3] + quat_rotate(q.reshape(-1, 4), local_point.reshape(-1, 3)).reshape_as(local_point)
    if face_xy:
        distances = face_xy_distances(hands, box, size)
        phi = torch.exp(-10*distances.amax(-1).square())
        hit = ((distances <= .18).all(-1)
               & ((hand_force * -normal[..., None, :]).sum(-1) >= .5).all(-1)
               & ((box_force*normal).sum(-1) >= .5))
        return phi, hit, point, normal
    local_hands = quat_rotate(quat_conjugate(q)[..., None, :].expand(*hands.shape[:-1], 4).reshape(-1, 4),
                              (hands-box[..., None, :3]).reshape(-1, 3)).reshape_as(hands)
    excess_x = (local_hands[..., 0].abs()-size[..., None, 0]/2).clamp_min(0)
    excess_z = (local_hands[..., 2].abs()-size[..., None, 2]/2).clamp_min(0)
    face_distances = (excess_x.square()+excess_z.square()+(local_hands[..., 1]+size[..., None, 1]/2).square()).sqrt()
    delta = hands.mean(-2)-point
    error_xy = delta[..., :2].norm(dim=-1)
    if not midpoint_only:
        error_xy = torch.maximum(error_xy, face_distances.amax(-1))
    error_z = (delta[..., 2].abs()-.0325).clamp_min(0)
    phi = torch.exp(-10*error_xy.square()-(error_z/.04).square())
    hit = ((face_distances <= .18).all(-1) & (delta[..., :2].norm(dim=-1) <= .18)
           & (delta[..., 2].abs() <= .0325+1e-6)
           & ((hand_force * -normal[..., None, :]).sum(-1) >= .5).all(-1)
           & ((box_force*normal).sum(-1) >= .5))
    return phi, hit, point, normal


class PushRuntime(Stage1ContextRuntime):
    def __init__(self, task, original):
        self.__dict__.update(original.__dict__)
        self.task = task

    def _success(self, ids, phi):
        graph = select_graph(self.graph, ids)
        data = self.task._push_state(ids)
        contact = self.task._push_hits[ids]
        radius = self.task._push_options.get('success_radius')
        at = data['phi_at'] >= .9 if radius is None else data['distance_xy'] <= radius
        at &= data['z_error'].abs() <= .001
        owner = graph.edge_owner
        push = self.task._push_env[ids, None] & graph.edge_valid
        success = torch.where(graph.edge_relation == PUSH, contact.gather(1, owner), at.gather(1, owner))
        return push, success, at, data

    def reset(self, ids, phi, z_error, feet_error=None, region_error=None):
        super().reset(ids, phi, z_error, feet_error, region_error)
        push, success, _, _ = self._success(ids, phi)
        self.own_success[ids] = torch.where(push, success, self.own_success[ids])
        self.achieved[ids] = self.own_success[ids]
        self.done[ids] = goal_success(self.own_success[ids], select_graph(self.graph, ids))

    def step(self, phi, progress, z_error, feet_error=None, region_error=None):
        result = super().step(phi, progress, z_error, feet_error, region_error)
        ids = torch.arange(len(phi), device=phi.device)
        mask, success, at, _ = self._success(ids, phi)
        graph = self.graph
        paired = at.gather(1, graph.edge_owner) & mask & (graph.edge_relation == PUSH)
        saturated = success | paired
        for key, raw in [('state_component', phi), ('progress_component', progress)]:
            result[key] = torch.where(mask, .2*torch.where(saturated, 1., raw), result[key])
        result['success_component'] = torch.where(mask, .2*saturated.float(), result['success_component'])
        result['own_success'] = torch.where(mask, success, result['own_success'])
        result['reward_saturated'] = torch.where(mask, saturated, result['reward_saturated'])
        result['total'] = result['state_component']+result['progress_component']+result['success_component']
        local = owner_sum(result['total'], graph)
        other = (local.sum(-1, keepdim=True)-local)/(graph.num_agents-1)
        result['local_task_reward'] = local
        result['agent_task_reward'] = torch.where(self.task._push_env[:, None], .9*local+.1*other, local)
        result['teammate_task_reward'] = torch.where(self.task._push_env[:, None], other, 0.)
        self.own_success.copy_(result['own_success'])
        self.achieved |= self.own_success
        self.done.copy_(goal_success(self.own_success, graph))
        return result
