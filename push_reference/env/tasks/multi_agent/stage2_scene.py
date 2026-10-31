import torch
from push_reference.utils.grasp_geometry import grasp_y

from push_reference.utils.task_role_spec import task_graph_packet


def joint_holding_geometry(task, phi, diag, ids):
    live = ids is None
    if ids is None:
        ids = torch.arange(task.num_envs, device=task.device)
    graph = task.relation_runtime.graph
    packet = task_graph_packet(graph, task.num_envs).reshape(-1, 4, 5)[ids].long()
    hands = task._kinematic_humanoid_rigid_body_states[ids][:, :, task._key_body_ids[:2], :3]
    if live:
        hands = task._rigid_body_pos[ids][:, :, task._key_body_ids[:2]]
    boxes = task._logical_box_values(task._box_states, ids)
    sizes = task._logical_box_values(task._box_size, ids)
    row = torch.arange(len(ids), device=task.device)[:, None]
    payload = (packet[..., 3]-4).clamp_min(0)
    box, size = boxes[row, payload], sizes[row, payload]
    local = torch.zeros(len(ids), 4, 3, device=task.device)
    local[..., 1] = grasp_y(size[..., 1]/2, task._grasp_sides[ids], task.cfg['env']['stage2']['anchor_inset'])
    q = box[..., 3:7]
    cross = 2*torch.cross(q[..., :3], local, dim=-1)
    target = box[..., :3]+local+q[..., 3:]*cross+torch.cross(q[..., :3], cross, dim=-1)
    delta = hands.mean(-2)-target
    score = torch.exp(-10*delta.square().sum(-1))
    owner = graph.edge_owner[ids]
    mask = (graph.edge_relation[ids] == 6) & graph.edge_valid[ids] & task._joint[ids].gather(1, owner)
    phi = torch.where(mask, score.gather(1, owner), phi)
    diag['target'] = torch.where(mask[..., None], target.gather(1, owner[..., None].expand(-1, -1, 3)), diag['target'])
    distance = (hands.mean(-2)-target).norm(dim=-1)
    diag['distance'] = torch.where(mask, distance.gather(1, owner), diag['distance'])
    return phi, diag


def scene_infeasible(task, ids):
    boxes = task._logical_box_values(task._box_states, ids)
    sizes = task._logical_box_values(task._box_size, ids)
    radius = sizes[..., :2].norm(dim=-1)/2
    delta = boxes[:, :, None]-boxes[:, None]
    overlap = (delta[..., :2].norm(dim=-1) < radius[:, :, None]+radius[:, None]+.05) & (
        delta[..., 2].abs() < (sizes[:, :, None, 2]+sizes[:, None, :, 2])/2-.001)
    eye = torch.eye(8, device=task.device, dtype=torch.bool)
    bad = overlap.masked_fill(eye, False).flatten(1).any(-1)
    bodies = task._kinematic_humanoid_rigid_body_states[ids, ..., :3]
    relative = bodies[:, :, None]-boxes[:, None, :, None, :3]
    q = boxes[:, None, :, None, 3:7].expand(*relative.shape[:-1], 4)
    cross = 2*torch.cross(q[..., :3], relative, dim=-1)
    local = relative-q[..., 3:]*cross+torch.cross(q[..., :3], cross, dim=-1)
    inside = (local.abs() < sizes[:, None, :, None]/2-.06).all(-1).any(-1)
    packet = task_graph_packet(task.relation_runtime.graph, task.num_envs).reshape(-1, 4, 5)[ids].long()
    owned = (packet[..., 3, None] == torch.arange(8, device=task.device)[None, None]+4) & (packet[..., 1, None] >= 2)
    owned |= (packet[..., 4, None] == torch.arange(8, device=task.device)[None, None]+4) & (packet[..., 1, None] <= 1)
    bad |= (inside & ~owned).flatten(1).any(-1)
    graph = task.relation_runtime.graph
    source = (graph.edge_src[ids]-4).clamp(0, 7)
    target = (graph.edge_dst[ids]-12).clamp(0, 3)
    at = (graph.edge_relation[ids] == 7) & graph.edge_valid[ids]
    row = torch.arange(len(ids), device=task.device)[:, None]
    goals = task._tar_pos[ids][row, target]
    distance = (goals[:, :, None, :2]-boxes[:, None, :, :2]).norm(dim=-1)
    other = torch.arange(8, device=task.device)[None, None] != source[..., None]
    threshold = radius.gather(1, source)[..., None]+radius[:, None]+.25
    bad |= (at[..., None] & other & (distance < threshold)).flatten(1).any(-1)
    return bad | ~torch.isfinite(bodies).flatten(1).all(-1)


def cpa_collision_penalty(root_pos, root_vel, min_distance, discount, dt, released=None):
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
    if released is not None:
        violation = violation.masked_fill(released, 0.)
    return violation.masked_fill(diagonal, 0.).amax(-1)
