import math

import torch

from utils.coupled_push_spec import inverse_rotate
from utils.jointpush_geometry import rotate


def set_pusher_gaps(task, ids, gaps):
    rows, people = task._bridge_pushers[ids].nonzero(as_tuple=True)
    env = ids[rows]
    box = task._box_states[env, task._bridge_slot[env]]
    width = task._box_size[env, task._bridge_slot[env], 1]
    yaw = task._structure_yaw[env, 1]
    away = torch.stack((-yaw.sin(), yaw.cos()), -1)
    away *= torch.where(task._bridge_side[env], 1., -1.)[:, None]
    relative = task._humanoid_root_states[env, people, :2]-box[:, :2]
    gap = torch.tensor(gaps, device=task.device).repeat(len(ids))
    delta = (width/2+gap-(relative*away).sum(-1))[:, None]*away
    task._humanoid_root_states[env, people, :2] += delta
    task._kinematic_humanoid_rigid_body_states[env, people, :, :2] += delta[:, None]


def rear_points(sizes, draws, spec):
    angle = (draws[..., 0]-.5)*math.radians(spec['angle_degrees'])
    distance = spec['surface_distance'][0]+draws[..., 1]*(spec['surface_distance'][1]-spec['surface_distance'][0])
    direction = torch.stack((angle.sin(), -angle.cos()), -1)
    half = sizes[:, None, :2]/2
    absolute = direction.abs()
    radius = ((half+distance[..., None])/absolute.clamp_min(1e-8)).amin(-1)
    corner = (radius[..., None]*absolute > half).all(-1)
    projection = (absolute*half).sum(-1)
    corner_radius = projection+(projection.square()-half.square().sum(-1)+distance.square()).clamp_min(0).sqrt()
    radius = torch.where(corner, corner_radius, radius)
    return radius[..., None]*direction


def rectangle_distance(points, centers, quats, sizes):
    delta = points[:, :, None]-centers[:, None, :, :2]
    local = inverse_rotate(quats[:, None].expand(-1, points.shape[1], -1, -1),
                           torch.cat((delta, torch.zeros_like(delta[..., :1])), -1))
    return (local[..., :2].abs()-sizes[:, None, :, :2]/2).clamp_min(0).norm(dim=-1)


def sample_pushers(task, ids, spec):
    placed = ~task._bridge_pushers[ids].clone()
    rounds = torch.zeros(len(ids), device=task.device, dtype=torch.long)
    for person in range(task.num_agents):
        pending = task._bridge_pushers[ids, person].nonzero().flatten()
        for _ in range(spec['candidate_rounds']):
            if not len(pending):
                break
            es = ids[pending]
            rounds[pending] += 1
            box = task._box_states[es, task._bridge_slot[es]]
            size = task._box_size[es, task._bridge_slot[es]]
            draws = torch.rand(len(es), spec['candidate_batch'], 2, device=task.device)
            local = rear_points(size, draws, spec)
            offset = torch.cat((local, torch.zeros_like(local[..., :1])), -1)
            xy = rotate(box[:, None, 3:7].expand(-1, spec['candidate_batch'], -1), offset)[..., :2]+box[:, None, :2]
            centers, quats, half = task._static_boxes(es)
            centers = torch.cat((centers, task._box_states[es, :, :3]), 1)
            quats = torch.cat((quats, task._box_states[es, :, 3:7]), 1)
            sizes = torch.cat((2*half, task._box_size[es]), 1)
            active = centers[..., 2]+sizes[..., 2]/2 > task._env_origins[es, None, 2]+.05
            clearance = rectangle_distance(xy, centers, quats, sizes)
            valid = ((clearance >= spec['obstacle_clearance']) | ~active[:, None]).all(-1)
            people = task._humanoid_root_states[es, :, :2]
            separation = (xy[:, :, None]-people[:, None]).norm(dim=-1)
            valid &= ((separation >= spec['human_separation']) | ~placed[pending, None]).all(-1)
            accepted = valid.any(-1)
            choice = valid.long().argmax(-1)
            accepted_rows = pending[accepted]
            env = ids[accepted_rows]
            selected = xy[torch.arange(len(es), device=task.device), choice][accepted]
            task._place_person(env, person, selected,
                task._heading_to(selected, box[accepted, :2]), torch.zeros(len(env), device=task.device))
            placed[accepted_rows, person] = True
            pending = pending[~accepted]
        task._place_failed[ids[pending]] = True
    return rounds
