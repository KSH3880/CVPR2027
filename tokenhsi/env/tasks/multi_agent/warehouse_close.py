import torch

from env.tasks.multi_agent.square_sensor import rotation_matrix
from utils.edge_context_spec import HOLDING
from utils.edge_ontop_spec import batched
from utils.sensor_stage2_spec import PUSH, WAREHOUSE
from utils.wall_proximity import box_box_gap, capsule_wall_gap, load_capsules, proximity


def allocate(task):
    from env.tasks.multi_agent.sensor_stage2_observation import WALL_CONTACT_BODIES
    task._wall_proximity_shapes = load_capsules(task.cfg['env']['asset'], WALL_CONTACT_BODIES, task.device)
    task._wall_proximity_sums = torch.zeros(6, device=task.device)


def held_people(task, ids):
    graph = task.relation_runtime.graph
    valid = batched(graph.edge_valid, task.num_envs)[ids]
    relation = batched(graph.edge_relation, task.num_envs)[ids]
    owner = batched(graph.edge_owner, task.num_envs)[ids]
    holding = valid & (relation == HOLDING) & task.relation_runtime.own_success[ids]
    people = torch.arange(task.num_agents, device=task.device)
    return (holding[..., None] & (owner[..., None] == people)).any(1)


def risks(task, ids):
    option = task._warehouse['wall_proximity']
    centers = task._static_centers[ids, WAREHOUSE, 1:]
    halves = task._static_half[ids, WAREHOUSE, 1:]
    rotations = rotation_matrix(task._static_quat[ids, WAREHOUSE, 1:]).reshape(len(ids), 5, 3, 3)
    bodies = task._wall_contact_body_ids
    positions = task._rigid_body_pos[ids][:, :, bodies]
    body_rot = rotation_matrix(task._rigid_body_rot[ids][:, :, bodies]).reshape(*positions.shape[:-1], 3, 3)
    human = proximity(capsule_wall_gap(positions, body_rot, task._wall_proximity_shapes,
                                      centers, rotations, halves), option['distance'])
    if option.get('push_box', False):
        obstacle = task._box_states[ids, task._bridge_slot[ids]]
        obstacle_half = task._box_size[ids, task._bridge_slot[ids]] / 2
        obstacle_rot = rotation_matrix(obstacle[:, 3:7])
        pushers = task._templates[ids] == PUSH
        others = ~pushers & pushers.any(-1, keepdim=True)
        gap = capsule_wall_gap(positions, body_rot, task._wall_proximity_shapes,
                               obstacle[:, None, :3], obstacle_rot[:, None], obstacle_half[:, None])
        human = torch.maximum(human, proximity(gap, option['distance']) * others)
    payload = torch.zeros_like(human)
    if option['payload']:
        row, person = held_people(task, ids).nonzero(as_tuple=True)
        if len(row):
            env = ids[row]
            slot = task._agent_box_assignment[env, person]
            boxes = task._box_states[env, slot]
            sizes = task._box_size[env, slot] / 2
            box_rot = rotation_matrix(boxes[:, 3:7])
            gap = torch.stack([box_box_gap(boxes[:, :3], box_rot, sizes,
                centers[row, wall], rotations[row, wall], halves[row, wall]) for wall in range(5)], -1).amin(-1)
            if option.get('push_box', False):
                obstacle_gap = box_box_gap(boxes[:, :3], box_rot, sizes,
                    obstacle[row, :3], obstacle_rot[row], obstacle_half[row])
                gap = torch.where(others[row, person], torch.minimum(gap, obstacle_gap), gap)
            payload[row, person] = proximity(gap, option['distance'])
    return human, payload


def combine(task, ids, existing):
    human, payload = risks(task, ids)
    coefficient = task._warehouse['wall_proximity']['coefficient']
    penalty = coefficient * torch.maximum(human, payload) / task._agent_collision_coeff
    combined = torch.maximum(existing, penalty)
    task._wall_proximity_sums += torch.stack(((human > 0).sum(), (payload > 0).sum(),
        human.sum() * coefficient, payload.sum() * coefficient,
        (combined - existing).sum() * task._agent_collision_coeff, human.new_tensor(human.numel())))
    return combined


def diagnostics(task, result):
    values = task._wall_proximity_sums
    for index, name in enumerate(('human_fraction', 'payload_fraction', 'human_penalty', 'payload_penalty', 'added_penalty')):
        result['warehouse/proximity/' + name] = values[index] / values[-1].clamp_min(1)
    values.zero_()
