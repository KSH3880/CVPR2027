"""Warehouse episodes with two BEFORE teams laid out on the apron; the room and its door box are unused."""
import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from utils.before_spec import sample_roles as before_roles
from utils.sensor_stage2_spec import WAREHOUSE

VARIANTS = ('push_before', 'before_pairs')


def allocate(task):
    task._warehouse_before_pairs = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
    task._warehouse_variant_samples = torch.zeros(2, device=task.device)
    # ended episodes, leaders, leader successes, followers, follower successes
    task._warehouse_variant_outcomes = torch.zeros(2, 5, device=task.device)


def sample(task, ids, warehouse):
    choice = task._warehouse_variant_choice
    flag = (warehouse & (torch.rand(len(ids), device=task.device) < task._warehouse_before_probability)
            if choice == 'mixed' else warehouse & (choice == 'before_pairs'))
    task._warehouse_before_pairs[ids] = flag
    task._warehouse_variant_samples += torch.stack(((warehouse & ~flag).sum(), flag.sum())).float()


def sample_roles(sizes, ranges, probabilities):
    n, m = sizes.shape[:2]
    weights = [probabilities[key] for key in ('SIT', 'CLIMB', 'HOLDING_AT', 'HOLDING_ON_TOP')]
    pre, tasks = before_roles(sizes, ranges, 'random_scenario', solo_probabilities=weights, pairing_probability=1.)
    people = torch.arange(m, device=sizes.device)[None].expand(n, -1)
    roles = torch.rand(n, m, device=sizes.device).argsort(-1)
    return torch.where(pre >= 0, pre, people), pre, tasks, roles


def apron_region(task, ids):
    """Room-centred apron rectangle between the front wall's outer face and the apron end."""
    half = task._surface_half[ids, WAREHOUSE, 0]
    thickness, apron = task._warehouse['wall_thickness'], task._warehouse_coop['apron_length']
    center = torch.zeros_like(half)
    center[:, 0] = -half[:, 0]-(apron+thickness)/2
    size = torch.stack((torch.full_like(half[:, 0], (apron-thickness)/2), half[:, 1]+.5), -1)
    return center, size


def place(task, ids):
    k, m = len(ids), task.num_agents
    task._place_failed[ids] = False
    predecessor = task._before_predecessor[ids]
    people = torch.arange(m, device=task.device)[None].expand(k, -1)
    head = torch.where(predecessor >= 0, predecessor, people)
    entities = task._group_entities(ids)
    occupied = scene.Occupancy(k, 5*m, task.device)
    region = apron_region(task, ids)
    for leader in range(m):
        rows = (head[:, leader] == leader).nonzero().flatten()
        if len(rows):
            task._place_group(ids, rows, leader, head, entities, occupied, region=region)
    # Like other warehouse workers these are floor starters: _place_group already raises them to the apron.
    if task._state_init == task.StateInit.Default:
        followers = predecessor >= 0
        task._tar_pos[ids] = torch.where(followers[..., None], task._humanoid_root_states[ids, :, :3],
                                       task._tar_pos[ids])
    _park_unused(task, ids, entities)


def _park_unused(task, ids, entities):
    used = torch.zeros(len(ids), task.num_physical_objects, dtype=torch.bool, device=task.device)
    for column, key in ((1, 'source'), (2, 'support')):
        row, person = entities['valid'][..., column].nonzero(as_tuple=True)
        used[row, entities[key][row, person]] = True
    parked = task._active_physical(ids) & ~used
    row, slot = parked.nonzero(as_tuple=True)
    if not len(row):
        return
    local = task._box_states.new_zeros(len(ids), task.num_physical_objects, 2)
    local[..., 0] = -10.-3*(parked.long().cumsum(-1)-1)-task._surface_half[ids, WAREHOUSE, 0, :1]
    local[..., 1] = 10.
    world = scene.to_world(local, task._structure_yaw[ids, WAREHOUSE], task._structure_shift[ids, WAREHOUSE],
                           task._env_origins[ids])
    e = ids[row]
    task._box_states[e, slot, :2] = world[row, slot]
    task._box_states[e, slot, 2] = task._env_origins[e, 2]+task._box_size[e, slot, 2]/2


def record(task, ending):
    ids = ((task._scene_ids == WAREHOUSE) & ending).nonzero().flatten()
    if not len(ids):
        return
    pre = task._before_predecessor[ids]
    follower = pre >= 0
    leader = torch.zeros_like(follower)
    row, person = follower.nonzero(as_tuple=True)
    leader[row, pre[row, person]] = True
    ever = task._edge_ever_goal[ids]
    counts = torch.stack((leader.sum(-1), (ever & leader).sum(-1), follower.sum(-1), (ever & follower).sum(-1)), -1)
    values = torch.cat((torch.ones(len(ids), 1, device=task.device), counts.float()), -1)
    task._warehouse_variant_outcomes.index_add_(0, task._warehouse_before_pairs[ids].long(), values)


def diagnostics(task, result):
    for index, name in enumerate(VARIANTS):
        row = task._warehouse_variant_outcomes[index]
        prefix = 'warehouse/variant/'+name+'/'
        result[prefix+'sampled'] = task._warehouse_variant_samples[index].clone()
        result[prefix+'episodes'] = row[0].clone()
        result[prefix+'leader_success'] = row[2]/row[1].clamp_min(1)
        result[prefix+'follower_success'] = row[4]/row[3].clamp_min(1)
    task._warehouse_variant_samples.zero_()
    task._warehouse_variant_outcomes.zero_()
    cache = getattr(task, '_warehouse_before_cache', None)
    if cache is not None:
        cache.diagnostics(task, result)
