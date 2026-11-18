"""Training-only Warehouse resets that start after the AT leader's delivery.

The door starts open, the leader's box rests on its apron goal and the leader stands beside it; pushers, the follower
and every other object keep the regular fixed-role layout.
"""
import math

import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from utils.sensor_stage2_spec import WAREHOUSE

MODES = ('main', 'leader_done')
CANDIDATES = 64
FACE_GAP = (.45, 1.)


def allocate(task):
    task._warehouse_leader_done = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
    task._warehouse_done_samples = torch.zeros(2, device=task.device)
    # ended episodes, leaders, leader successes, followers, follower successes
    task._warehouse_done_outcomes = torch.zeros(2, 5, device=task.device)


def sample(task, ids, warehouse):
    flag = warehouse & (torch.rand(len(ids), device=task.device) < task._warehouse_done_probability)
    task._warehouse_leader_done[ids] = flag
    task._warehouse_done_samples += torch.stack(((warehouse & ~flag).sum(), flag.sum())).float()


def place(task, ids):
    """After warehouse_sensor.place: move each flagged leader's box onto its goal and the leader next to it."""
    ids = ids[task._warehouse_leader_done[ids] & ~task._place_failed[ids]]
    if not len(ids):
        return
    pre = task._before_predecessor[ids]
    row, follower = (pre >= 0).nonzero(as_tuple=True)
    env, leader = ids[row], pre[row, follower]
    # The leader's route is already finished, so the near-pair route ordering no longer applies.
    task._before_near_pairs[env] = False
    slot = task._agent_box_assignment[env, leader]
    task._box_states[env, slot, :3] = task._tar_pos[env, leader]
    task._box_states[env, slot, 7:] = 0
    xy, good = _leader_spot(task, env, leader, slot)
    task._place_failed[env[~good]] = True
    box = task._box_states[env, slot, :2]
    heading = torch.atan2(box[:, 1]-xy[:, 1], box[:, 0]-xy[:, 0])
    for person in range(task.num_agents):
        chosen = good & (leader == person)
        if chosen.any():
            # Already lifted onto the apron by the first placement.
            task._place_person(env[chosen], person, xy[chosen], heading[chosen], torch.zeros(int(chosen.sum()), device=task.device))


def _leader_spot(task, env, leader, slot):
    k = len(env)
    yaw, shift, origin = task._structure_yaw[env, WAREHOUSE], task._structure_shift[env, WAREHOUSE], task._env_origins[env]
    to_local = lambda xy: scene.rotate_xy(xy-origin[:, None, :2]-shift[:, None], -yaw[:, None])
    center = to_local(task._box_states[env, slot, None, :2])
    half = task._box_size[env, slot, None, :2]/2
    angle = torch.rand(k, CANDIDATES, device=task.device)*2*math.pi
    reach = half.norm(dim=-1)+FACE_GAP[0]+torch.rand(k, CANDIDATES, device=task.device)*(FACE_GAP[1]-FACE_GAP[0]+.3)
    points = center+reach[..., None]*torch.stack((angle.cos(), angle.sin()), -1)
    gap = ((points-center).abs()-half).clamp_min(0).norm(dim=-1)
    good = (gap >= FACE_GAP[0]) & (gap <= FACE_GAP[1])
    # Stay on the apron, clear of the front wall's outer face and the apron edges.
    room = task._surface_half[env, WAREHOUSE, 0]
    front = -room[:, :1]-task._warehouse['wall_thickness']-.4
    end = -room[:, :1]-task._warehouse_coop['apron_length']+.4
    good &= (points[..., 0] <= front) & (points[..., 0] >= end) & (points[..., 1].abs() <= room[:, 1:]+.1)
    others = to_local(task._humanoid_root_states[env, :, :2])
    mine = torch.arange(task.num_agents, device=task.device)[None] == leader[:, None]
    near_person = (points[:, :, None]-others[:, None]).norm(dim=-1) < .85
    good &= ~(near_person & ~mine[:, None]).any(-1)
    boxes = to_local(task._box_states[env, :, :2])
    radius = task._box_size[env, :, :2].norm(dim=-1)/2
    own = torch.arange(task.num_physical_objects, device=task.device)[None] == slot[:, None]
    blocked = (points[:, :, None]-boxes[:, None]).norm(dim=-1) < radius[:, None]+.5
    good &= ~(blocked & (task._active_physical(env) & ~own)[:, None]).any(-1)
    pick = good.long().argmax(-1)
    local = points[torch.arange(k, device=task.device), pick]
    return scene.to_world(local[:, None], yaw, shift, origin)[:, 0], good.any(-1)


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
    task._warehouse_done_outcomes.index_add_(0, task._warehouse_leader_done[ids].long(), values)


def diagnostics(task, result):
    for index, name in enumerate(MODES):
        row = task._warehouse_done_outcomes[index]
        prefix = 'warehouse/'+name+'/'
        result[prefix+'sampled'] = task._warehouse_done_samples[index].clone()
        result[prefix+'episodes'] = row[0].clone()
        result[prefix+'leader_success'] = row[2]/row[1].clamp_min(1)
        result[prefix+'follower_success'] = row[4]/row[3].clamp_min(1)
    task._warehouse_done_samples.zero_()
    task._warehouse_done_outcomes.zero_()
