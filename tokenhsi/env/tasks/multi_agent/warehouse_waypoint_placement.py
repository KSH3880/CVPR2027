"""Unoptimized viewer placement, retained only for reset performance comparisons."""
import torch
from env.tasks.multi_agent import square_sensor_scene as scene
from utils.warehouse_waypoint import worker_spawns
from utils.before_spec import task_objects
from utils.sensor_stage2_spec import WAREHOUSE


def finish_legacy_placement(self, ids):
    ids = ids[~self._place_failed[ids]]
    if not len(ids):
        return
    rows = torch.arange(len(ids), device=self.device)
    people = torch.arange(self.num_agents, device=self.device)
    pre = self._before_predecessor[ids]
    follower = (pre >= 0).long().argmax(-1)
    leader = pre.gather(1, follower[:, None]).squeeze(1)
    yaw = self._structure_yaw[ids, WAREHOUSE]
    shift = self._structure_shift[ids, WAREHOUSE]
    origin = self._env_origins[ids]
    half_x = self._surface_half[ids, WAREHOUSE, 0, 0]

    def local(xy):
        result = scene.rotate_xy(xy-origin[:, None, :2]-shift[:, None], -yaw[:, None])
        result[..., 0] += half_x[:, None]
        return result

    primary = self._agent_box_assignment[ids]
    slots = torch.cat((primary, (3*self.num_agents+people).expand(len(ids), -1),
                       self._bridge_slot[ids, None]), -1)
    objects = local(self._box_states[ids[:, None], slots, :2])
    sizes = self._box_size[ids[:, None], slots, :2].clone()
    sizes[:, -1] = sizes[:, -1, [1, 0]]
    valid = task_objects(self._before_predecessor[ids], self._templates[ids])
    valid[:, :self.num_agents] &= self._templates[ids] != 5
    valid = torch.cat((valid, torch.ones(len(ids), 1, device=self.device, dtype=torch.bool)), -1)
    roots = local(self._humanoid_root_states[ids, :, :2])
    goals = local(self._tar_pos[ids, :, :2])
    roots, good = worker_spawns(roots, objects, sizes, valid, leader, follower,
        objects[:, :self.num_agents], goals[rows, leader], self._templates[ids, follower],
        self._before_near_pairs[ids, follower])
    source = objects[rows, leader]
    extent = sizes[rows, leader]/2
    half_y = self._surface_half[ids, WAREHOUSE, 0, 1]
    wall_gap = torch.stack((source[:, 0]-extent[:, 0],
        2*half_x-source[:, 0]-extent[:, 0], half_y-source[:, 1].abs()-extent[:, 1]), -1)
    good &= (wall_gap >= 1.2).all(-1)
    self._place_failed[ids] |= ~good
    roots[..., 0] -= half_x[:, None]
    world = scene.to_world(roots, yaw, shift, origin)
    for person in (leader, follower):
        target = torch.where((self._templates[ids, person] < 3), leader, person)
        box_xy = self._box_states[ids, primary[rows, target], :2]
        direction = box_xy-world[rows, person]
        heading = torch.atan2(direction[:, 1], direction[:, 0])
        for index in range(self.num_agents):
            chosen = good & (person == index)
            self._place_person(ids[chosen], index, world[chosen, index], heading[chosen],
                               torch.zeros_like(heading[chosen]))
