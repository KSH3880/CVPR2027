"""Track pickup and outward doorway crossings for the box-only experiment."""
import torch

from env.tasks.multi_agent.square_sensor_scene import rotate_xy
from env.tasks.multi_agent.warehouse_door_progress import doorway_geometry
from utils.sensor_stage2_spec import WAREHOUSE
from utils.warehouse_waypoint import role_eligibility, role_holding


def crossed_outward(previous, current, door_x, half_width):
    before = previous[..., 0] - door_x
    after = current[..., 0] - door_x
    fraction = before / (before - after).clamp_min(torch.finfo(previous.dtype).eps)
    at_door_y = torch.lerp(previous[..., 1], current[..., 1], fraction.clamp(0, 1))
    return (before >= 0) & (after < 0) & (at_door_y.abs() <= half_width)


class WarehouseCarryEval:
    def __init__(self, task):
        self.ids = torch.arange(task.num_envs, device=task.device)
        self.people, self.eligible = role_eligibility(task._scene_ids, task._warehouse_before_pairs,
                                                      task._before_predecessor, task._templates)
        self.payload = task._agent_box_assignment[self.ids[:, None], self.people]
        self.origin = task._env_origins[:, :2] + task._structure_shift[:, WAREHOUSE]
        self.yaw = task._structure_yaw[:, WAREHOUSE]
        door_x, half_width = doorway_geometry(task, self.ids)
        self.door_x = door_x[:, None, None] - task._warehouse['wall_thickness']/2
        self.half_width = half_width[:, None, None]
        self.previous = self.positions(task)
        self.pickup_step = torch.full_like(self.people, -1)
        self.crossing_step = self.pickup_step[..., None].expand(-1, -1, 2).clone()

    def positions(self, task):
        rows = self.ids[:, None]
        roots = task._humanoid_root_states[rows, self.people, :2]
        boxes = task._box_states[rows, self.payload, :2]
        xy = torch.stack((roots, boxes), dim=2) - self.origin[:, None, None]
        return rotate_xy(xy, -self.yaw[:, None, None])

    def record(self, task, alive, step):
        valid = self.eligible & alive[self.ids[:, None], self.people]
        holding = role_holding(task.relation_runtime, self.ids, self.people)
        first_pickup = valid & holding & (self.pickup_step < 0)
        self.pickup_step[first_pickup] = step
        current = self.positions(task)
        crossed = crossed_outward(self.previous, current, self.door_x, self.half_width)
        tracked = valid & (self.pickup_step >= 0)
        self.crossing_step[crossed & tracked[..., None] & (self.crossing_step < 0)] = step
        self.previous = current

    def finish(self, scored, ever):
        rows = self.ids[:, None]
        eligible = self.eligible & scored[rows, self.people]
        goal_success = ever[rows, self.people]
        both = (self.crossing_step >= 0).all(-1)
        exited = self.crossing_step.amax(-1).masked_fill(~both, -1)
        totals = {}
        for index, role in enumerate(('leader', 'follower')):
            mask = eligible[:, index]
            totals[role] = dict(
                eligible=int(mask.sum()),
                picked_up=int((self.pickup_step[:, index][mask] >= 0).sum()),
                exited=int((exited[:, index][mask] >= 0).sum()),
                point_goal_success=int(goal_success[:, index, 0][mask].sum()),
                region_goal_success=int(goal_success[:, index, 1][mask].sum()),
            )
        trace = dict(warehouse_role_ids=self.people.masked_fill(~eligible, -1).cpu().numpy(),
                     warehouse_pickup_step=self.pickup_step.masked_fill(~eligible, -1).cpu().numpy(),
                     warehouse_root_exit_step=self.crossing_step[..., 0].masked_fill(~eligible, -1).cpu().numpy(),
                     warehouse_payload_exit_step=self.crossing_step[..., 1].masked_fill(~eligible, -1).cpu().numpy(),
                     warehouse_exit_step=exited.masked_fill(~eligible, -1).cpu().numpy())
        return totals, trace
