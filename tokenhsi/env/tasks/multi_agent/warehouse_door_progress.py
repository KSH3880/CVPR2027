import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from utils.sensor_stage2_spec import WAREHOUSE, PUSH


def doorway_geometry(task, ids):
    half = task._surface_half[ids, WAREHOUSE, 0]
    thickness = task._warehouse['wall_thickness']
    return -half[:, 0]-thickness/2, half[:, 1]+thickness-2*task._static_half[ids, WAREHOUSE, 4, 1]


def reward(task):
    n, m = task.num_envs, task.num_agents
    progress = task._humanoid_root_states.new_zeros(n, m)
    task.extras['warehouse_door_progress'] = progress
    chosen = (task._scene_ids == WAREHOUSE) & ((task._templates == PUSH).sum(-1) == 2)
    if task._warehouse_before_probability:
        chosen &= ~task._warehouse_before_pairs
    ids = chosen.nonzero().flatten()
    if not len(ids):
        return progress
    pre = task._before_predecessor[ids]
    people = torch.arange(m, device=task.device)
    leader = (pre[:, :, None] == people).any(1) & (pre < 0) & (task._templates[ids] == 3)
    xy = task._humanoid_root_states[ids, :, :2]-task._env_origins[ids, None, :2]
    xy = scene.rotate_xy(xy-task._structure_shift[ids, WAREHOUSE, None],
                         -task._structure_yaw[ids, WAREHOUSE, None])
    door_x, door_half_width = doorway_geometry(task, ids)
    crossed = (xy[..., 0] >= door_x[:, None]) & (xy[..., 1].abs() <= door_half_width[:, None])
    task._warehouse_door_passed[ids] |= crossed & leader
    offset = xy.clone()
    offset[..., 0] -= door_x[:, None]
    distance = offset.norm(dim=-1)
    sigma = task._relation_cfg['progress']['sigma']
    raw = 1/(1+distance/sigma)
    saturated = task._warehouse_door_passed[ids]
    if getattr(task, '_waypoint_box_only', False):
        from utils.warehouse_waypoint import role_holding
        holding = role_holding(task.relation_runtime, ids, people.expand(len(ids), -1))
        task._waypoint_door_pickup[ids] |= holding & leader
        saturated = saturated | task._waypoint_door_pickup[ids]
    progress[ids] = torch.where(saturated, 1., raw)*leader
    return progress*task._relation_cfg['progress_reward_weight']
