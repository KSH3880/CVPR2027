import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from utils.sensor_stage2_spec import WAREHOUSE
from utils.warehouse_coop import DIRECTIONS


FIELDS = ('_static_half', '_static_quat', '_static_local', '_surface_center', '_surface_half',
          '_surface_height', '_structure_bounds', '_structure_yaw', '_structure_shift',
          '_structure_axis', '_structure_direction', '_structure_near')


def build_geometry(task, original_fields):
    room_half, wall_local = task._room_half, task._wall_local
    variants, walls = [], []
    for variant in range(2*len(task._warehouse_coop['pusher_probabilities'])):
        task._warehouse_geometry_variant = variant
        task._build_scene_geometry()
        variants.append({key: getattr(task, key).clone() for key in FIELDS})
        walls.append(tuple(value.clone() for value in task._warehouse_wall_local))
    task._warehouse_geometry = {key: torch.stack([row[key] for row in variants], 1) for key in FIELDS}
    task._warehouse_walls = tuple(torch.stack([row[i] for row in walls], 1) for i in range(2))
    task._warehouse_wall_local = walls[0]
    task._warehouse_geometry_variant = 0
    for key in original_fields:
        setattr(task, key, task._bridge_geometry[key][:, 0].clone())
    task._room_half, task._wall_local = room_half, wall_local


def select_geometry(task, ids):
    if not len(ids):
        return
    spec = task._warehouse_coop
    direction = torch.multinomial(torch.tensor(spec['direction_probabilities'], device=task.device), len(ids), True)
    count = task._bridge_count[ids]
    variant = (direction >= 2).long()*len(spec['pusher_probabilities'])+count-1
    task._warehouse_direction[ids] = direction
    task._bridge_slot[ids] = task._warehouse_bank+count-1
    for key in FIELDS:
        getattr(task, key)[ids, WAREHOUSE] = task._warehouse_geometry[key][ids, variant, WAREHOUSE]
    origin = task._base_scene_origins[ids, WAREHOUSE].clone()
    origin[:, 0] += variant*2*task._scene_spacing
    task._scene_origins[ids, WAREHOUSE] = origin
    task._static_centers[ids, WAREHOUSE] = task._static_local[ids, WAREHOUSE]+origin[:, None]
    task._warehouse_wall_starts[ids] = task._warehouse_walls[0][ids, variant]+origin[:, None, :2]
    task._warehouse_wall_ends[ids] = task._warehouse_walls[1][ids, variant]+origin[:, None, :2]


def opened(task, ids, xy, boxes, sizes):
    cfg = task._warehouse
    direction = task._warehouse_direction[ids]
    inverse = boxes[:, 3:7].clone()
    inverse[:, :3] *= -1
    yaw = task._structure_yaw[ids, WAREHOUSE]
    from utils.geometry import rotate
    axis = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), -1)
    extent = (rotate(inverse, axis).abs()*sizes/2).sum(-1)
    x = xy[:, 0]+task._surface_half[ids, WAREHOUSE, 0, 0]
    clearance = torch.where(direction == 2, x-extent, -cfg['wall_thickness']-x-extent)
    from utils.warehouse_sensor import opening_clearance
    side_clearance = opening_clearance(xy[:, 1], boxes[:, 3:7], sizes, yaw,
        sizes[:, 1]-2*cfg['opening_overlap'])
    clearance = torch.where(direction < 2, side_clearance, clearance)
    return clearance >= cfg['open_clearance']-.001


def diagnostics(task, result):
    spec = task._warehouse_coop
    if not spec:
        return
    for count in range(len(spec['pusher_probabilities'])):
        for direction, name in enumerate(DIRECTIONS):
            for mode, start in enumerate(('closed', 'partial', 'open')):
                key = (count*4+direction)*3+mode
                row = task._warehouse_coop_outcomes[key]
                prefix = 'warehouse/pushers{}/{}/{}/'.format(count+1, name, start)
                result[prefix+'sampled'] = task._warehouse_coop_samples[key].clone()
                result[prefix+'episodes'] = row[0].clone()
                result[prefix+'worker_episodes'] = row[4].clone()
                result[prefix+'opened_fraction'] = row[1]/row[0].clamp_min(1)
                result[prefix+'entered_people_fraction'] = row[2]/row[4].clamp_min(1)
                result[prefix+'pusher_arrived_fraction'] = row[3]/row[0].clamp_min(1)
    task._warehouse_coop_samples.zero_()
    task._warehouse_coop_outcomes.zero_()
