import torch

from utils.warehouse_sensor import walls
from utils.jointpush_scene import segment_distance


def enabled(value, evaluation, warehouse):
    if value not in ('0', '1'):
        raise ValueError('WAREHOUSE_STRAIGHT_PATHS must be 0 or 1')
    if value == '1' and (not evaluation or not warehouse or not warehouse.get('cooperative')):
        raise ValueError('WAREHOUSE_STRAIGHT_PATHS requires cooperative Warehouse evaluation')
    return value == '1'


def wall_rectangles(cfg, opening):
    half = opening.new_tensor(cfg['cooperative']['room_size']).expand(len(opening), 2)/2
    centers, sizes, _, _ = walls(half, cfg['wall_thickness'], cfg['wall_height'], opening)
    centers[..., 0] += half[:, :1]
    return centers[..., :2], sizes[..., :2]


def clear_segments(start, end, centers, half):
    start, end = torch.broadcast_tensors(start, end)
    corners = centers[:, :, None]+half[:, :, None]*start.new_tensor([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    gap = segment_distance(start[:, :, None, None], end[:, :, None, None],
                           corners[:, None], corners.roll(-1, dims=2)[:, None]).amin((-1, -2))
    inside = (((start[:, :, None]-centers[:, None]).abs() <= half[:, None]).all(-1)
              | ((end[:, :, None]-centers[:, None]).abs() <= half[:, None]).all(-1)).any(-1)
    return (gap > 1e-6) & ~inside
