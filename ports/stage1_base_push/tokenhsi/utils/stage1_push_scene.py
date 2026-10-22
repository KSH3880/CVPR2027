"""Current two PUSH corridors plus two physical, observed spare boxes."""
import torch
from utils.push_scene import sample_random_at_scene, forward_xy, segment_distance


def sample_four_box_scene(count, env, device, generator):
    xy, q, goals = sample_random_at_scene(count, env, device, generator)
    positions = torch.empty(count, 4, 2, device=device)
    rotations = torch.zeros(count, 4, 4, device=device)
    positions[:, :2] = xy
    rotations[:, :2] = q
    rotations[:, 2:, 3] = 1.
    rear = xy - 1.5 * forward_xy(q)
    # Also protect all later RSI positions along each complete box trajectory.
    starts = torch.cat((rear, xy), 1)
    ends = torch.cat((goals - 1.5 * forward_xy(q), goals), 1)
    radius = torch.tensor(env['pushBox']['size'][:2], device=device).norm()/2
    anchors = torch.tensor(env['pushScene']['owner_anchor_xy'], device=device)
    for box in (2, 3):
        pending = torch.arange(count, device=device)
        for _ in range(128):
            if not len(pending):
                break
            candidate = (torch.rand(len(pending), 2, device=device, generator=generator)*2-1)*10
            clearance = segment_distance(starts[pending], ends[pending],
                candidate[:, None], candidate[:, None]).amin(-1)
            valid = clearance >= 2*radius + .5
            valid &= ((candidate[:, None] - anchors).norm(dim=-1) >= radius + 1.).all(-1)
            valid &= ((candidate[:, None] - positions[pending, :box]).norm(dim=-1) >= 2*radius + .5).all(-1)
            positions[pending[valid], box] = candidate[valid]
            pending = pending[~valid]
        if len(pending):
            raise RuntimeError('Could not place four PUSH boxes without corridor interference')
    return positions, rotations, goals
