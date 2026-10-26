"""Current two PUSH corridors plus two physical, observed spare boxes."""
import torch
from utils.push_scene import sample_random_at_scene, forward_xy, segment_distance


def advance_rsi_box(initial_position, fixed_goal, progress):
    """Advance along the original segment; never resample or translate the goal."""
    if not torch.isfinite(progress).all() or ((progress < 0) | (progress > 1)).any():
        raise ValueError('RSI box progress must be in [0, 1]')
    return initial_position + progress[..., None] * (fixed_goal-initial_position)


def sample_four_box_scene(count, env, device, generator):
    xy, q, goals = sample_random_at_scene(count, env, device, generator)
    positions = torch.empty(count, 4, 2, device=device)
    rotations = torch.zeros(count, 4, 4, device=device)
    positions[:, :2] = xy
    rotations[:, :2] = q
    rotations[:, 2:, 3] = 1.
    rear = xy - 1.5 * forward_xy(q)
    direction = forward_xy(q)
    if env['pushScene']['random_at_goal'].get('sampling') == 'stage1_square_cone':
        direction = (goals-xy)/(goals-xy).norm(dim=-1,keepdim=True)
        rear = xy - 1.5*direction
    # Also protect all later RSI positions along each complete box trajectory.
    starts = torch.cat((rear, xy), 1)
    ends = torch.cat((goals - 1.5 * direction, goals), 1)
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


def resample_uniform_rsi_goals(plan, goals, interaction, env, generator, distance_range,
                               check_spares=True, return_failures=False):
    """Keep boxes/loco goals fixed; propose RSI distance uniformly in a forward cone.

    Accepted distances are conditioned on the same path/owner clearance and on
    the two already placed spare boxes. No phase or box_progress enters here.
    """
    from utils.push_scene import random_at_goal_valid
    import math
    lo, hi = distance_range
    if not 0 < lo < hi:
        raise ValueError('Uniform RSI needs positive increasing distance bounds')
    result = goals.clone()
    pending = interaction.any(-1).nonzero().flatten()
    forward = forward_xy(plan[:, :2, 3:7])
    tangent = torch.stack((forward[..., 1], -forward[..., 0]), -1)
    angle_limit = math.radians(env['pushScene']['random_at_goal']['half_angle_degrees'])
    radius = math.hypot(*env['pushBox']['size'][:2])/2
    for _ in range(256):
        if not len(pending):
            return (result, torch.zeros(len(plan), dtype=torch.bool, device=plan.device)) if return_failures else result
        draw = torch.rand(len(pending), 2, 2, device=plan.device, generator=generator)
        distance = lo+(hi-lo)*draw[..., 0]
        angle = (2*draw[..., 1]-1)*angle_limit
        direction = forward[pending]*angle.cos()[..., None]+tangent[pending]*angle.sin()[..., None]
        candidate = plan[pending, :2, :2]+distance[..., None]*direction
        active = interaction[pending]
        target = torch.where(active[..., None], candidate, result[pending, :, :2])
        valid = random_at_goal_valid(plan[pending, :2, :2], plan[pending, :2, 3:7],
            target, env, uniform_rsi=active, distance_range=distance_range)
        direction = target-plan[pending, :2, :2]
        direction = direction/direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        rear = plan[pending, :2, :2]-1.5*direction
        starts = torch.cat((rear, plan[pending, :2, :2]), 1)
        ends = torch.cat((target-1.5*direction, target), 1)
        spare = plan[pending, 2:, :2]
        gap = segment_distance(starts[:, :, None], ends[:, :, None],
            spare[:, None], spare[:, None]).amin((1, 2))
        if check_spares:
            valid &= gap >= 2*radius+.5
        accepted = pending[valid]
        result[accepted, :, :2] = target[valid]
        pending = pending[~valid]
    if return_failures:
        failed = torch.zeros(len(plan), dtype=torch.bool, device=plan.device)
        failed[pending] = True
        return result, failed
    raise RuntimeError('Could not sample collision-separated uniform RSI goals')


def support_push_paths_clear(boxes, sizes, goals, push):
    """Each SUPPORT must clear actual PUSH corridors, including RSI body space."""
    delta = goals[..., :2]-boxes[:, :2, :2]
    direction = delta/delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    start = boxes[:, :2, :2]-1.5*direction
    end = goals[..., :2]
    supports = boxes[:, 2:, :2]
    gap = segment_distance(start[:, :, None], end[:, :, None],
        supports[:, None], supports[:, None])
    radius = sizes[..., :2].norm(dim=-1)/2
    required = radius[:, :2, None]+radius[:, None, 2:]+.5
    return ((gap >= required) | ~push[..., None]).all(1)


def clear_unused_stage1_supports(task, ids, templates):
    """Use Stage1's arena proposal; ON_TOP bindings retain their support poses.

    Unused supports remain physical distractors. Only invalid distractor XYs are
    resampled; no asset scale, mass, task binding or ON_TOP target is replaced.
    """
    push = templates == 5
    if not push.any():
        return
    unused = templates != 4  # Canonical support i+2 belongs to owner i.
    radii = task._box_size[ids, ..., :2].norm(dim=-1)/2
    for _ in range(128):
        boxes = task._box_states[ids]
        valid = support_push_paths_clear(boxes, task._box_size[ids], task._tar_pos[ids], push)
        gap = (boxes[:, 2:, None, :2]-boxes[:, None, :, :2]).norm(dim=-1)
        required = radii[:, 2:, None]+radii[:, None, :]+.05
        gap[:, 0, 2] = float('inf');gap[:, 1, 3] = float('inf')
        valid &= (gap >= required).all(-1)
        human_gap = (boxes[:, 2:, None, :2]-task._humanoid_root_states[ids, None, :, :2]).norm(dim=-1)
        valid &= (human_gap >= torch.maximum(radii[:, 2:]+.35,
            radii.new_full(radii[:, 2:].shape, task._box_min_agent_dist))[..., None]).all(-1)
        rows, owners = ((~valid) & unused & push.any(-1)[:, None]).nonzero(as_tuple=True)
        if not len(rows):
            return
        env = ids[rows]
        task._box_states[env, owners+2, :2] = task._sample_arena_xy(task._env_origins[env, :2])
    # Let the inherited bounded whole-scene rejection retry any remaining scene.
