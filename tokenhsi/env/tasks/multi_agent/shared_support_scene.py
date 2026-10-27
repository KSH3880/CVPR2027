import torch
from utils.shared_support import rotate


def scene_infeasible(task, ids):
    boxes, sizes = task._box_states[ids], task._box_size[ids]
    active = task._active_physical(ids)
    radius = sizes[..., :2].norm(dim=-1)/2
    delta = boxes[:, :, None, :3]-boxes[:, None, :, :3]
    overlap = (delta[..., :2].norm(dim=-1) < radius[:, :, None]+radius[:, None]+.05) & (
        delta[..., 2].abs() < (sizes[:, :, None, 2]+sizes[:, None, :, 2])/2-.001)
    eye = torch.eye(task.num_physical_objects, dtype=torch.bool, device=ids.device)
    bad = (overlap & active[:, :, None] & active[:, None] & ~eye).flatten(1).any(-1)
    bodies = task._kinematic_humanoid_rigid_body_states[ids, ..., :3]
    relative = bodies[:, :, None]-boxes[:, None, :, None, :3]
    q = boxes[:, None, :, None, 3:7].clone()
    q[..., :3] *= -1
    inside = (rotate(q, relative).abs() < sizes[:, None, :, None]/2-.03).all(-1)
    bad |= (inside & active[:, None, :, None]).flatten(1).any(-1)
    roots = bodies[:, :, 0, :2]
    distance = (roots[:, :, None]-roots[:, None]).norm(dim=-1)
    human_eye = torch.eye(task.num_agents, dtype=torch.bool, device=ids.device)
    bad |= ((distance < .7) & ~human_eye).flatten(1).any(-1)
    goals = task._tar_pos[ids]
    distance = (goals[:, :, None, :2]-boxes[:, None, :, :2]).norm(dim=-1)
    source = task._agent_box_assignment[ids]
    source_radius = radius.gather(1, source)
    other = torch.arange(task.num_physical_objects, device=ids.device)[None, None] != source[..., None]
    if task.cfg['env'].get('before', {}).get('full_rsi') is not None:
        # A dependent putDown payload may already be above its leader's AT goal.
        # Box/body collision checks still apply; this is only the empty-goal XY heuristic.
        pre = task._before_predecessor[ids]
        row, follower = ((pre >= 0) & (task._templates[ids] == 4)
                         & (task._full_rsi_skill[ids] == 7)).nonzero(as_tuple=True)
        other[row, pre[row, follower], task._agent_box_assignment[ids[row], follower]] = False
    bad |= ((task._templates[ids] == 3)[..., None] & active[:, None] & other &
            (distance < source_radius[..., None]+radius[:, None]+.25)).flatten(1).any(-1)
    return bad | ~torch.isfinite(bodies).flatten(1).all(-1)
