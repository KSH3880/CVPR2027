import colorsys

import torch

from env.tasks.multi_agent.scene_features import object_task_owners
from utils.edge_context_spec import at_goal_marker_slots
from utils.stage1_jointpush import PUSH


def group_colors(owners_by_box, num_agents, base):
    groups = list(range(num_agents))
    for owners in owners_by_box.values():
        labels = {groups[owner] for owner in owners}
        leader = min(labels)
        groups = [leader if label in labels else label for label in groups]
    palette = list(base)
    candidates = [colorsys.hsv_to_rgb(i / (8*num_agents), .75, .95)
                  for i in range(8*num_agents)]
    while len(palette) < len(set(groups)):
        color = max(candidates, key=lambda c: min(
            sum((a-b)**2 for a, b in zip(c, p)) for p in palette))
        palette.append(color)
        candidates.remove(color)
    indices = {label: index for index, label in enumerate(sorted(set(groups)))}
    return [palette[indices[label]] for label in groups]


def role_colors(colors, predecessor):
    if predecessor is None:
        return colors
    leaders = {owner for owner in predecessor if owner >= 0}
    result = []
    for owner, color in enumerate(colors):
        hue, saturation, value = colorsys.rgb_to_hsv(*color)
        if predecessor[owner] >= 0:
            color = colorsys.hsv_to_rgb(hue, max(saturation, .80), max(.70, value*.90))
        elif owner in leaders:
            color = colorsys.hsv_to_rgb(hue, saturation*.40, min(1., value+.12))
        result.append(color)
    return result


def prepare_scene(task, ids):
    mapping = task._logical_box_order[ids]
    retained = torch.zeros(len(ids), task.num_physical_objects,
                           device=task.device, dtype=torch.bool)
    if not hasattr(task, '_design_colors'):
        task._design_colors = [list(task._AGENT_COLOR_PALETTE) for _ in range(task.num_envs)]
        task._design_box_colors = [{} for _ in range(task.num_envs)]
        task._design_saved_boxes = task._logical_box_values(task._box_states).clone()
        task._design_box_mask = torch.zeros(task.num_envs, task.num_objects,
                                           device=task.device, dtype=torch.bool)
    for row, env_id in enumerate(ids.tolist()):
        owners_by_box = {}
        for logical, owners in object_task_owners(task.relation_runtime.graph, env_id).items():
            physical = int(mapping[row, logical])
            owners_by_box.setdefault(physical, set()).update(owners)
            retained[row, physical] = True
        colors = group_colors(owners_by_box, task.num_agents, task._AGENT_COLOR_PALETTE)
        predecessor = (task._before_predecessor[env_id].tolist()
                       if hasattr(task, '_before_predecessor') else None)
        colors = role_colors(colors, predecessor)
        task._design_colors[env_id] = colors
        task._design_box_colors[env_id] = {
            physical: tuple(sum(colors[owner][channel] for owner in owners)/len(owners)
                            for channel in range(3))
            for physical, owners in owners_by_box.items()}
        if task.viewer is not None or (task._video_enabled and env_id == 0):
            from isaacgym import gymapi
            for owner, handle in enumerate(task.humanoid_handles_all[env_id]):
                for body in range(task.num_bodies):
                    task.gym.set_rigid_body_color(task.envs[env_id], handle, body,
                                                 gymapi.MESH_VISUAL, gymapi.Vec3(*colors[owner]))
    task._design_saved_boxes[ids] = task._logical_box_values(task._box_states, ids)
    task._design_box_mask[ids] = ~retained.gather(1, mapping)
    row, physical = (~retained).nonzero(as_tuple=True)
    env = ids[row]
    task._box_states[env, physical, :2] = task._env_origins[env, :2]+1000.+12*physical[:, None]
    task._box_states[env, physical, 7:13] = 0.
    task._update_box_assignment_colors(ids)


def observed_boxes(task, boxes, ids):
    if not hasattr(task, '_design_box_mask'):
        return boxes
    mask = task._design_box_mask if ids is None else task._design_box_mask[ids]
    saved = task._design_saved_boxes if ids is None else task._design_saved_boxes[ids]
    return torch.where(mask[..., None], saved, boxes)


def push_goals(task, env_id):
    graph = task.relation_runtime.graph
    at_edges, slots, _ = at_goal_marker_slots(graph)
    push = graph.edge_valid[env_id] & (graph.edge_relation[env_id] == PUSH)
    owners = set(graph.edge_owner[env_id, push].tolist())
    seen = set()
    result = []
    for edge in at_edges[env_id].nonzero().flatten().tolist():
        owner = int(graph.edge_owner[env_id, edge])
        if owner not in owners:
            continue
        goal = int(slots[env_id, edge])
        shared = hasattr(task, '_members') and int(task._members[env_id, owner]) > 1
        key = ('group', int(task._groups[env_id, owner])) if shared else ('goal', goal)
        if key in seen:
            continue
        seen.add(key)
        target = (task._center_tar_pos[env_id, owner]
                  if shared and task._individual_push_goals else task._tar_pos[env_id, goal])
        result.append((goal, owner, target))
    return result


def hide_goal_markers(task, ids):
    task._marker_pos[ids] = task._env_origins[ids, None]+1000.


def draw_design(task):
    from env.tasks.multi_agent.viewer_goal_markers import draw_goal_markers
    task.gym.clear_lines(task.viewer)
    if task._show_lines_flag:
        draw_goal_markers(task)

