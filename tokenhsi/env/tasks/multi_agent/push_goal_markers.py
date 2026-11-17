import numpy as np
from utils.stage1_jointpush import PUSH
from utils.edge_context_spec import at_goal_marker_slots


def draw_push_goal_markers(task):
    graph = task.relation_runtime.graph
    at_edges, slots, _ = at_goal_marker_slots(graph)
    angles = np.linspace(0, 2*np.pi, 33)
    for env_id in range(task.num_envs):
        push = graph.edge_valid[env_id] & (graph.edge_relation[env_id] == PUSH)
        owners = set(graph.edge_owner[env_id, push].tolist())
        goals = {}
        for edge in at_edges[env_id].nonzero().flatten().tolist():
            owner = int(graph.edge_owner[env_id, edge])
            if owner not in owners:
                continue
            shared = hasattr(task, '_members') and int(task._members[env_id, owner]) > 1
            agent_color = task._agent_color(owner)
            color = (1., 1., 0.) if shared else (agent_color.x, agent_color.y, agent_color.z)
            goals[int(slots[env_id, edge])] = color
        for goal_id, color in goals.items():
            target = task._tar_pos[env_id, goal_id].detach().cpu().numpy()
            ring = np.column_stack((target[0]+.35*np.cos(angles),
                target[1]+.35*np.sin(angles), np.full_like(angles, .015)))
            lines = [np.stack((ring[:-1], ring[1:]), axis=1),
                     np.array([[[target[0], target[1], .015], target]])]
            for a, b in ((0, 1), (0, 2), (1, 2)):
                circle = np.tile(target, (len(angles), 1))
                circle[:, a] += .20*np.cos(angles)
                circle[:, b] += .20*np.sin(angles)
                lines.append(np.stack((circle[:-1], circle[1:]), axis=1))
            vertices = np.concatenate(lines).astype(np.float32)
            colors = np.tile(color, (len(vertices), 1)).astype(np.float32)
            task.gym.add_lines(task.viewer, task.envs[env_id], len(vertices), vertices, colors)
