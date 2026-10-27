import numpy as np
from isaacgym import gymapi

from utils.stage1_jointpush import PUSH


def update_push_goal_markers(task, env_ids, graph, at_edges, goal_slots):
    if not hasattr(task, '_push_goal_markers'):
        task._push_goal_markers = {}
    for row, env_id in enumerate(env_ids.tolist()):
        push_owners = set(graph.edge_owner[row][graph.edge_valid[row] &
            (graph.edge_relation[row] == PUSH)].tolist())
        goals = {}
        for edge in at_edges[row].nonzero().flatten().tolist():
            owner = int(graph.edge_owner[row,edge])
            goal_id = int(goal_slots[row,edge])
            is_push = owner in push_owners
            shared = is_push and hasattr(task, '_members') and int(task._members[env_id,owner]) > 1
            color = gymapi.Vec3(1.,1.,0.) if shared else task._agent_color(owner)
            marker = task._marker_handles[env_id*task.num_agents+goal_id]
            task.gym.set_rigid_body_color(task.envs[env_id],marker,0,gymapi.MESH_VISUAL,color)
            if is_push:
                goals[goal_id] = (color.x,color.y,color.z)
        task._push_goal_markers[env_id] = goals


def draw_push_goal_markers(task):
    angles = np.linspace(0,2*np.pi,33)
    for env_id, goals in getattr(task,'_push_goal_markers',{}).items():
        if not goals:
            continue
        targets = task._tar_pos[env_id].detach().cpu().numpy()
        for goal_id, color in goals.items():
            target = targets[goal_id]
            ring = np.column_stack((target[0]+.35*np.cos(angles),
                target[1]+.35*np.sin(angles),np.full_like(angles,.015)))
            segments = np.stack((ring[:-1],ring[1:]),axis=1)
            stem = np.array([[[target[0],target[1],.015],target]])
            sphere = []
            for a,b in ((0,1),(0,2),(1,2)):
                circle = np.tile(target,(len(angles),1))
                circle[:,a] += .1*np.cos(angles)
                circle[:,b] += .1*np.sin(angles)
                sphere.append(np.stack((circle[:-1],circle[1:]),axis=1))
            vertices = np.concatenate((segments,stem,*sphere)).astype(np.float32)
            colors = np.tile(color,(len(vertices),1)).astype(np.float32)
            task.gym.add_lines(task.viewer,task.envs[env_id],len(vertices),vertices,colors)
