import numpy as np

from utils.edge_context_spec import AT
from utils.edge_interaction_spec import SIT, CLIMB
from utils.edge_ontop_spec import ON_TOP
from utils.stage1_jointpush import PUSH


def rotation(quaternion):
    x, y, z, w = quaternion/np.linalg.norm(quaternion)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def rectangle(half, corners=False):
    points = np.array([[-half[0], -half[1], 0], [half[0], -half[1], 0],
                       [half[0], half[1], 0], [-half[0], half[1], 0]])
    edges = np.stack((points, np.roll(points, -1, axis=0)), axis=1)
    if corners:
        short = edges[:, 0]+.24*(edges[:, 1]-edges[:, 0])
        end = edges[:, 1]+.24*(edges[:, 0]-edges[:, 1])
        edges = np.concatenate((np.stack((edges[:, 0], short), 1),
                                np.stack((end, edges[:, 1]), 1)))
    return edges


def surface_lines(state, size, relation, payload_size=None):
    half = size[:2]*.49
    if relation == ON_TOP:
        half = np.minimum(payload_size[:2]/2, size[:2]*.42)
    edges = rectangle(half, corners=relation == ON_TOP)
    if relation == SIT:
        x = np.linspace(-half[0]*.9, half[0]*.9, 7)
        start = np.column_stack((x, np.full(7, -half[1]*.9), np.zeros(7)))
        end = start.copy()
        end[:, 1] *= -1
        edges = np.concatenate((edges, np.stack((start, end), 1)))
    edges[..., 2] += size[2]/2+.012
    vertices = edges@rotation(state[3:7]).T+state[:3]
    if relation in (ON_TOP, CLIMB):
        top = state[:3]+rotation(state[3:7])@np.array([0, 0, size[2]/2])
        bottom = top+np.array([0, 0, .12])
        upper = top+np.array([0, 0, .42])
        start, tip = (upper, bottom) if relation == ON_TOP else (bottom, upper)
        sign = 1 if relation == ON_TOP else -1
        head = [tip+np.array([-.055, 0, .07*sign]),
                tip+np.array([.055, 0, .07*sign])]
        vertices = np.concatenate((vertices, np.array([[start, tip], [tip, head[0]], [tip, head[1]]])))
    return vertices


def draw_goal_markers(task):
    graph = task.relation_runtime.graph
    objects = task._logical_box_values(task._box_states).detach().cpu().numpy()
    sizes = task._logical_box_values(task._box_size).detach().cpu().numpy()
    goals = task._tar_pos.detach().cpu().numpy()
    origins = task._env_origins.cpu().numpy()
    m, o = task.num_agents, task.num_objects
    for env in range(task.num_envs):
        valid = graph.edge_valid[env].cpu().numpy()
        relations = graph.edge_relation[env].cpu().numpy()
        owners = graph.edge_owner[env].cpu().numpy()
        src = graph.edge_src[env].cpu().numpy()-m
        dst = graph.edge_dst[env].cpu().numpy()-m
        push = set(owners[valid & (relations == PUSH)])
        for edge in np.flatnonzero(valid):
            relation, owner = relations[edge], int(owners[edge])
            if relation == AT and owner not in push:
                target = goals[env, dst[edge]-o].copy()
                target[2] = max(target[2]-sizes[env, src[edge], 2]/2, .012)
                vertices = rectangle(sizes[env, src[edge], :2]/2)+target
            elif relation in (ON_TOP, SIT, CLIMB):
                payload = sizes[env, src[edge]] if relation == ON_TOP else None
                vertices = surface_lines(objects[env, dst[edge]], sizes[env, dst[edge]], relation, payload)
            else:
                continue
            color = task._viewer_agent_color(env, owner)
            colors = np.tile((color.x, color.y, color.z), (len(vertices), 1)).astype(np.float32)
            task.gym.add_lines(task.viewer, task.envs[env], len(vertices),
                               (vertices-origins[env]).astype(np.float32), colors)
    if hasattr(task, '_joint'):
        from env.tasks.multi_agent.push_goal_markers import draw_push_goal_markers
        draw_push_goal_markers(task)
