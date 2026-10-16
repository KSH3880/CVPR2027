import torch

from utils.task_role_spec import task_graph_packet
from utils.group_carry_spec import owner_tasks
from learning.multi_agent.human_coordination import human_relations
from env.tasks.multi_agent.joint_carry_reward import anchor_scores


def group_cpa_release(roots, objects, sizes, graph, buffer):
    packet = task_graph_packet(graph, len(roots))
    tasks = owner_tasks(packet, graph.num_agents).long()
    coupled = human_relations(packet, graph.num_agents) == 2
    rows = torch.arange(len(roots), device=roots.device)[:, None]
    payload = tasks[..., 3] - graph.num_agents
    near = (roots[..., :2]-objects[rows, payload, :2]).norm(dim=-1) <= \
        sizes[rows, payload, :2].norm(dim=-1)/2 + buffer
    return coupled & near[:, :, None] & near[:, None, :]


def apply_group_geometry(phi, diag, hands, objects, sizes, graph, config):
    n, m = hands.shape[:2]
    packet = task_graph_packet(graph, n)
    tasks = owner_tasks(packet, m).long()
    coupled = human_relations(packet, m) == 2
    joint = coupled.any(-1)
    rows = torch.arange(n, device=phi.device)[:, None]
    payload = tasks[..., 3]-m
    box = objects[rows, payload]
    size = sizes[rows, payload]
    scores, anchors = anchor_scores(hands.reshape(n*m, 1, 2, 3), box.reshape(n*m, 13),
        size.reshape(n*m, 3), config['joint_carry']['anchor_inset'],
        config['holding']['hand_distance_scale'])
    scores = scores.reshape(n, m, 2)
    anchors = anchors.reshape(n, m, 2, 3)
    close = scores >= config['satisfaction_threshold']
    opposite = (close[:, :, None, 0] & close[:, None, :, 1]) | \
        (close[:, :, None, 1] & close[:, None, :, 0])
    gate = (coupled & opposite).any(-1)
    best, choice = scores.max(-1)
    target = anchors[rows, torch.arange(m, device=phi.device)[None], choice]
    owner = graph.edge_owner
    holding = graph.edge_valid & (graph.edge_relation == 6) & joint.gather(1, owner)
    phi = torch.where(holding, best.gather(1, owner), phi)
    diag['target'] = torch.where(holding[..., None], target[rows, owner], diag['target'])
    distance = (hands.mean(-2)-target).norm(dim=-1)
    diag['distance'] = torch.where(holding, distance.gather(1, owner), diag['distance'])
    return phi, gate
