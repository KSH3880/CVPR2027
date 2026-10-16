import torch

from utils.edge_context_spec import HOLDING, AT
from utils.edge_ontop_spec import ON_TOP, select_graph
from utils.edge_stage2_spec import _empty_graph
from utils.mixed_carry_spec import MIXED_SAMPLER


def demo_graph_spec(agents=6):
    return dict(mode='joint_carry_viewer_demo', sampler=MIXED_SAMPLER,
                semantic_only=True, policy_task_roles=True, edge_capacity=2 * agents)


def compile_demo_graph(spec, agents, objects, device=None):
    if agents not in (4, 6) or objects != 8 or spec != demo_graph_spec(agents):
        raise ValueError('Mixed-scene viewer requires 4 or 6 humans and 8 objects')
    graph = _empty_graph(1, agents, objects, 2 * agents, device or 'cpu')
    payload_ids = [0, 0, 1, 1, 2, 3] if agents == 6 else [0, 0, 2, 3]
    target_ids = [14, 14, 16, 16, 18, 10] if agents == 6 else [12, 12, 14, 8]
    payload = torch.tensor(payload_ids, device=device) + agents
    target = torch.tensor(target_ids, device=device)
    graph.edge_src[:, 0::2] = torch.arange(agents, device=device)
    graph.edge_dst[:, 0::2] = payload
    graph.edge_src[:, 1::2] = payload
    graph.edge_dst[:, 1::2] = target
    graph.edge_owner[:] = torch.arange(agents, device=device).repeat_interleave(2)
    graph.edge_relation[:, 0::2] = HOLDING
    graph.edge_relation[:, 1::2] = AT
    graph.edge_relation[:, -1] = ON_TOP
    graph.edge_valid[:] = True
    graph.required_goal[:, 1::2] = True
    return select_graph(graph, 0)


def demo_layout(device):
    centers = torch.tensor([[-1.25, -4.2, 0.], [-1.25, -1.4, 0.],
                            [-1.25, 1.4, 0.], [-1.25, 4.2, 0.]], device=device)
    goals = centers.clone()
    goals[:, 0] += 2.5
    return centers, goals
