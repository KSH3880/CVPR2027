import copy
import torch

from utils.edge_context_spec import compile_edge_context_graph
from utils.edge_scenario_spec import INDEPENDENT_CLIMB_SAMPLER

BOX_SIZE = (.4, .4, .4)


def cleanup_graph_spec(round_index=0):
    return dict(mode='stage1_cleanup_demo', sampler=INDEPENDENT_CLIMB_SAMPLER,
                semantic_only=True, round_index=round_index)


def compile_cleanup_graph(spec, agents, objects, device=None):
    if (agents, objects) != (4, 16) or spec != cleanup_graph_spec(spec.get('round_index')):
        raise ValueError('Cleanup demo requires 4 agents, 16 objects and semantic edges')
    if spec['round_index'] not in (0, 1):
        raise ValueError('Cleanup demo has two rounds')
    edges = []
    for agent in range(agents):
        box = agent + 4 * spec['round_index']
        edges.extend([
            dict(id='hold_' + str(agent), owner=agent, src='H_' + str(agent),
                 dst='O_' + str(box), relation='HOLDING', required_goal=True),
            dict(id='at_' + str(agent), owner=agent, src='O_' + str(box),
                 dst='G_' + str(agent), relation='AT', required_goal=True)])
    return compile_edge_context_graph({'edges': edges}, agents, objects, device)


def cleanup_layout(box_sizes):
    device = box_sizes.device
    directions = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]], device=device)
    boxes = torch.zeros(16, 3, device=device)
    columns = torch.tensor([[1.05, -.35], [.35, .35], [-1.05, .35], [-.35, -.35],
                            [1.05, .35], [-.35, .35], [-1.05, -.35], [.35, -.35]], device=device)
    row_shift = (torch.rand(2, device=device) - .5) * .14
    columns[:, 0] += row_shift[(columns[:, 1] > 0).long()]
    columns += (torch.rand(8, 2, device=device) - .5) * .05
    boxes[8:, :2] = columns
    boxes[:8, :2] = columns + (torch.rand(8, 2, device=device) - .5) * .03
    boxes[:8, 2] = box_sizes[8:, 2] + box_sizes[:8, 2] / 2 + .001
    boxes[8:, 2] = box_sizes[8:, 2] / 2
    yaw = (torch.rand(16, device=device) - .5) * (torch.pi / 6)
    rotations = torch.zeros(16, 4, device=device)
    rotations[:, 2] = torch.sin(yaw / 2)
    rotations[:, 3] = torch.cos(yaw / 2)
    goals = torch.zeros(2, 4, 3, device=device)
    side = torch.stack([-directions[:, 1], directions[:, 0]], -1)
    offsets = torch.tensor([-.4, .4], device=device)[:, None, None]
    goals[..., :2] = directions[None] * 3.25 + offsets * side[None]
    goals[..., 2] = box_sizes[:8, 2].reshape(2, 4) / 2
    return directions * 2.65, boxes, rotations, goals


def cleanup_reward_config(saved):
    config = copy.deepcopy(saved)
    if config.get('stage1_variant') != 'scenario_independent_stage1_self_sum_rescue':
        raise ValueError('Cleanup demo requires the supplied Stage-1 rescue checkpoint')
    config['stage1_variant'] = 'scenario_independent_stage1_self_sum'
    config['hard_skill_training'].pop('at_placement')
    config['hard_skill_training'].pop('climb_state')
    return config
