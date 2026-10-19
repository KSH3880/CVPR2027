import torch

from utils.edge_context_spec import HOLDING, AT
from utils.edge_ontop_spec import ON_TOP, permute_graph, select_graph
from utils.edge_stage2_spec import _empty_graph
from utils.joint_carry_spec import JOINT_PRESETS


MIXED_SAMPLER = 'two_agent_mixed_carry_task_embedding'
INDEPENDENT_PRESETS = ('independent', 'independent_at_at', 'independent_at_ontop',
                       'independent_ontop_at', 'independent_ontop_ontop')
MIXED_PRESETS = JOINT_PRESETS + INDEPENDENT_PRESETS


def environment_mask(n, device, preset='random_scenario'):
    if preset == 'random_scenario':
        return torch.randperm(n, device=device) >= round(n * .2)
    return torch.full((n,), preset in JOINT_PRESETS, device=device, dtype=torch.bool)


def validate_sampler(spec, m=2, o=4):
    expected = dict(mode='edge_composition', sampler=MIXED_SAMPLER,
        semantic_only=True, policy_task_roles=True, max_edges_per_agent=2,
        edge_capacity=4, family_probabilities={'joint_carry_at': .4,
            'joint_carry_ontop': .4, 'independent': .2},
        shuffle_edge_order=True, shuffle_token_order=True,
        coupling='same_carry_task_payload_target')
    if spec != expected or (m, o) != (2, 4):
        raise ValueError('Mixed carry requires 2 humans/4 objects and joint/independent 80/20')


def joint_scenes(graph):
    hold = graph.edge_valid & (graph.edge_relation == HOLDING)
    payload_a = (graph.edge_dst * hold * (graph.edge_owner == 0)).sum(-1)
    payload_b = (graph.edge_dst * hold * (graph.edge_owner == 1)).sum(-1)
    return payload_a == payload_b


def sample_graph(n, spec, device='cpu', preset='random_scenario', role_swap=False,
                 generator=None, joint_mask=None):
    validate_sampler(spec)
    if preset not in MIXED_PRESETS:
        raise ValueError('Unknown mixed-carry preset: ' + preset)
    if joint_mask is None:
        joint_mask = (torch.rand(n, device=device, generator=generator) < .8
                      if preset == 'random_scenario' else
                      torch.full((n,), preset in JOINT_PRESETS, device=device, dtype=torch.bool))
    ontop = torch.rand(n, 2, device=device, generator=generator) >= .5
    if preset == 'joint_carry_at':
        ontop[:] = False
    elif preset == 'joint_carry_ontop':
        ontop[:] = True
    elif preset.startswith('independent_'):
        ontop[:] = torch.tensor([part == 'ontop' for part in preset.split('_')[1:]], device=device)
    ontop[:, 1] = torch.where(joint_mask, ontop[:, 0], ontop[:, 1])
    graph = _empty_graph(n, 2, 4, 4, device)
    payload_b = torch.where(joint_mask, 2, 3)
    target_b = torch.where(joint_mask, torch.where(ontop[:, 0], 4, 6),
                          torch.where(ontop[:, 1], 5, 7))
    graph.edge_src[:] = torch.tensor([0, 2, 1, 2], device=device)
    graph.edge_src[:, 3] = payload_b
    graph.edge_dst[:, 0] = 2
    graph.edge_dst[:, 1] = torch.where(ontop[:, 0], 4, 6)
    graph.edge_dst[:, 2] = payload_b
    graph.edge_dst[:, 3] = target_b
    graph.edge_relation[:, 0::2] = HOLDING
    graph.edge_relation[:, 1::2] = torch.where(ontop, ON_TOP, AT)
    graph.edge_owner[:] = torch.tensor([0, 0, 1, 1], device=device)
    graph.edge_valid[:] = True
    graph.required_goal[:, 1::2] = True
    return permute_graph(graph, torch.rand(n, 4, device=device, generator=generator).argsort(-1))


def compile_graph(spec, m, o, device=None):
    validate_sampler(spec, m, o)
    return select_graph(sample_graph(1, spec, device or 'cpu', 'joint_carry_at'), 0)
