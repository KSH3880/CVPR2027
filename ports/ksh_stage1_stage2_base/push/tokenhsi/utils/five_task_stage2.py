"""Stage2 keeps all five solo tasks; coupled groups share one PUSH box and goal."""
import torch
from utils.shared_support import sample_groups
from utils.edge_scenario_spec import compose_canonical_graph
from utils.edge_context_spec import PUSH_CONTACT, AT
from utils.edge_ontop_spec import ON_TOP

PRESETS = ('random_scenario', 'all_solo', 'all_shared', 'sit', 'climb',
           'holding_at', 'holding_ontop', 'push_at', 'shared_push')


def sample_graph(probabilities, graph_spec, shared_spec, preset='random_scenario'):
    if preset not in PRESETS:
        raise ValueError('Unknown five-task Stage2 preset: '+preset)
    n, m = probabilities.shape[:2]
    device = probabilities.device
    fixed = {'sit':1, 'climb':2, 'holding_at':3, 'holding_ontop':4, 'push_at':5}
    group_preset = 'all_shared' if preset == 'shared_push' else 'all_solo' if preset in fixed else preset
    groups, shared = sample_groups(n,m,shared_spec['coupled_probability'],device,group_preset)
    p = graph_spec['template_probabilities']['PUSH_AT']
    probs = torch.cat((probabilities[...,:5]*(1-p),torch.full_like(probabilities[...,:1],p)), -1)
    templates = torch.multinomial(probs.flatten(0,1),1).reshape(n,m)
    if preset in fixed:
        templates[:] = fixed[preset]
    templates = torch.where(shared,5,templates)
    graph = compose_canonical_graph(templates,shuffle=graph_spec['shuffle_edge_order'],num_objects=3*m)
    people = torch.arange(m,device=device)[None].expand(n,-1)
    source = torch.where(templates == 5,m+torch.where(shared,groups,people),people)
    goal = torch.where(shared,groups,people)
    owners = graph.edge_owner
    push = graph.edge_relation == PUSH_CONTACT
    at = graph.edge_relation == AT
    top = graph.edge_relation == ON_TOP
    graph.edge_dst.copy_(torch.where(push, m+source.gather(1,owners),graph.edge_dst))
    graph.edge_src.copy_(torch.where(at|top,m+source.gather(1,owners),graph.edge_src))
    graph.edge_dst.copy_(torch.where(at,4*m+goal.gather(1,owners),graph.edge_dst))
    graph.edge_dst.copy_(torch.where(top,3*m+owners,graph.edge_dst))
    return graph, groups, shared, templates, source
