"""Explicit task graphs and contracts for scalar PRE/TERM context (no simulator)."""
from dataclasses import dataclass
import math
import re
import torch

CONTEXT_MODE = 'state_relation_v1'
HOLDING, AT = 6, 7


def context_suffix_size(num_edges):
    return 2 * num_edges


@dataclass(frozen=True)
class EdgeContextGraph:
    ids: tuple
    num_agents: int
    num_objects: int
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_relation: torch.Tensor
    edge_owner: torch.Tensor
    edge_valid: torch.Tensor
    required_goal: torch.Tensor
    prereq_mask: torch.Tensor
    term_index: torch.Tensor

    def to(self, device):
        from dataclasses import fields
        return EdgeContextGraph(**{f.name: (getattr(self, f.name).to(device)
            if isinstance(getattr(self, f.name), torch.Tensor) else getattr(self, f.name)) for f in fields(self)})


def compile_edge_context_graph(spec, num_agents, num_objects, device=None):
    if num_agents < 1 or num_objects < num_agents:
        raise ValueError('Carry scene requires M>=1 and O>=M')
    spec = spec or {'template': 'independent_carry'}
    if set(spec) == {'template'} and spec['template'] == 'independent_carry':
        edges = []
        for a in range(num_agents):
            edges.extend([
                dict(id='hold_' + str(a), owner=a, src='H_' + str(a), relation='HOLDING',
                     dst='O_' + str(a), pre=[], term='at_' + str(a), required_goal=False),
                dict(id='at_' + str(a), owner=a, src='O_' + str(a), relation='AT',
                     dst='G_' + str(a), pre=['hold_' + str(a)], term=None, required_goal=True)])
    elif set(spec) == {'edges'} and isinstance(spec['edges'], list):
        edges = spec['edges']
    else:
        raise ValueError('relationGraph requires edges or template: independent_carry')
    if not edges or any(not isinstance(e, dict) for e in edges):
        raise ValueError('Task graph must contain edge mappings')
    names = [e.get('id') for e in edges]
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError('Edge ids must be unique nonempty strings')
    lookup = {name: i for i, name in enumerate(names)}
    def entity(name, expected):
        m = re.fullmatch(r'([HOG])_(\d+)', str(name))
        if not m or m[1] != expected:
            raise ValueError('Invalid source/target type: ' + str(name))
        index = int(m[2]); limit = num_objects if expected == 'O' else num_agents
        if index >= limit:
            raise ValueError('Entity index out of range: ' + name)
        return index + {'H': 0, 'O': num_agents, 'G': num_agents + num_objects}[expected]
    src, dst, rel, owner, valid, required, term = [], [], [], [], [], [], []
    pre = torch.zeros(len(edges), len(edges), dtype=torch.bool, device=device)
    for i, e in enumerate(edges):
        if set(e) - {'id', 'owner', 'src', 'dst', 'relation', 'pre', 'term', 'required_goal', 'valid'}:
            raise ValueError('Unknown graph edge fields')
        relation = e.get('relation')
        if relation not in ('HOLDING', 'AT'):
            raise ValueError('Unsupported relation: ' + str(relation))
        if type(e.get('owner')) is not int or not 0 <= e['owner'] < num_agents:
            raise ValueError('Invalid edge owner')
        if type(e.get('valid', True)) is not bool or type(e.get('required_goal', False)) is not bool:
            raise ValueError('valid and required_goal must be booleans')
        src.append(entity(e.get('src'), 'H' if relation == 'HOLDING' else 'O'))
        dst.append(entity(e.get('dst'), 'O' if relation == 'HOLDING' else 'G'))
        rel.append(HOLDING if relation == 'HOLDING' else AT)
        owner.append(e['owner']); valid.append(e.get('valid', True)); required.append(e.get('required_goal', False))
        refs = e.get('pre', [])
        if not isinstance(refs, list) or any(not isinstance(r, str) or r not in lookup for r in refs):
            raise ValueError('Unknown prerequisite edge')
        pre[i, [lookup[r] for r in refs]] = True
        t = e.get('term')
        if t is not None and (not isinstance(t, str) or t not in lookup or t == names[i]):
            raise ValueError('TERM requires one other existing edge id or null')
        term.append(-1 if t is None else lookup[t])
    if not any(valid):
        raise ValueError('Task graph must contain a valid edge')
    for i in range(len(edges)):
        if required[i] and not valid[i]:
            raise ValueError('Invalid edge cannot be a required goal')
        if term[i] >= 0 and not valid[term[i]]:
            raise ValueError('TERM cannot reference invalid edges')
        if any(pre[i, j] and not valid[j] for j in range(len(edges))):
            raise ValueError('PRE cannot reference invalid edges')
    tensor = lambda x, dtype=torch.long: torch.tensor(x, dtype=dtype, device=device)
    return EdgeContextGraph(tuple(names), num_agents, num_objects, tensor(src), tensor(dst), tensor(rel),
        tensor(owner), tensor(valid, torch.bool), tensor(required, torch.bool), pre, tensor(term))


def validate_edge_context_config(c):
    expected = {'mode', 'schema_version', 'state_reward_weight', 'progress_reward_weight',
                'success_reward_weight', 'satisfaction_threshold', 'holding', 'at', 'progress',
                'context', 'success', 'observation', 'diagnostics'}
    if set(c) != expected or c['mode'] != CONTEXT_MODE or c['schema_version'] != 2:
        raise ValueError('Invalid edge-context schema/fields')
    fixed = {
        'holding': {'hand_distance_scale': 10.0},
        'at': {'state_definition': 'box_near', 'near_distance_scale': 10.0},
        'progress': {'kind': 'distance', 'delta': .5, 'sigma': 1.0},
        'context': {'kind': 'pre_term_scalar', 'prerequisite_reduction': 'min'},
        'success': {'at_z_tolerance': .001, 'saturation': 'own_or_term_success',
                    'terminate_when_all_subgoals_done': False},
        'observation': {'edge_context_fields': ['pre', 'term']}}
    for key, value in fixed.items():
        if c[key] != value:
            raise ValueError('Unsupported edge-context ' + key)
    for key, expected_value in [('state_reward_weight', .2), ('progress_reward_weight', .2),
                               ('success_reward_weight', .2), ('satisfaction_threshold', .9)]:
        if isinstance(c[key], bool) or not isinstance(c[key], (float, int)) or not math.isfinite(c[key]) or c[key] != expected_value:
            raise ValueError('Unsupported edge-context ' + key)
    d = c['diagnostics']
    if not isinstance(d, dict) or set(d) - {'enabled', 'sample_envs', 'log_interval'}:
        raise ValueError('Unsupported edge-context diagnostics')
    if type(d.get('enabled', True)) is not bool:
        raise ValueError('diagnostics.enabled must be boolean')
    for k, default, minimum in [('sample_envs', 2, 0), ('log_interval', 30, 1)]:
        if type(d.get(k, default)) is not int or d.get(k, default) < minimum:
            raise ValueError('Invalid diagnostics.' + k)


def task_instance(spec, num_agents, num_objects):
    return dict(graph=spec or {'template': 'independent_carry'}, num_agents=num_agents, num_objects=num_objects)


def check_task_resume(weights, spec, num_agents, num_objects):
    if weights.get('relation_task_instance') != task_instance(spec, num_agents, num_objects):
        raise ValueError('Training resume requires identical task graph and entity counts')
