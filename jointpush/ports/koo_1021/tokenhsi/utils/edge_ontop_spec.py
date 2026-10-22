"""Per-episode edge composition and versioned, replayable graph packets (no simulator)."""
from dataclasses import fields, replace
from copy import deepcopy
import math
import torch
from utils.edge_context_spec import EdgeContextGraph, HOLDING, AT, validate_edge_context_config, CONTEXT_MODE

ONTOP_CONTEXT_MODE = 'state_relation_edge_ontop_v1'
ON_TOP = 8
PACKET_FIELDS = ('valid', 'src', 'dst', 'relation', 'owner', 'pre', 'term')
PACKET_WIDTH = len(PACKET_FIELDS)
PRESETS = ('random', 'at_ontop', 'ontop_chain', 'independent_ontop')
DEFAULT_GRAPH = dict(mode='edge_composition', sampler='two_agent_three_object',
    edge_capacity=4, max_edges_per_agent=2,
    second_edge_probabilities={'NONE': .2, 'AT': .5, 'ON_TOP': .3},
    ontop={'free_support_probability_when_other_at': .5,
           'first_agent_probability_when_both_ontop': .5,
           'max_children_per_support': 1, 'reject_cycles': True}, shuffle_edge_order=True)
SHARING = dict(enabled=True, kind='fixed_two_agent_task_mix', self_weight=.9, other_weight=.1,
               apply_to='task_edges_only', apply_on_all_training_graphs=True)


def validate_ontop_context_config(c):
    base = deepcopy(c)
    ontop = base.pop('ontop', None); sharing = base.pop('sharing', None)
    if base.get('mode') != ONTOP_CONTEXT_MODE or base.get('schema_version') != 3:
        raise ValueError('Expected OnTop scalar-context schema 3')
    if ontop != dict(state_definition='centered_stack_world_z', near_distance_scale=10.,
                     z_tolerance=.001, vertical_extent='rotated_bbox') or sharing != SHARING:
        raise ValueError('Unsupported OnTop geometry or .9/.1 task sharing contract')
    if base['observation'].pop('graph_packet_fields', None) != list(PACKET_FIELDS):
        raise ValueError('OnTop requires the versioned 7-field graph packet')
    base.update(mode=CONTEXT_MODE, schema_version=2)
    validate_edge_context_config(base)


def validate_sampler(spec, m=2, o=3):
    if (m, o) != (2, 3):
        raise ValueError('Edge composition sampler/presets require M=2, O=3; use explicit edges for generic inference')
    if set(spec) != set(DEFAULT_GRAPH):
        raise ValueError('Unsupported edge composition fields')
    for k in ('mode', 'sampler', 'edge_capacity', 'max_edges_per_agent', 'ontop'):
        if spec[k] != DEFAULT_GRAPH[k]:
            raise ValueError('Unsupported edge composition ' + k)
    p = spec['second_edge_probabilities']
    if set(p) != {'NONE', 'AT', 'ON_TOP'} or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in p.values()) or not math.isclose(sum(p.values()), 1., abs_tol=1e-8):
        raise ValueError('Second edge probabilities must be finite, nonnegative and sum to one')
    if type(spec['shuffle_edge_order']) is not bool:
        raise ValueError('shuffle_edge_order must be boolean')


def batched(value, n):
    return value.unsqueeze(0).expand(n, *value.shape) if value.ndim == 1 else value


def select_graph(g, ids):
    if g.edge_src.ndim == 1:
        return g
    return replace(g, **{f.name: getattr(g, f.name)[ids] for f in fields(g)
                         if isinstance(getattr(g, f.name), torch.Tensor)})


def expand_graph(g, n):
    return replace(g, **{f.name: getattr(g, f.name).unsqueeze(0).expand(n, *getattr(g, f.name).shape).clone()
                         for f in fields(g) if isinstance(getattr(g, f.name), torch.Tensor)})


def copy_graph_rows(dst, ids, src):
    for f in fields(dst):
        v = getattr(dst, f.name)
        if isinstance(v, torch.Tensor):
            v[ids] = getattr(src, f.name)


def permute_graph(g, permutation):
    """Per-scene edge permutation, including both PRE axes and TERM references."""
    n, e = permutation.shape
    inv = permutation.argsort(-1)
    data = {}
    for f in fields(g):
        v = getattr(g, f.name)
        if not isinstance(v, torch.Tensor):
            continue
        if f.name == 'prereq_mask':
            data[f.name] = v.gather(1, permutation[..., None].expand(n,e,e)).gather(2, permutation[:,None].expand(n,e,e))
        elif f.name == 'term_index':
            old = v.gather(1, permutation)
            data[f.name] = torch.where(old >= 0, inv.gather(1, old.clamp_min(0)), -1)
        else:
            data[f.name] = v.gather(1, permutation)
    return replace(g, **data)


def compose_graph(second, target, shuffle=False, generator=None):
    """second [N,2]: 0=NONE,1=AT,2=ON_TOP; target is logical support slot."""
    n = second.shape[0]; device = second.device
    a = torch.arange(2, device=device)[None].expand(n, -1)
    valid = torch.stack([torch.ones_like(second, dtype=torch.bool), second != 0], -1).flatten(1)
    src = torch.stack([a, 2+a], -1).flatten(1)
    dst = torch.stack([2+a, torch.where(second == 1, 5+a, 2+target)], -1).flatten(1)
    rel = torch.stack([torch.full_like(second,HOLDING), torch.where(second == 1, AT, ON_TOP)], -1).flatten(1)
    owner = a.repeat_interleave(2, -1)
    required = torch.stack([second == 0, second != 0], -1).flatten(1)
    term = torch.stack([torch.where(second != 0, 2*a+1, -1), torch.full_like(a,-1)], -1).flatten(1)
    pre = torch.zeros(n,4,4,dtype=torch.bool,device=device)
    pre[:,1,0] = second[:,0] != 0; pre[:,3,2] = second[:,1] != 0
    pre[:,1,3] = (second[:,0] == 2) & (target[:,0] == 1)
    pre[:,3,1] = (second[:,1] == 2) & (target[:,1] == 0)
    # Canonical safe padding, never an embedded NONE task edge.
    src = src.masked_fill(~valid,0); dst = dst.masked_fill(~valid,0)
    rel = rel.masked_fill(~valid,0); owner = owner.masked_fill(~valid,0)
    g = EdgeContextGraph(('slot0','slot1','slot2','slot3'),2,3,src,dst,rel,owner,valid,required,pre,term)
    if shuffle:
        g = permute_graph(g,torch.rand(n,4,device=device,generator=generator).argsort(-1))
    return g


def sample_graph(n, spec, device='cpu', preset='random', role_swap=False, generator=None):
    validate_sampler(spec)
    if preset not in PRESETS:
        raise ValueError('Unknown TASK_GRAPH preset: ' + preset)
    if preset == 'random':
        p = spec['second_edge_probabilities']
        u = torch.rand(n,2,device=device,generator=generator)
        second = (u >= p['NONE']).long() + (u >= p['NONE']+p['AT']).long()
        target = torch.full_like(second,2)
        u = torch.rand(n,2,device=device,generator=generator)
        for a in range(2):
            other=1-a
            dependent=(second[:,a]==2)&(second[:,other]==1)&(u[:,a]>=.5)
            target[dependent,a]=other
        both=(second==2).all(-1)
        lower=(torch.rand(n,device=device,generator=generator)>=.5).long()
        rows=both.nonzero(as_tuple=False).flatten()
        upper=1-lower[rows];target[rows,upper]=lower[rows]
    else:
        types={'at_ontop':(1,2),'ontop_chain':(2,2),'independent_ontop':(2,0)}[preset]
        second=torch.tensor(types,device=device).expand(n,-1).clone()
        target=torch.tensor((2,0) if preset!='independent_ontop' else (2,2),device=device).expand(n,-1).clone()
        if role_swap:
            second=second.flip(-1);target=target.flip(-1)
            target=torch.where(target<2,1-target,target)
    return compose_graph(second,target,shuffle=spec['shuffle_edge_order'] if preset=='random' else False,generator=generator)


def packet_size(capacity):
    return PACKET_WIDTH * capacity


def graph_packet(g, context):
    n=context.shape[0]
    values=[batched(getattr(g,k),n).to(context.dtype) for k in
            ('edge_valid','edge_src','edge_dst','edge_relation','edge_owner')]
    packet=torch.cat([torch.stack(values,-1),context],-1)
    return (packet * values[0][...,None]).flatten(1)


def parse_packet(packet):
    if packet.ndim != 2 or packet.shape[-1] % PACKET_WIDTH:
        raise ValueError('Expected stored graph packet [N,7*E]')
    p=packet.reshape(packet.shape[0],-1,PACKET_WIDTH)
    return p[...,0].bool(),p[...,1].long(),p[...,2].long(),p[...,3].long(),p[...,4].long(),p[...,5:7]


def compile_ontop_graph(spec, m, o, device=None):
    """Static prototype for layout; explicit graphs also support generic M/O/E inference."""
    if spec.get('mode')=='edge_composition':
        validate_sampler(spec,m,o)
        return select_graph(sample_graph(1,spec,device=device or 'cpu',preset='at_ontop'),0)
    # Reuse the existing strict syntax parser by temporarily mapping Object->Object
    # ON_TOP to a correctly typed AT placeholder, then restore the real bindings.
    from utils.edge_context_spec import compile_edge_context_graph
    if set(spec) != {'edges'}:
        raise ValueError('Expected edge_composition or explicit edges')
    adapted=deepcopy(spec); tops=[]
    for i,e in enumerate(adapted['edges']):
        if e.get('relation')=='ON_TOP':
            dst=e.get('dst','')
            if not dst.startswith('O_') or not dst[2:].isdigit() or int(dst[2:])>=o:
                raise ValueError('ON_TOP target must be an existing Object')
            tops.append((i,int(dst[2:])))
            e.update(relation='AT',dst='G_0')
    g=compile_edge_context_graph(adapted,m,o,device)
    for i,t in tops:
        g.edge_relation[i]=ON_TOP;g.edge_dst[i]=m+t
    validate_graph(g)
    return g


def validate_graph(g):
    """Validate explicit topology; PRE and support DAGs separately (TERM may point back)."""
    if g.edge_src.ndim != 1:
        for i in range(g.edge_src.shape[0]):validate_graph(select_graph(g,i))
        return
    active=g.edge_valid.nonzero(as_tuple=False).flatten().tolist()
    def acyclic(adjacency):
        visiting=set();done=set()
        def visit(node):
            if node in visiting:raise ValueError('Cyclic support or PRE graph')
            if node in done:return
            visiting.add(node)
            for child in adjacency.get(node,[]):visit(child)
            visiting.remove(node);done.add(node)
        for node in adjacency:visit(node)
    acyclic({i:g.prereq_mask[i].nonzero(as_tuple=False).flatten().tolist() for i in active})
    support={};children=set()
    placements={int(g.edge_src[i]):i for i in active if int(g.edge_relation[i]) in (AT,ON_TOP)}
    held={int(g.edge_dst[i]) for i in active if int(g.edge_relation[i])==HOLDING}
    for i in active:
        if int(g.edge_relation[i])!=ON_TOP:continue
        s,t=int(g.edge_src[i]),int(g.edge_dst[i])
        if s==t or t in children:raise ValueError('Self support or multiple children on one support')
        if t in held and t not in placements:raise ValueError('HOLDING-only object is not a free support')
        children.add(t);support.setdefault(s,[]).append(t)
    acyclic(support)
