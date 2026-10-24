"""Variable Push groups and fixed eight-token scene geometry, without a simulator."""
import torch
from itertools import permutations


def random_regions(same_group):
    """Uniform random bijection within each physical group, with no distance input."""
    n,m,_=same_group.shape
    choices=torch.tensor(list(permutations(range(m))),device=same_group.device)
    rows=torch.arange(m,device=same_group.device)
    allowed=same_group[:,rows,choices].all(-1)
    if not allowed.any(-1).all():raise ValueError('No group-preserving region bijection')
    selected=torch.multinomial(allowed.float(),1).flatten()
    return choices[selected]


def sample_rear_spawn(boxes,sizes,goals,distance_range,min_separation=.9,clearance=.45):
    """Goal-opposite 180-degree center positions; independent full-circle headings."""
    lo,hi=distance_range
    if not 0<=lo<=hi or min_separation<=0:raise ValueError('Invalid spawn range')
    n,agents=boxes.shape[:2]
    delta=boxes[...,:2]-goals[...,:2]
    if (delta.norm(dim=-1)<1e-6).any():raise ValueError('Spawn goal must differ from box center')
    rear_angle=torch.atan2(delta[...,1],delta[...,0])
    minimum=(sizes[...,:2].norm(dim=-1)/2+clearance).clamp_min(lo)
    if (minimum>hi).any():raise ValueError('Box clearance exceeds maximum spawn radius')
    positions=boxes.new_empty(n,agents,2)
    for agent in range(agents):
        pending=torch.arange(n,device=boxes.device)
        for _ in range(128):
            angle=rear_angle[pending,agent]+(torch.rand(len(pending),device=boxes.device)-.5)*torch.pi
            radius=minimum[pending,agent]+(hi-minimum[pending,agent])*torch.rand(len(pending),device=boxes.device)
            candidate=boxes[pending,agent,:2]+radius[:,None]*torch.stack((angle.cos(),angle.sin()),-1)
            valid=((candidate[:,None]-positions[pending,:agent]).norm(dim=-1)>=min_separation).all(-1)
            positions[pending[valid],agent]=candidate[valid];pending=pending[~valid]
            if not len(pending):break
        else:raise RuntimeError('Could not sample separated rear-half-plane spawns')
    return positions,torch.rand(n,agents,device=boxes.device)*(2*torch.pi)


def sample_groups(n, probability, device='cpu', preset='random_scenario'):
    """Shared-support sampling: shuffle humans, then consume 1 or 2..remaining."""
    order = torch.rand(n, 4, device=device).argsort(-1)
    groups = torch.full_like(order, -1)
    cursor = torch.zeros(n, dtype=torch.long, device=device)
    rows = torch.arange(n, device=device)
    rank = torch.arange(4, device=device)[None]
    for _ in range(4):
        remaining = 4-cursor
        coupled = (remaining >= 2) & (torch.rand(n, device=device) < probability)
        if preset == 'all_solo': coupled[:] = False
        count = torch.where(coupled,
            2+(torch.rand(n, device=device)*(remaining-1).clamp_min(0)).long(), 1)
        if preset == 'all_shared': count = remaining
        active = remaining > 0
        leader = order[rows, cursor.clamp_max(3)]
        chosen = active[:, None] & (rank >= cursor[:, None]) & (rank < (cursor+count)[:, None])
        ordered = groups.gather(1, order)
        groups.scatter_(1, order, torch.where(chosen, leader[:, None], ordered))
        cursor += count*active
    return groups


def group_layout(groups):
    same = groups[..., None] == groups[:, None, :]
    counts = same.sum(-1)
    ids = torch.arange(4, device=groups.device)
    ranks = (same & (ids[None, None, :] < ids[None, :, None])).sum(-1)
    # Four candidate leaders x group sizes1..4, after the eight ordinary boxes.
    physical = 8+4*groups+counts-1
    return physical, counts, ranks


def sample_asset_bank(n, spec, device='cpu'):
    lo = torch.tensor(spec['size_min'], device=device)
    hi = torch.tensor(spec['size_max'], device=device)
    sizes = lo+torch.rand(n, 24, 3, device=device)*(hi-lo)
    if spec.get('depth_at_least_height'): sizes[...,1:] = sizes[...,1:].sort(-1,descending=True).values
    members = torch.cat((torch.ones(8, device=device), torch.arange(1,5,device=device).repeat(4)))
    sizes[..., 0] *= members
    mass_lo, mass_hi = spec['mass_per_agent']
    masses = (mass_lo+torch.rand(n,24,device=device)*(mass_hi-mass_lo))*members
    return sizes, masses


def group_tokens(states, sizes, assignment, keep):
    """n strips per n-person physical group + four unassigned ordinary boxes."""
    from jointpush.partial.transforms import rotate
    _, count, rank = group_layout(assignment)
    parents = torch.cat((assignment, keep), -1)
    rows = torch.arange(len(states), device=states.device)[:, None]
    physical = states[rows, parents]
    token_sizes = sizes[rows, parents].clone()
    offsets = physical.new_zeros(len(states),8,3)
    offsets[:, :4, 0] = token_sizes[:, :4, 0]*((rank+.5)/count-.5)
    token_sizes[:, :4, 0] /= count
    delta = rotate(physical[...,3:7], offsets)
    tokens = physical.clone()
    tokens[..., :3] += delta
    tokens[..., 7:10] += torch.cross(tokens[...,10:13], delta, dim=-1)
    return tokens, token_sizes, offsets

