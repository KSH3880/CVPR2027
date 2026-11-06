import torch

from utils.before_spec import sample_roles
from utils.coupled_push_spec import group_sizes


def sample_groups(sizes, ranges, probabilities, preset):
    count, people = sizes.shape[:2]
    device = sizes.device
    weights = sizes.new_tensor(list(probabilities.values()))
    groups = torch.arange(people, device=device)[None].expand(count, -1).clone()
    predecessor = torch.full_like(groups, -1)
    templates = torch.multinomial(weights, count*people, replacement=True).reshape_as(groups)
    pending = torch.arange(count, device=device)
    for _ in range(1024):
        order = torch.rand(len(pending), people, device=device).argsort(-1)
        cursor = torch.zeros(len(pending), dtype=torch.long, device=device)
        rows = torch.arange(len(pending), device=device)
        rank = torch.arange(people, device=device)[None]
        for _ in range(people):
            remaining = people-cursor
            choices = sizes.new_ones(len(pending), 3)
            if preset == 'before' or preset.startswith('before_'):
                choices[:, 2] = 0
            elif preset == 'coupled':
                choices[:, 1] = 0
            elif preset not in ('random_scenario', 'mixed'):
                choices[:, 1:] = 0
            if preset == 'all_coupled':
                choices[:, 2] = 1
                choices[remaining >= 2, 0] = 0
            choices[remaining < 2, 1:] = 0
            kind = torch.multinomial(choices, 1).flatten()
            size = torch.where(kind == 2,
                2+(torch.rand(len(pending), device=device)*(remaining.clamp_max(4)-1).clamp_min(0)).long(),
                torch.where(kind == 1, 2, 1))
            leader = order[rows, cursor.clamp_max(people-1)]
            selected = (remaining > 0)[:, None] & (rank >= cursor[:, None]) & (rank < (cursor+size)[:, None])
            ids, owner = pending[:, None].expand_as(order)[selected], order[selected]
            groups[ids, owner] = leader[:, None].expand_as(order)[selected]
            templates[ids, owner] = torch.where(kind[:, None].expand_as(order)[selected] == 2,
                                               5, templates[ids, owner])
            pair_rows = ((remaining >= 2) & (kind == 1)).nonzero(as_tuple=True)[0]
            if len(pair_rows):
                pair = order[pair_rows].gather(1, cursor[pair_rows, None]+torch.arange(2, device=device))
                env = pending[pair_rows]
                pre, tasks = sample_roles(sizes[env[:, None], pair], ranges,
                    preset if preset.startswith('before_') else 'random_scenario',
                    solo_probabilities=list(probabilities.values())[1:], pairing_probability=1.)
                parent = pair.gather(1, pre.clamp_min(0))
                predecessor[env[:, None], pair] = torch.where(pre >= 0, parent, -1)
                groups[env[:, None], pair] = torch.where(pre >= 0, parent, pair)
                templates[env[:, None], pair] = tasks
            cursor += size*(remaining > 0)
        has_before = (predecessor[pending] >= 0).any(-1)
        has_coupled = ((group_sizes(groups[pending]) > 1) & (templates[pending] == 5)).any(-1)
        if preset != 'mixed':
            break
        pending = pending[~(has_before & has_coupled)]
        if not len(pending):
            break
        groups[pending] = torch.arange(people, device=device)
        predecessor[pending] = -1
        templates[pending] = torch.multinomial(weights, len(pending)*people, replacement=True).reshape(-1, people)
    else:
        raise RuntimeError('Could not sample both BEFORE and COUPLED groups')
    forced = dict(sit=1, climb=2, holding_at=3, holding_ontop=4, push=5, all_coupled=5)
    if preset in forced:
        templates.fill_(forced[preset])
    before = (predecessor >= 0).any(-1)
    coupled = ((group_sizes(groups) > 1) & (templates == 5)).any(-1)
    domain = torch.where(before & coupled, 2, torch.where(coupled, 1, 0))
    return domain, groups, predecessor, templates
