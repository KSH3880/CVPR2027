import torch


SKILLS = ('loco', 'pickUp', 'carryWith')
SPEC = dict(probabilities=[.4, .3, .3], skills=list(SKILLS),
            placement_retry_limit=8, fallback='loco', frames='uniform_valid')


def select_skills(predecessor, templates, available, mode='mix'):
    eligible = (predecessor >= 0) & (templates == 4)
    requested = torch.zeros_like(templates)
    if mode == 'mix' and eligible.any():
        weights = torch.tensor(SPEC['probabilities'], device=templates.device)
        requested[eligible] = torch.multinomial(weights, int(eligible.sum()), replacement=True)
    elif mode in ('pickup', 'carry'):
        requested[eligible] = 1 if mode == 'pickup' else 2
    elif mode not in ('loco', 'mix'):
        raise ValueError('Unknown follower pickup/carry RSI mode')
    valid = available.gather(-1, requested[..., None]).squeeze(-1)
    selected = torch.where(valid, requested, 0)
    return eligible, requested, selected
