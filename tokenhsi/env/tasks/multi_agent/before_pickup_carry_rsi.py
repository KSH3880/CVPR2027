import torch
from utils.before_pickup_carry_rsi import SKILLS, select_skills


def allocate(t):
    t._before_rsi_skill = torch.zeros_like(t._templates)
    t._before_rsi_attempts = torch.zeros_like(t._templates)
    t._before_rsi_counts = torch.zeros(3, 3, device=t.device)


def select(t, ids):
    mode = getattr(t.cfg['args'], 'before_pickup_carry_rsi', 'loco') if t._is_eval else 'mix'
    available = torch.zeros(len(ids), t.num_agents, 3, device=t.device, dtype=torch.bool)
    available[..., 0] = True
    if mode != 'loco':
        if not hasattr(t, '_size_rsi_cache'):
            from env.tasks.multi_agent.before_carry_rsi import BeforePickupCarryRsiCache
            t._size_rsi_cache = BeforePickupCarryRsiCache(t)
        for j, skill in enumerate(SKILLS[1:], 1):
            available[..., j] = t._size_rsi_cache.skill_available[skill][ids]
    eligible, requested, selected = select_skills(
        t._before_predecessor[ids], t._templates[ids], available, mode)
    t._before_rsi_skill[ids] = selected
    t._before_rsi_attempts[ids] = 0
    t._before_carry_mask[ids] = selected > 0
    t._before_rsi_counts[0] += torch.bincount(requested[eligible], minlength=3)
    t._before_rsi_counts[2] += torch.bincount(requested[eligible & (requested != selected)], minlength=3)


def screen(t, ids, bad):
    eligible = (t._before_predecessor[ids] >= 0) & (t._templates[ids] == 4)
    accepted = eligible & ~bad[:, None]
    t._before_rsi_counts[1] += torch.bincount(t._before_rsi_skill[ids][accepted], minlength=3)
    advanced = eligible & (t._before_rsi_skill[ids] > 0) & bad[:, None]
    t._before_rsi_attempts[ids] += advanced.long()
    exhausted = advanced & (t._before_rsi_attempts[ids] >=
        t.cfg['env']['before']['follower_pickup_carry_rsi']['placement_retry_limit'])
    rows, owners = exhausted.nonzero(as_tuple=True)
    es = ids[rows]
    t._before_rsi_counts[2] += torch.bincount(t._before_rsi_skill[es, owners], minlength=3)
    t._before_rsi_skill[es, owners] = 0
    t._before_carry_mask[es, owners] = False
    return bad


def diagnostics(t):
    counts = t._before_rsi_counts
    result = {}
    for j, skill in enumerate(SKILLS):
        for i, name in enumerate(('requested', 'executed', 'fallback')):
            result[f'sampling/before_pickup_carry/{skill}_{name}'] = counts[i, j].clone()
        result[f'sampling/before_pickup_carry/{skill}_fraction'] = counts[1, j]/counts[1].sum().clamp_min(1)
    counts.zero_()
    return result
