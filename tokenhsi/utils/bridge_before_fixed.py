import torch

from utils.before_spec import eligible_families


NAME = 'stage2_sensor_900_bridge_before_fixed'
VARIANT = 'bridge_before_fixed_v1'
PUSHERS = 2
FOLLOWER_TASKS = (1, 2, 4)


def validate(scene):
    if scene['mixture'] != dict(cliff=0., bridge=1., room=0.):
        raise ValueError('Fixed BEFORE Bridge trains only the Bridge scene')
    if scene['bridge']['branches'] != dict(controlled=0., free=0., cooperative=1.):
        raise ValueError('Fixed BEFORE Bridge uses only the cooperative branch')
    training = scene['bridge'].get('before_training')
    if training is not None:
        progress = training['bridge_center_progress']
        if not 0 <= training['same_platform_probability'] <= 1 or progress['weight'] < 0 or progress['sigma'] <= 0:
            raise ValueError('Invalid fixed BEFORE training mixture or bridge progress settings')


def bridge_center_progress(payload_xy, bridge_xy, direction, passed, crossing, sigma):
    offset = payload_xy-bridge_xy
    passed = (passed | ((offset*direction).sum(-1) >= 0)) & crossing
    reward = 1/(1+offset.norm(dim=-1)/sigma)
    reward = torch.where(passed, 1., reward)*crossing
    return reward, passed


def fixed_roles(tasks, groups, roles, crosser_sizes, ranges):
    """Crosser (column 1) leads with AT; the non-pusher in column 3 follows with SIT, CLIMB or ON_TOP."""
    n, m = tasks.shape
    rows = torch.arange(n, device=tasks.device)
    crosser, follower = roles[:, 1], roles[:, 3]
    family = torch.multinomial(eligible_families(crosser_sizes, ranges).float(), 1).flatten()
    tasks, pre = tasks.clone(), torch.full_like(tasks, -1)
    tasks[rows, crosser] = 3
    tasks[rows, follower] = tasks.new_tensor(FOLLOWER_TASKS)[family]
    pre[rows, follower] = crosser
    groups = torch.where(tasks == 5, groups, torch.arange(m, device=tasks.device))
    groups = torch.where(pre >= 0, pre, groups)
    return groups, pre, tasks
