import torch

from utils.before_spec import sample_roles
from utils.sensor_stage2_spec import BRIDGE, CLIFF, bridge_roles


NAME = 'stage2_sensor_900_cliff_bridge'
BRANCHES = ('controlled', 'free', 'cooperative')
CONTROLLED, FREE, COOPERATIVE = range(3)


def sample_branches(count, spec, device, mode='training'):
    if mode != 'training':
        return torch.full((count,), BRANCHES.index(mode), device=device, dtype=torch.long)
    return torch.multinomial(torch.tensor([spec[k] for k in BRANCHES], device=device), count, True)


def sample_roles_for_scene(sizes, ranges, probabilities, scenes, branches, counts):
    n, m = sizes.shape[:2]
    device = sizes.device
    roles = torch.rand(n, m, device=device).argsort(-1)
    tasks = torch.randint(1, 5, (n, m), device=device)
    pre = torch.full_like(tasks, -1)
    groups = torch.arange(m, device=device).expand(n, -1).clone()
    push = torch.zeros_like(tasks, dtype=torch.bool)
    domain = torch.zeros(n, device=device, dtype=torch.long)
    for branch in range(3):
        rows = ((scenes == BRIDGE) & (branches == branch)).nonzero().flatten()
        if not len(rows):
            continue
        if branch == CONTROLLED:
            paired = rows[torch.rand(len(rows), device=device) < .5]
            if len(paired):
                pre[paired], tasks[paired] = sample_roles(sizes[paired], ranges,
                    'random_scenario', pairing_probability=1.)
            full = rows[(tasks[rows] == 4).all(-1)]
            while len(full):
                tasks[full] = torch.randint(1, 5, (len(full), m), device=device)
                full = full[(tasks[full] == 4).all(-1)]
        elif branch == FREE:
            pusher = roles[rows, 0]
            tasks[rows, pusher], push[rows, pusher] = 5, True
            pair = roles[rows, 1:3]
            local, picked = sample_roles(sizes[rows[:, None], pair], ranges,
                'random_scenario', pairing_probability=1.)
            mapped = pair.gather(1, local.clamp_min(0))
            pre[rows[:, None], pair] = torch.where(local >= 0, mapped, -1)
            tasks[rows[:, None], pair] = picked
        else:
            g, p, t, order, pushing = bridge_roles(sizes[rows], ranges, probabilities, counts[rows])
            groups[rows], pre[rows], tasks[rows], roles[rows], push[rows] = g, p, t, order, pushing
            single = rows[counts[rows] == 1]
            pair = roles[single, 2:]
            if len(single):
                local, picked = sample_roles(sizes[single[:, None], pair], ranges,
                    'random_scenario', pairing_probability=1.)
                mapped = pair.gather(1, local.clamp_min(0))
                pre[single[:, None], pair] = torch.where(local >= 0, mapped, -1)
                tasks[single[:, None], pair] = picked
            domain[rows] = (counts[rows] > 1).long()
    groups = torch.where(tasks == 5, groups, torch.arange(m, device=device))
    groups = torch.where(pre >= 0, pre, groups)
    return domain, groups, pre, tasks, roles, push


def bridge_schedule(distance, arrival, speed_range, minimum):
    speed = (distance/arrival).clamp(*speed_range)
    duration = (distance/speed).clamp_min(minimum)
    return duration, arrival-duration


def bridge_trajectory(start, goal, elapsed, duration):
    phase = (elapsed/duration).clamp(0, 1)
    blend = phase**3*(10+phase*(6*phase-15))
    rate = 30*phase**2*(1-phase)**2/duration
    return start+blend[:, None]*(goal-start), rate[:, None]*(goal-start)


def bridge_control(position, velocity, state, mass, spec):
    omega = 2*torch.pi*spec['tracking_frequency']
    acceleration = omega**2*(position-state[:, :3])+2*spec['damping_ratio']*omega*(velocity-state[:, 7:10])
    acceleration *= (spec['max_acceleration']/acceleration.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp_max(1)
    return mass[:, None]*acceleration
