"""Per-pair reset mixture; all poses still start from loco."""
import torch


MIX226_DISTANCE_MIXTURE = dict(
    probabilities=[.2, .2, .6],
    leader_distance=[[1., 10.], [1., 10.], [1., 5.]],
    follower_distance=[[1., 10.], [1., 5.], [1., 2.5]],
    payload_distance=[[1., 5.], [1., 2.5], [1., 1.25]],
    ontop_follower_distance=[[1., 5.], [1., 2.5], [1., 1.25]],
    leader_goal_distance=[[1., 10.], [1., 10.], [1., 5.]],
    sit_original_distance=[1., 5.],
)


def sample_distance_modes(predecessor, mixture, force_near=False, force_short=False):
    """0=original/solo, 1=near, 2=short; sample once per episode pair."""
    if force_short:
        if force_near or mixture not in (MIX226_DISTANCE_MIXTURE,):
            raise ValueError('Short BEFORE mode requires the 20/20/60 mixture without force_near')
        return (predecessor >= 0).long() * 2
    if mixture not in (MIX226_DISTANCE_MIXTURE,):
        return sample_near_pairs(predecessor, mixture, force_near).long()
    modes = torch.zeros_like(predecessor)
    rows, followers = (predecessor >= 0).nonzero(as_tuple=True)
    if force_near:
        # Select the middle mode using this experiment's distance ranges.
        modes[rows, followers] = 1
    else:
        draw = torch.rand(len(rows), device=predecessor.device)
        p = mixture['probabilities']
        modes[rows, followers] = (draw >= p[0]).long() + (draw >= p[0]+p[1]).long()
    return modes


def sample_follower_modes(center, templates, modes, mixture=None, payload=False):
    if mixture == MIX226_DISTANCE_MIXTURE:
        bounds = center.new_tensor(mixture['payload_distance' if payload else 'follower_distance'])[modes]
        if not payload:
            bounds = torch.where((templates == 4)[:, None],
                center.new_tensor(mixture['ontop_follower_distance'])[modes], bounds)
            bounds = torch.where(((templates == 1) & (modes == 0))[:, None],
                center.new_tensor(mixture['sit_original_distance']), bounds)
        return sample_radial_xy(center, bounds)
    raise ValueError('Follower mode sampling requires mix226')


def sample_radial_xy(center, bounds):
    direction = torch.randn_like(center)
    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    radius = bounds[:, 0] + torch.rand_like(bounds[:, 0]) * (bounds[:, 1]-bounds[:, 0])
    return center + direction * radius[:, None]


def sample_before_goal_xy(human_xy, payload_xy, modes, mixture):
    """BEFORE leaders only: radial transport distance, retaining human exclusion."""
    bounds = payload_xy.new_tensor(mixture['leader_goal_distance'])[modes]
    target = sample_radial_xy(payload_xy, bounds)
    for _ in range(256):
        invalid = (target-human_xy).square().sum(-1) < 1.
        if not invalid.any():
            return target
        target[invalid] = sample_radial_xy(payload_xy[invalid], bounds[invalid])
    raise RuntimeError('BEFORE goal sampler exhausted rejection attempts')


def shorten_leader_xy(human_xy, payload_xy):
    """Map the original U[1,10] radius to U[1,5], keeping the payload fixed."""
    delta = human_xy - payload_xy
    radius = delta.norm(dim=-1, keepdim=True)
    return payload_xy + delta / radius.clamp_min(1e-6) * (1. + (radius - 1.) * (4./9.))


def sample_near_pairs(predecessor, mixture, force_near=False):
    near = torch.zeros_like(predecessor, dtype=torch.bool)
    if mixture is not None:
        if force_near:
            return predecessor >= 0
        rows, followers = (predecessor >= 0).nonzero(as_tuple=True)
        near[rows, followers] = (
            torch.rand(len(rows), device=predecessor.device) < mixture['near_probability']
        )
    return near


