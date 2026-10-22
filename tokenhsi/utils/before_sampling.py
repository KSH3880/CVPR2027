"""Per-pair reset mixture; all poses still start from loco."""
import torch

from utils.stage1_sampling import sample_source_xy


DISTANCE_MIXTURE = dict(near_probability=.5, follower_distance=[1., 5.],
                        payload_distance=[1., 5.])
THREE_WAY_DISTANCE_MIXTURE = dict(
    probabilities=[.3, .3, .4],
    leader_distance=[[1., 10.], [1., 10.], [1., 5.]],
    follower_distance=[[1., 10.], [1., 5.], [1., 2.5]],
    payload_distance=[[1., 10.], [1., 5.], [1., 2.5]],
    sit_original_distance=[1., 5.],
)


def sample_distance_modes(predecessor, mixture, force_near=False):
    """0=original/solo, 1=near, 2=short; sample once per episode pair."""
    if mixture != THREE_WAY_DISTANCE_MIXTURE:
        return sample_near_pairs(predecessor, mixture, force_near).long()
    modes = torch.zeros_like(predecessor)
    rows, followers = (predecessor >= 0).nonzero(as_tuple=True)
    if force_near:
        # Match mix50's existing viewer flag: select the 1--5 m follower mode.
        modes[rows, followers] = 1
    else:
        draw = torch.rand(len(rows), device=predecessor.device)
        p = mixture['probabilities']
        modes[rows, followers] = (draw >= p[0]).long() + (draw >= p[0]+p[1]).long()
    return modes


def sample_follower_modes(center, templates, modes):
    if not (modes == 2).any():
        return sample_follower_xy(center, templates, modes > 0)
    maximum = torch.where((modes > 0) | (templates == 1), 5., 10.)
    maximum = torch.where(modes == 2, 2.5, maximum)
    direction = torch.randn_like(center)
    direction /= direction.norm(dim=-1, keepdim=True)
    radius = 1. + torch.rand_like(maximum) * (maximum - 1.)
    return center + direction * radius[:, None]


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


def sample_follower_xy(center, templates, near):
    if not near.any():
        return sample_source_xy(center, templates)
    maximum = torch.where(near | (templates == 1), 5., 10.)
    direction = torch.randn_like(center)
    direction /= direction.norm(dim=-1, keepdim=True)
    radius = 1. + torch.rand_like(maximum) * (maximum - 1.)
    return center + direction * radius[:, None]
