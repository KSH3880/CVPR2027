import torch
from utils.stage1_sampling import sample_source_xy


DISTANCE_MIXTURE = dict(near_probability=.5, mode='relative_sum', base='original')


def pair_distances(leader, support, follower, payload, templates):
    first = (leader-support).norm(dim=-1)
    second = (follower-support).norm(dim=-1)
    second += torch.where(templates == 4, (follower-payload).norm(dim=-1), 0.)
    return first, second


def sample_layout(leader, support, follower, payload, templates, near, advanced):
    support = support.clone()
    offset = payload-follower
    follower = sample_source_xy(support, templates)
    payload = torch.where(advanced[:, None], follower+offset,
                          sample_source_xy(follower, torch.full_like(templates, 4)))
    pending = near.nonzero(as_tuple=True)[0]
    for _ in range(256):
        if not len(pending):
            return support, follower, payload
        r = pending
        support[r] = sample_source_xy(leader[r], torch.full_like(templates[r], 3))
        follower[r] = sample_source_xy(support[r], templates[r])
        payload[r] = torch.where(advanced[r, None], follower[r]+offset[r],
            sample_source_xy(follower[r], torch.full_like(templates[r], 4)))
        first, second = pair_distances(leader[r], support[r], follower[r], payload[r], templates[r])
        pending = r[second >= first]
    raise RuntimeError('BEFORE relative-distance rejection sampler exhausted')
