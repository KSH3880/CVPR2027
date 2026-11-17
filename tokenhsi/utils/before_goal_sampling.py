import torch


DISTANCE_MIXTURE = dict(near_probability=.5, mode='goal_route_sum', base='goal_centered',
    sit_climb_distance=[1., 10.], ontop_goal_distance=[1., 5.],
    ontop_pickup_distance=[1., 5.], goal_min_human_distance=1.)


def route_distances(leader, support, goal, follower, payload, templates):
    first = (leader-support).norm(dim=-1)+(support-goal).norm(dim=-1)
    second = torch.where(templates == 4,
        (follower-payload).norm(dim=-1)+(payload-goal).norm(dim=-1),
        (follower-goal).norm(dim=-1))
    return first, second


def ontop_distances(limit):
    # Inverse CDF of U[1,5]+U[1,5], conditioned on the route upper bound.
    limit = limit.clamp(2., 10.)
    probability = torch.where(limit <= 6., (limit-2.).square()/32.,
                              1.-(10.-limit).square()/32.)
    draw = torch.rand_like(limit)*probability
    total = torch.where(draw <= .5, 2.+(32.*draw).sqrt(),
                        10.-(32.*(1.-draw)).sqrt())
    low, high = (total-5.).clamp_min(1.), (total-1.).clamp_max(5.)
    first = low+torch.rand_like(limit)*(high-low)
    return first, total-first


def sample_layout(goal, leader_route, templates, near, margin=.001):
    limit = torch.where(near, leader_route-margin, torch.full_like(leader_route, 10.))
    approach, pickup = ontop_distances(limit)
    angle = torch.rand_like(limit)*2*torch.pi
    direction = torch.stack((angle.cos(), angle.sin()), -1)
    payload = goal+approach[:, None]*direction
    minimum = DISTANCE_MIXTURE['goal_min_human_distance']+margin
    cosine = ((minimum**2-approach.square()-pickup.square())/(2*approach*pickup)).clamp(-1., 1.)
    relative = (2*torch.rand_like(limit)-1)*cosine.acos()
    heading = angle+relative
    top_root = payload+pickup[:, None]*torch.stack((heading.cos(), heading.sin()), -1)
    distance = 1.+torch.rand_like(limit)*(limit.clamp_min(1.)-1.)
    follower = goal+distance[:, None]*direction
    return torch.where((templates == 4)[:, None], top_root, follower), payload
