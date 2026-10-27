import torch

from utils.jointpush_geometry import push_rsi_hand_targets, rotate
from utils.stage2_jointpush import region_states


def sample_push_initialization(groups, templates, probability=.5):
    draws = torch.rand(groups.shape,device=groups.device) < probability
    return draws.gather(1,groups) & (templates == 5)


def hand_targets(boxes, sizes, goals, members, regions, contact):
    tokens, cells = region_states(boxes,sizes,members,regions)
    forward = torch.zeros_like(goals)
    forward[...,1] = 1.
    region_goals = tokens[...,:3]+rotate(tokens[...,3:7],forward)
    target_goals = torch.where((members > 1)[...,None],region_goals,goals)
    result = push_rsi_hand_targets(tokens.reshape(-1,1,13),target_goals.reshape(-1,1,3),
        cells.reshape(-1,1,3),dict(push=contact,contact_at=contact))
    return result.reshape(*boxes.shape[:-1],2,3)


def coupled_body_overlap(body_states, geometry, groups, interaction):
    body = body_states[:,:,geometry['ids']]
    points = body[...,:3]+rotate(body[...,3:7],geometry['offsets'].expand(*body.shape[:-1],3))
    radii = geometry['radii']
    bad = torch.zeros(len(groups),device=groups.device,dtype=torch.bool)
    for a in range(groups.shape[1]):
        for b in range(a):
            selected = (groups[:,a] == groups[:,b]) & interaction[:,a]
            rows = selected.nonzero().flatten()
            if not len(rows):
                continue
            distance = (points[rows,a,:,None]-points[rows,b,None]).square().sum(-1)
            bad[rows] |= (distance < (radii[:,None]+radii[None]+.005).square()).any((1,2))
    return bad
