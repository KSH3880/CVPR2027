"""Spatial sampling for expanded Stage-2 evaluations."""
import math

import torch


def sample_clear_xy(centers, reach, occupied, clearance, generator=None):
    """Sample a disc while retaining the final candidate for outer reset validation."""
    def sample(rows):
        angle = torch.rand(len(rows), device=centers.device, generator=generator) * (2 * math.pi)
        radius = torch.rand(len(rows), device=centers.device, generator=generator).sqrt() * reach
        return centers[rows] + radius[:, None] * torch.stack((angle.cos(), angle.sin()), -1)

    candidates = sample(torch.arange(len(centers), device=centers.device))
    for _ in range(256):
        invalid = ((candidates[:, None] - occupied).norm(dim=-1) < clearance).any(-1)
        if not invalid.any():
            break
        rows = invalid.nonzero().flatten()
        candidates[rows] = sample(rows)
    return candidates


def sample_at_goals(graph, boxes, sizes, roots, goals, centers, reach):
    """Keep each AT goal clear of all boxes, its owner and earlier AT goals."""
    from utils.edge_context_spec import AT
    radii = sizes[..., :2].norm(dim=-1) / 2
    result = goals.clone()
    previous_xy, previous_radius, previous_valid = [], [], []
    for edge in range(graph.edge_valid.shape[1]):
        active = graph.edge_valid[:, edge] & (graph.edge_relation[:, edge] == AT)
        rows = active.nonzero().flatten()
        if not len(rows):
            continue
        source = graph.edge_src[rows, edge] - graph.num_agents
        target = graph.edge_dst[rows, edge] - graph.num_agents - graph.num_objects
        owner = graph.edge_owner[rows, edge]
        radius = radii[rows, source]
        clearance = radius[:, None] + radii[rows] + .25
        clearance[torch.arange(len(rows), device=boxes.device), source] = 1.
        occupied = torch.cat((boxes[rows, :, :2], roots[rows, owner, :2][:, None]), 1)
        clearance = torch.cat((clearance, torch.ones(len(rows), 1, device=boxes.device)), 1)
        if previous_xy:
            occupied = torch.cat((occupied, torch.stack(previous_xy, 1)[rows]), 1)
            other_radius = torch.stack(previous_radius, 1)[rows]
            valid = torch.stack(previous_valid, 1)[rows]
            clearance = torch.cat((clearance,
                (radius[:, None] + other_radius + .35) * valid), 1)
        xy = sample_clear_xy(centers[rows], reach, occupied, clearance)
        result[rows, target, :2] = xy
        stored_xy = centers.clone()
        stored_radius = radii.new_zeros(len(boxes))
        stored_xy[rows], stored_radius[rows] = xy, radius
        previous_xy.append(stored_xy)
        previous_radius.append(stored_radius)
        previous_valid.append(active)
    return result
