import torch
import math
from utils.edge_context_spec import HOLDING, AT
from utils.edge_ontop_spec import ON_TOP, batched
from utils.edge_interaction_spec import SIT, CLIMB
from env.tasks.multi_agent.edge_ontop_reward import (
    evaluate_ontop_edges,
    inner_xy_region_error,
    vertical_extent,
)


def evaluate_interaction_edges(
    hands, feet, roots, objects, sizes, goals, graph, config, char_h
):
    phi, diag = evaluate_ontop_edges(hands, roots, objects, sizes, goals, graph, config)
    n, m, o = (roots.shape[0], graph.num_agents, graph.num_objects)
    src = batched(graph.edge_src, n)
    dst = batched(graph.edge_dst, n)
    relation = batched(graph.edge_relation, n)
    valid = batched(graph.edge_valid, n)
    batch = torch.arange(n, device=roots.device)[:, None]
    human = roots[batch, src.clamp(0, m - 1)]
    support = objects[batch, (dst - m).clamp(0, o - 1)]
    support_size = sizes[batch, (dst - m).clamp(0, o - 1)]
    sit = (relation == SIT) & valid
    climb = (relation == CLIMB) & valid
    top_z = support[..., 2] + vertical_extent(support[..., 3:7], support_size / 2)
    sit_target = support[..., :3].clone()
    sit_target[..., 2] = top_z + float(config["sit"]["pelvis_clearance"])
    climb_target = support[..., :3].clone()
    climb_target[..., 2] = top_z + float(char_h)
    target = torch.where(sit[..., None], sit_target, climb_target)
    delta = human - target
    climb_scale = config.get("climb", {}).get(
        "near_distance_scale", config["sit"]["near_distance_scale"]
    )
    scale = torch.where(
        sit, float(config["sit"]["near_distance_scale"]), float(climb_scale)
    )
    interaction_phi = torch.exp(-scale * delta.square().sum(-1))
    feet_z = feet.mean(-2)[..., 2][batch, src.clamp(0, m - 1)]
    feet_error = (feet_z - top_z).abs()
    progress_target_xy = torch.where(
        sit[..., None], sit_target[..., :2], support[..., :2]
    )
    distance_xy = (human[..., :2] - progress_target_xy).norm(dim=-1)
    p = config["progress"]
    pinning_radius = torch.full_like(distance_xy, float(p.get("delta", 0.0)))
    if "bbox_buffers" in p:
        bbox_radius = support_size[..., :2].norm(dim=-1) / 2
        pinning_radius = bbox_radius + torch.where(
            sit, float(p["bbox_buffers"]["SIT"]), float(p["bbox_buffers"]["CLIMB"])
        )
    progress = 1 / (1 + (distance_xy - pinning_radius).clamp_min(0) / p["sigma"])
    mask = sit | climb
    phi = torch.where(mask, interaction_phi, phi)
    diag["progress"] = torch.where(mask, progress, diag["progress"])
    diag["z_error"] = torch.where(mask, delta[..., 2], diag["z_error"])
    diag["distance"] = torch.where(mask, delta.norm(dim=-1), diag["distance"])
    diag["distance_xy"] = torch.where(mask, distance_xy, diag["distance_xy"])
    diag["target"] = torch.where(mask[..., None], target, diag["target"])
    individual_feet_z = feet[batch, src.clamp(0, m - 1), :, 2]
    diag["z_feet"] = feet_z * climb
    diag["z_surface"] = top_z * climb
    diag["feet_height_error"] = feet_error * climb
    diag["feet_individual_max_error"] = (
        individual_feet_z - top_z[..., None]
    ).abs().amax(-1) * climb
    if "success_inner_margin_fraction" in config.get("climb", {}):
        region = inner_xy_region_error(
            human[..., :2],
            support,
            support_size,
            config["climb"]["success_inner_margin_fraction"],
        )
        diag["region_error"] = torch.where(climb, region, diag["region_error"])
    region = inner_xy_region_error(
        human[..., :2],
        support,
        support_size,
        config["sit"]["success_inner_margin_fraction"],
    )
    diag["region_error"] = torch.where(sit, region, diag["region_error"])
    return (phi, diag)


def interaction_own_success(phi, z_error, feet_error, graph, config, region_error=None):
    relation = batched(graph.edge_relation, phi.shape[0])
    valid = batched(graph.edge_valid, phi.shape[0])
    threshold = config["satisfaction_threshold"]
    success = (phi >= threshold) & (
        (relation == HOLDING) | (z_error.abs() <= config["success"]["at_z_tolerance"])
    )
    if "success_distance" in config["at"]:
        at_threshold = math.exp(
            -config["at"]["near_distance_scale"] * config["at"]["success_distance"] ** 2
        )
        success = torch.where(
            relation == AT,
            (phi >= at_threshold)
            & (z_error.abs() <= config["success"]["at_z_tolerance"]),
            success,
        )
    if region_error is None:
        raise ValueError("Stage1 plane success requires region_error")

    success = torch.where(
        relation == ON_TOP,
        (region_error <= 0) & (z_error.abs() <= config["ontop"]["z_tolerance"]),
        success,
    )
    success = torch.where(
        relation == CLIMB,
        (region_error <= 0)
        & (z_error.abs() <= config["climb"]["root_height_tolerance"])
        & (feet_error <= config["climb"]["feet_height_tolerance"]),
        success,
    )
    success = torch.where(
        relation == SIT,
        (region_error <= 0) & (z_error.abs() <= config["sit"]["z_tolerance"]),
        success,
    )
    return success & valid
