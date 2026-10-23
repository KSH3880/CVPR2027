"""Stage1 reward validation, interaction grounding and graph compilation."""

import torch
from utils.task_role_spec import STAGE1_VARIANTS, STAGE1_V2_VARIANT
from utils.edge_interaction_spec import SIT, CLIMB

STAGE1_CONTEXT_MODE = "state_relation_edge_stage1_v1"


def standalone_interaction_owners(graph, env_ids):
    """Return agents whose SIT/CLIMB edge targets their own box, not O_X."""
    relation = graph.edge_relation[env_ids][:, None, :]
    destination = graph.edge_dst[env_ids][:, None, :]
    owner = graph.edge_owner[env_ids][:, None, :]
    valid = graph.edge_valid[env_ids][:, None, :]
    agents = torch.arange(graph.num_agents, device=relation.device)[None, :, None]
    return (
        valid
        & ((relation == SIT) | (relation == CLIMB))
        & (owner == agents)
        & (destination == graph.num_agents + agents)
    ).any(-1)


def ground_standalone_interaction_targets(
    graph,
    env_ids,
    assignments,
    box_states,
    box_sizes,
    platform_pos=None,
    platform_default_pos=None,
):
    """Floor only standalone SIT/CLIMB targets and deactivate their source platforms."""
    mask = standalone_interaction_owners(graph, env_ids)
    local_env, agent = mask.nonzero(as_tuple=True)
    if not len(local_env):
        return
    world_env = env_ids[local_env]
    physical_box = assignments[world_env, agent]
    box_states[world_env, physical_box, 2] = box_sizes[world_env, physical_box, 2] / 2
    if platform_pos is not None:
        platform_pos[world_env, agent] = platform_default_pos[world_env, agent]


def max_stage1_stack_height(box_sizes):
    """Upper bound for the Stage-1 O_i -> O_X two-box stack in each environment."""
    return box_sizes[..., 2].topk(min(2, box_sizes.shape[-2]), dim=-1).values.sum(-1)


_REWARD_CONTRACT = {
    "mode": "state_relation_edge_stage1_v1",
    "schema_version": 9,
    "state_reward_weight": 0.2,
    "progress_reward_weight": 0.2,
    "success_reward_weight": 0.2,
    "satisfaction_threshold": 0.9,
    "holding": {"hand_distance_scale": 10.0},
    "ontop": {
        "state_definition": "centered_stack_world_z",
        "near_distance_scale": 10.0,
        "z_tolerance": 0.001,
        "vertical_extent": "rotated_bbox",
        "success_inner_margin_fraction": 0.1,
    },
    "sit": {
        "state_definition": "box_top_plus_pelvis_clearance",
        "near_distance_scale": 10.0,
        "pelvis_clearance": 0.12,
        "success_inner_margin_fraction": 0.1,
        "z_tolerance": 0.07,
    },
    "progress": {
        "kind": "distance",
        "sigma": 1.0,
        "bbox_buffers": {
            "HOLDING": 0.2,
            "SIT": 0.3,
            "CLIMB": 0.3,
            "AT": 0.1,
            "ON_TOP": 0.1,
        },
    },
    "success": {
        "at_z_tolerance": 0.001,
        "saturation": "paired_placement_current_success",
        "terminate_when_all_subgoals_done": False,
    },
    "task_sharing": {"self": 1.0, "teammate": 0.0},
    "edge_aggregation": "self_sum",
    "hard_skill_training": {
        "near_start": {
            "probability_start": 0.0,
            "probability_end": 0.0,
            "anneal_steps": 300000,
            "at_xy_range": [0.75, 1.5],
            "ontop_xy_range": [1.1, 1.8],
        },
        "climb_rsi": {"late_fraction": 0.0, "phase_range": [0.65, 0.98]},
    },
    "observation": {
        "graph_packet_fields": ["valid", "task", "actor", "payload", "target"]
    },
}


def validate_stage1_context_config(config):
    variant = config.get("stage1_variant")
    if variant not in STAGE1_VARIANTS:
        raise ValueError("Only stage1 and stage1_v2 reward variants are supported")
    expected = dict(_REWARD_CONTRACT)
    expected["stage1_variant"] = variant
    expected["diagnostics"] = config.get("diagnostics")
    if "task_types" in config:
        from utils.task_role_spec import TASKS
        expected["task_types"] = list(TASKS)
    expected["at"] = dict(state_definition="box_near", near_distance_scale=10.0)
    if variant == STAGE1_V2_VARIANT:
        expected["at"]["success_distance"] = 0.2
    expected["climb"] = dict(
        state_definition="root_target",
        near_distance_scale=10.0,
        target_height="rotated_bbox_top_plus_char_h",
        success_inner_margin_fraction=0.1 if variant == STAGE1_V2_VARIANT else 0.05,
        root_height_tolerance=0.2,
        feet_height_tolerance=0.07,
    )
    if config != expected:
        raise ValueError("Unsupported Stage1 reward configuration")
    diagnostics = config["diagnostics"]
    if not isinstance(diagnostics, dict) or set(diagnostics) - {
        "enabled",
        "sample_envs",
        "log_interval",
    }:
        raise ValueError("Unsupported diagnostics fields")
    if type(diagnostics.get("enabled", True)) is not bool:
        raise ValueError("diagnostics.enabled must be boolean")
    for key, default, minimum in [("sample_envs", 2, 0), ("log_interval", 30, 1)]:
        if (
            type(diagnostics.get(key, default)) is not int
            or diagnostics.get(key, default) < minimum
        ):
            raise ValueError("Invalid diagnostics." + key)


def compile_stage1_graph(spec, m, o, device=None):
    if spec.get("sampler") == "shared_support_groups":
        from utils.shared_support import compile_graph
        return compile_graph(spec, m, o, device)
    if spec.get("sampler") == "stage2_random_groups":
        from utils.stage2_spec import sample_graph
        probabilities = torch.ones(1, m, 5, device=device)
        probabilities[..., 0] = 0
        from dataclasses import replace, fields
        graph = sample_graph(probabilities, spec)[0]
        return replace(graph, **{field.name: getattr(graph, field.name)[0]
                                 for field in fields(graph) if isinstance(getattr(graph, field.name), torch.Tensor)})
    from utils.edge_scenario_spec import compile_graph

    return compile_graph(spec, m, o, device)


def sample_graph(
    n, spec, device="cpu", preset="random_scenario", role_swap=False, generator=None
):
    from utils.edge_scenario_spec import sample_graph as sample

    return sample(n, spec, device, preset, role_swap, generator)
