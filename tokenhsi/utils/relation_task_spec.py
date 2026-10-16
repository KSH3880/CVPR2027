from utils.task_role_spec import TASKS, TASK_FIELDS


REWARD = {
    "mode": "state_relation_edge_stage2_v1",
    "schema_version": 10,
    "stage1_variant": "scenario_independent_stage1_unified_size_rsi_task_embedding",
    "state_reward_weight": 0.2,
    "progress_reward_weight": 0.2,
    "success_reward_weight": 0.2,
    "satisfaction_threshold": 0.9,
    "holding": {"hand_distance_scale": 10.0},
    "at": {"state_definition": "box_near", "near_distance_scale": 10.0},
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
    "climb": {
        "state_definition": "root_target",
        "near_distance_scale": 10.0,
        "target_height": "rotated_bbox_top_plus_char_h",
        "success_inner_margin_fraction": 0.05,
        "root_height_tolerance": 0.2,
        "feet_height_tolerance": 0.07,
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
        "saturation": "none",
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
    "joint_carry": {
        "anchor_inset": 0.15,
        "placement_gate": "opposite_current_holding",
        "collision": {"mode": "cpa", "ttc_discount": 0.99, "release_buffer": 1.0},
        "independent": {
            "environment_fraction": 0.2,
            "placement_gate": "own_current_holding",
            "rsi": "stage1_size_aware",
        },
    },
}


def validate_relation_config(config):
    expected = REWARD
    if "at_distance_tolerance" in config["success"]:
        expected = dict(REWARD, success=dict(REWARD["success"], at_distance_tolerance=0.2))
    if {k: v for k, v in config.items() if k != "diagnostics"} != expected:
        raise ValueError("Unsupported SA joint-carry reward contract")


def checkpoint_metadata(config):
    return dict(
        reward_mode=config["mode"],
        schema_version=config["schema_version"],
        relation_taxonomy=dict(zip(TASKS, range(4))),
        suffix_fields=list(TASK_FIELDS),
        packet_version=5,
        graph_record_width=5,
        context_dim_per_edge=0,
        context_fusion="task_category_embedding64_type_pair_projection",
        relation_reward_config=config,
    )


def check_checkpoint_metadata(weights, expected):
    saved = weights.get("relation_metadata", {})
    for key in (
        "reward_mode",
        "schema_version",
        "relation_taxonomy",
        "suffix_fields",
        "packet_version",
        "graph_record_width",
        "context_dim_per_edge",
        "context_fusion",
    ):
        if saved.get(key) != expected[key]:
            raise ValueError("Checkpoint relation schema mismatch: " + key)

    def objective(config):
        return {k: v for k, v in config.items() if k != "diagnostics"}

    if objective(saved.get("relation_reward_config", {})) != objective(
        expected["relation_reward_config"]
    ):
        raise ValueError("Checkpoint reward contract mismatch")
