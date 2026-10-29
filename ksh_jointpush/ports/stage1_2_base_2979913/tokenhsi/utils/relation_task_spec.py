"""Stage1 reward schema and existing checkpoint metadata contract."""

from utils.edge_stage1_spec import validate_stage1_context_config
from utils.task_role_spec import TASKS, TASK_FIELDS


def validate_relation_config(config):
    validate_stage1_context_config(config)


def checkpoint_metadata(config):
    validate_relation_config(config)
    return dict(
        reward_mode=config["mode"],
        schema_version=config["schema_version"],
        relation_taxonomy=dict(zip(TASKS, range(4))),
        suffix_fields=list(TASK_FIELDS),
        packet_version=5,
        graph_record_width=5,
        context_dim_per_edge=0,
        context_fusion="task_embedding4_shared_type_projection_action_tokenhsi_platforms_v1",
        relation_reward_config=config,
    )


def check_checkpoint_metadata(weights, expected):
    saved = weights.get("relation_metadata")
    if saved is None:
        raise ValueError("Stage1 checkpoint requires relation metadata")
    for key in (
        "reward_mode",
        "schema_version",
        "relation_taxonomy",
        "suffix_fields",
        "context_dim_per_edge",
        "context_fusion",
        "packet_version",
        "graph_record_width",
    ):
        if saved.get(key) != expected[key]:
            raise ValueError("Checkpoint relation schema/architecture mismatch: " + key)

    def objective(config):
        from utils.checkpoint_compat import reward_config
        return {k: v for k, v in reward_config(config).items() if k != "diagnostics"}

    if objective(saved.get("relation_reward_config", {})) != objective(
        expected["relation_reward_config"]
    ):
        raise ValueError(
            "Checkpoint relation reward config differs (diagnostics-only overrides allowed)"
        )
