import torch


from utils.edge_interaction_spec import SIT, CLIMB


def sample_ontop_putdown_times(motion_lib, motion_ids, phase_range):
    lo, hi = phase_range
    phase = lo + torch.rand(len(motion_ids), device=motion_ids.device) * (hi - lo)
    return motion_lib.get_motion_length(motion_ids) * phase


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
    return box_sizes[..., 2].topk(2, dim=-1).values.sum(-1)


def mix_late_climb_rsi_times(motion_lib, motion_ids, times, spec):
    selected = torch.rand(len(times), device=times.device) < spec["late_fraction"]
    if selected.any():
        lo, hi = spec["phase_range"]
        phase = lo + torch.rand(int(selected.sum()), device=times.device) * (hi - lo)
        times[selected] = motion_lib.get_motion_length(motion_ids[selected]) * phase
    return times


CURRICULUM_VARIANTS = ("scenario_independent_stage1_unified_size_rsi_task_embedding",)
