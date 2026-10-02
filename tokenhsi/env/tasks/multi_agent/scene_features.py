"""Simulator-independent helpers for clean-scene observations and viewer markers."""

import torch

SHARED_TASK_COLOR = (1.0, 0.85, 0.15)


def at_goal_marker_positions(target_positions, graph):
    """Show each AT destination slot, irrespective of the edge owner's index."""
    from utils.edge_context_spec import AT
    from utils.edge_ontop_spec import batched

    batch, goals, _ = target_positions.shape
    dst = batched(graph.edge_dst, batch) - graph.num_agents - graph.num_objects
    active_edges = (batched(graph.edge_valid, batch) &
                    (batched(graph.edge_relation, batch) == AT) &
                    (dst >= 0) & (dst < goals))
    counts = torch.zeros(batch, goals, dtype=torch.long, device=target_positions.device)
    counts.scatter_add_(1, dst.clamp(0, goals - 1), active_edges.long())
    markers = target_positions.clone()
    markers[..., 2] = torch.where(counts > 0, target_positions[..., 2], 20.)
    return markers


def object_task_owners(graph, env_id):
    """Map logical objects to distinct owners using every valid task endpoint.

    ON_TOP supports count alongside carried sources and SIT/CLIMB destinations.
    Human and goal nodes do not contribute to object sharing.
    """
    row = lambda value: value if value.ndim == 1 else value[env_id]
    valid = row(graph.edge_valid)
    src = row(graph.edge_src)[valid].detach().cpu().tolist()
    dst = row(graph.edge_dst)[valid].detach().cpu().tolist()
    owners = row(graph.edge_owner)[valid].detach().cpu().tolist()
    result = {}
    for source, target, owner in zip(src, dst, owners):
        for entity in (source, target):
            obj = entity - graph.num_agents
            if 0 <= obj < graph.num_objects:
                result.setdefault(obj, set()).add(owner)
    return result


def task_target_owners(graph, env_id):
    """Group distinct owners by both action and destination for marker colors."""
    row = lambda value: value if value.ndim == 1 else value[env_id]
    valid = row(graph.edge_valid)
    relations = row(graph.edge_relation)[valid].detach().cpu().tolist()
    targets = row(graph.edge_dst)[valid].detach().cpu().tolist()
    owners = row(graph.edge_owner)[valid].detach().cpu().tolist()
    result = {}
    for relation, target, owner in zip(relations, targets, owners):
        result.setdefault((relation, target), set()).add(owner)
    return result


def task_marker_vertices(relation, target):
    """SIT: horizontal ring; ON_TOP: wire cube; CLIMB: three-axis cross."""
    import numpy as np
    from utils.edge_interaction_spec import SIT, CLIMB
    from utils.edge_ontop_spec import ON_TOP

    target = np.asarray(target, dtype=np.float32)
    if relation == SIT:
        angles = np.linspace(0., 2. * np.pi, 25)
        points = target + np.stack((.18 * np.cos(angles), .18 * np.sin(angles),
                                    np.zeros_like(angles)), axis=-1)
        vertices = np.stack((points[:-1], points[1:]), axis=1).reshape(-1, 3)
    elif relation == ON_TOP:
        corners = np.array([[x, y, z] for x in (-.12, .12)
                            for y in (-.12, .12) for z in (-.12, .12)])
        edges = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4)
                 if i < (i ^ bit)]
        vertices = target + corners[np.asarray(edges)].reshape(-1, 3)
    elif relation == CLIMB:
        offsets = np.eye(3) * .12
        vertices = np.stack((target - offsets, target + offsets), axis=1).reshape(-1, 3)
    else:
        raise ValueError('Unsupported task marker relation: ' + str(relation))
    return vertices.astype(np.float32)


def scenario_neutral_targets(env_origins, env_ids, num_targets):
    """Place inactive scenario targets at each environment's local origin."""
    return env_origins[env_ids].unsqueeze(1).expand(-1, num_targets, -1)


def build_env_local_position_features(world_positions, env_origins, arena_scale):
    """Return shared scene positions with deterministic XY scaling.

    All entity types use the same convention: subtract the Isaac Gym environment
    origin, divide X/Y by the arena scale, and keep Z in env-local metres.

    Args:
        world_positions: ``(B, K, 3)`` entity positions in simulator world space.
        env_origins: ``(B, 3)`` Isaac Gym environment origins.
        arena_scale: positive scalar shared by every entity type.
    """
    if world_positions.ndim != 3 or world_positions.shape[-1] != 3:
        raise ValueError("world_positions must have shape (B, K, 3)")
    if env_origins.ndim != 2 or env_origins.shape[-1] != 3:
        raise ValueError("env_origins must have shape (B, 3)")
    if world_positions.shape[0] != env_origins.shape[0]:
        raise ValueError("world_positions and env_origins must have the same batch size")
    if arena_scale <= 0:
        raise ValueError("arena_scale must be positive")

    local = world_positions - env_origins.unsqueeze(1)
    return torch.cat([local[..., 0:2] / arena_scale, local[..., 2:3]], dim=-1)


def build_gta_pose_records(human_positions, human_heading_quaternions,
                           object_positions, object_quaternions,
                           target_positions, env_origins, goal_rotation="human_heading"):
    """Pack canonical ``[Human | Object | Target]`` GTA pose records.

    Input positions are simulator-world coordinates. Quaternions are local-to-world in
    xyzw order; the caller supplies heading-only Human quaternions and full Object
    quaternions. Targets use legacy Human headings or a fixed identity frame.
    """
    position_groups = (human_positions, object_positions, target_positions)
    if any(x.ndim != 3 or x.shape[-1] != 3 for x in position_groups):
        raise ValueError("entity positions must have shape (B, K, 3)")
    if human_heading_quaternions.ndim != 3 or human_heading_quaternions.shape[-1] != 4:
        raise ValueError("human heading quaternions must have shape (B, M, 4)")
    if object_quaternions.ndim != 3 or object_quaternions.shape[-1] != 4:
        raise ValueError("object quaternions must have shape (B, O, 4)")
    if env_origins.ndim != 2 or env_origins.shape[-1] != 3:
        raise ValueError("env_origins must have shape (B, 3)")

    batch = env_origins.shape[0]
    if any(x.shape[0] != batch for x in position_groups):
        raise ValueError("all entity positions must match the origin batch size")
    if human_heading_quaternions.shape[:2] != human_positions.shape[:2]:
        raise ValueError("Human positions and headings must have matching slots")
    if object_quaternions.shape[:2] != object_positions.shape[:2]:
        raise ValueError("Object positions and quaternions must have matching slots")
    if target_positions.shape[:2] != human_positions.shape[:2]:
        raise ValueError("each Target must have one owner Human")

    origin = env_origins.unsqueeze(1)
    human_pose = torch.cat([
        human_positions - origin,
        human_heading_quaternions,
    ], dim=-1)
    object_pose = torch.cat([
        object_positions - origin,
        object_quaternions,
    ], dim=-1)
    if goal_rotation not in ('human_heading', 'identity'):
        raise ValueError('Unknown goal rotation: ' + goal_rotation)
    goal_quat = human_heading_quaternions
    if goal_rotation == 'identity':
        goal_quat = torch.zeros_like(human_heading_quaternions)
        goal_quat[..., 3] = 1.
    target_pose = torch.cat([
        target_positions - origin,
        goal_quat,
    ], dim=-1)
    return torch.cat([human_pose, object_pose, target_pose], dim=1)
