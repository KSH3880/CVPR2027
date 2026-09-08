"""Small, simulator-independent helpers for clean-scene observations."""

import torch


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
                           target_positions, env_origins):
    """Pack canonical ``[Human | Object | Target]`` GTA pose records.

    Input positions are simulator-world coordinates. Quaternions are local-to-world in
    xyzw order; the caller supplies heading-only Human quaternions and full Object
    quaternions. Targets inherit the corresponding owner Human heading.
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
    target_pose = torch.cat([
        target_positions - origin,
        human_heading_quaternions,
    ], dim=-1)
    return torch.cat([human_pose, object_pose, target_pose], dim=1)
