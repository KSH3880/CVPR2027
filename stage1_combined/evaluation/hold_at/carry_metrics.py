"""TokenHSI carry criteria re-scored on a shared 30-second first episode."""
import numpy as np

PROTOCOL = 'tokenhsi_carry_20s_upright_on_common_rollout_v1'
# The native episodeLength=600 loop times out at progress=599.
ACTION_LIMIT = 599
UPRIGHT_DOT = .86615


def first_success_steps(mask):
    mask = np.asarray(mask, dtype=bool)
    return np.where(mask.any(0), mask.argmax(0) + 1, 0)


def score(data, up_facing, dtype=np.float64):
    limit = min(ACTION_LIMIT, len(data['objects']))
    objects = np.asarray(data['objects'][:limit], dtype=dtype)
    goals = np.asarray(data['goals'][:limit], dtype=dtype)
    up = np.asarray(up_facing, dtype=dtype)[None]
    active = np.asarray(data['active'][:limit], dtype=bool)
    if not limit or objects.shape[:2] != active.shape or up.shape != (1, objects.shape[1], 3):
        raise ValueError('Invalid carry trace or per-environment up axes')
    distance = np.linalg.norm(objects[..., :3] - goals, axis=-1)
    q = objects[..., 3:7]
    w, xyz = q[..., 3:4], q[..., :3]
    # Same unit-quaternion rotation expression as isaacgym.torch_utils.quat_rotate.
    rotated = up * (2 * w ** 2 - 1) + np.cross(xyz, up) * w * 2
    rotated += xyz * (xyz * up).sum(-1, keepdims=True) * 2
    dot = (rotated * up).sum(-1)
    hit = (distance <= .20) & (dot >= UPRIGHT_DOT) & active
    success = hit.any(0)
    minimum = np.where(hit, distance, np.inf).min(0)
    return dict(protocol=PROTOCOL, horizon_action_steps=ACTION_LIMIT,
                source='same_common_first_episode_not_native_environment_or_repeat_loop',
                success_count=int(success.sum()), success_rate=float(success.mean()),
                success_precision_cm=float(minimum[success].mean() * 100) if success.any() else 0.,
                success_by_env=success.tolist(), first_success_step=first_success_steps(hit).tolist(),
                min_success_error_cm=[float(x * 100) if np.isfinite(x) else None for x in minimum])
