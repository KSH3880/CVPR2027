"""Optional premature box motion cost, allocated by positive hand work.

Net hand contact forces plus rear-face proximity are an attribution proxy;
Isaac Gym's net force tensor does not identify the contacted body pair.
"""
import math
import torch


def validate_config(config):
    expected = {"weight", "speed_threshold", "force_epsilon"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("premature_motion_penalty requires weight, speed_threshold, force_epsilon")
    for key, value in config.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Invalid premature_motion_penalty " + key)
        if value < 0 or (key == "force_epsilon" and value == 0):
            raise ValueError("Invalid premature_motion_penalty " + key)


def motion_penalty(hands, hand_forces, boxes, sizes, groups, templates,
                   tolerance, weight, speed_threshold, force_epsilon):
    """All arrays are per owner: hands/forces [B,M,2,3], boxes [B,M,13].

    Cost is weight * excess horizontal COM speed, once per group per sample.
    Each hand contributes only positive force along actual box motion, with
    rear-face proximity. No force means no attributed cost (including coasting).
    No goal exception, force threshold, saturation latch, or cross-group sharing.
    """
    from utils.coupled_push_spec import inverse_rotate
    local = inverse_rotate(boxes[..., None, 3:7].expand(*hands.shape[:-1], 4),
                           hands - boxes[..., None, :3])
    dx = (local[..., 0].abs() - sizes[..., None, 0] / 2).clamp_min(0)
    dz = (local[..., 2].abs() - sizes[..., None, 2] / 2).clamp_min(0)
    face_distance = (dx.square() + dz.square() +
                     (local[..., 1] + sizes[..., None, 1]/2).square()).sqrt()
    near = (face_distance <= tolerance) & (local[..., 1] <= 0)
    same = groups[:, :, None] == groups[:, None, :]
    shared_push = (same.sum(-1) > 1) & (templates == 5)
    ready = (near.all(-1)[:, None, :] | ~same).all(-1)
    velocity = boxes[..., 7:9]
    speed = velocity.norm(dim=-1)
    direction = velocity / speed.clamp_min(1e-12)[..., None]
    # Physics reports force ON the hand; negate for force applied by the hand.
    contribution = ((-hand_forces[..., :2] * direction[..., None, :]).sum(-1)
                    .clamp_min(0) * near).sum(-1) * shared_push
    total = (contribution[:, None, :] * same).sum(-1)
    share = torch.where(total > force_epsilon,
                        contribution / total.clamp_min(force_epsilon), 0.)
    excess = (speed - speed_threshold).clamp_min(0)
    penalty = -weight * excess * share * (~ready) * shared_push
    return dict(penalty=penalty, ready=ready, speed=speed, share=share,
                force=contribution, eligible=shared_push)
