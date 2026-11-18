import torch


def motion_penalty(velocity, hand_forces, near, groups, eligible, config):
    speed = velocity.norm(dim=-1)
    direction = velocity / speed.clamp_min(1e-12)[..., None]
    force = (-(hand_forces * direction[..., None, :]).sum(-1)).clamp_min(0)
    force = (force * near).sum(-1) * eligible
    same = groups[:, :, None] == groups[:, None, :]
    ready = (near.all(-1)[:, None, :] | ~same).all(-1)
    total = (force[:, None, :] * same).sum(-1)
    share = torch.where(total > config['force_epsilon'],
                        force / total.clamp_min(config['force_epsilon']), 0.)
    moving_unready = eligible & ~ready & (speed > config['speed_threshold'])
    penalty = -config['weight'] * (speed-config['speed_threshold']).clamp_min(0) * share * moving_unready
    return penalty, moving_unready, force
