from copy import deepcopy
import math

import torch


def validate_config(config):
    if set(config) != {'weight', 'speed_threshold', 'force_epsilon'}:
        raise ValueError('Invalid premature motion penalty settings')
    if any(not math.isfinite(value) or value < 0 for value in config.values()) or config['force_epsilon'] == 0:
        raise ValueError('Premature motion penalty settings must be finite and nonnegative, with positive epsilon')


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


def check_contact_resume(checkpoint, env):
    from utils.coupled_push_spec import checkpoint_contract, check_checkpoint
    expected = checkpoint_contract(env)
    if checkpoint.get('coupled_push_contract') == expected:
        return check_checkpoint(checkpoint, env)
    original = deepcopy(expected)
    original['coupled'].pop('premature_motion_penalty', None)
    source = checkpoint.get('relation_experiment_config', {}).get('experiment', {}).get('name')
    baseline = 'stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg'
    if (source != baseline or checkpoint.get('epoch') != 17000
            or checkpoint.get('coupled_push_contract') != original):
        raise ValueError('Contact finetuning requires the matching zerobg 17000 checkpoint or a contact resume')
