"""Goal-relative horizontal velocity tracking for BEFORE followers."""
import math
import torch


def validate_target_velocity(config):
    keys = {'mode', 'max_speed', 'deceleration_time', 'goal_buffer', 'arrival_fade_distance'}
    if not isinstance(config, dict) or not keys.issubset(config) or not set(config).issubset(keys | {'ontop', 'after_release', 'far_approach_blend_width'}) or config['mode'] not in ('sit_climb_l2_v1', 'sit_climb_l2_squared_v1', 'sit_climb_speed_cap_v1'):
        raise ValueError('waiting penalty target_velocity requires sit_climb_l2_v1 settings')
    if 'far_approach_blend_width' in config:
        width = config['far_approach_blend_width']
        if (config['mode'] != 'sit_climb_l2_squared_v1' or config.get('after_release') != 'disable'
                or type(width) not in (int, float) or not math.isfinite(width) or width <= 0):
            raise ValueError('waiting penalty far approach blend requires squared/release-off and positive width')
    if 'after_release' in config and config['after_release'] != 'disable':
        raise ValueError('waiting penalty after_release must be disable')
    if 'ontop' in config:
        top = config['ontop']
        if (not isinstance(top, dict) or set(top) != {'activation', 'goal_buffer'}
                or top['activation'] != 'first_holding_success'
                or type(top['goal_buffer']) not in (int, float)
                or not math.isfinite(top['goal_buffer']) or top['goal_buffer'] <= 0):
            raise ValueError('waiting penalty ON_TOP requires first_holding_success and positive goal_buffer')
    if any(type(config[k]) not in (int, float) or not math.isfinite(config[k]) or config[k] <= 0
           for k in keys - {'mode'}):
        raise ValueError('waiting penalty target_velocity values must be positive and finite')


def target_velocity_penalty(position, velocity, goal, support_position, support_size,
                            released, radius_buffer, coefficient, config):
    """Return positive cost, target XY velocity and arrival weight; no state mutation.

    Before release track the waiting boundary around the predecessor goal. After
    first release track the physical support box, fading out near the skill's
    bbox-buffer radius (legacy). With after_release=disable, first release
    removes the velocity cost completely. Vertical velocity is excluded.
    """
    target = torch.where(released[:, None], support_position[:, :2], goal[:, :2])
    offset = target - position[:, :2]
    distance = offset.norm(dim=-1)
    half_diagonal = support_size[:, :2].norm(dim=-1) / 2
    radius = half_diagonal + torch.where(released, config['goal_buffer'], radius_buffer)
    remaining = (distance - radius).clamp_min(0.)
    speed = (remaining / config['deceleration_time']).clamp_max(config['max_speed'])
    desired = offset / distance.clamp_min(1e-6)[:, None] * speed[:, None]
    fade = torch.where(released,
                       (remaining / config['arrival_fade_distance']).clamp(0., 1.),
                       torch.ones_like(distance))
    if config.get('after_release') == 'disable':
        fade = torch.where(released, torch.zeros_like(fade), fade)
        desired = torch.where(released[:, None], torch.zeros_like(desired), desired)
    if config['mode'] == 'sit_climb_speed_cap_v1':
        error = (velocity[:, :2].norm(dim=-1) - speed).clamp_min(0.)
    elif config['mode'] == 'sit_climb_l2_squared_v1':
        error = (velocity[:, :2] - desired).square().sum(dim=-1)
        if 'far_approach_blend_width' in config:
            # Far away, constrain only goalward speed; blend into vector tracking
            # before the distance-dependent target begins decelerating.
            direction = offset / distance.clamp_min(1e-6)[:, None]
            radial_speed = (velocity[:, :2] * direction).sum(dim=-1)
            floor_error = (config['max_speed'] - radial_speed).clamp_min(0.).square()
            braking_distance = config['max_speed'] * config['deceleration_time']
            width = config['far_approach_blend_width']
            u = ((braking_distance + width - remaining) / width).clamp(0., 1.)
            tracking_weight = u.square() * (3. - 2.*u)
            error = (1. - tracking_weight) * floor_error + tracking_weight * error
    else:
        error = (velocity[:, :2] - desired).norm(dim=-1)
    penalty = coefficient * fade * error
    return penalty, desired, fade
