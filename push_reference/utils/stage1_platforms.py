"""TokenHSI height sampling, restricted to Stage1 carry payloads and AT goals."""
import torch


def validate_stage1_platform_config(env):
    reset = env['box']['reset']
    selected = env['relationReward'].get('stage1_variant') in ('stage1', 'stage1_v2')
    if selected != (reset.get('platformMode') == 'carry_payload_tokenhsi'):
        raise ValueError('stage1 and carry_payload_tokenhsi platforms must be paired')
    if reset.get('platformMode') not in (None, 'carry_payload_tokenhsi'):
        raise ValueError('Unknown platformMode')
    if selected and (reset.get('randomHeight') is not True
                     or reset.get('randomHeightProb') != .5
                     or reset.get('minPlatformHeight') != 0.
                     or reset.get('maxTopSurfaceHeight') != 1.2
                     or reset.get('testRandomArenaSpawn') is not False
                     or reset.get('groundStandaloneSitClimb') is not True
                     or env.get('agentCollisionMode') != 'cpa'
                     or env.get('agentCollisionPenalty') is not True
                     or env.get('agentCollisionCoeff') != .5
                     or env.get('agentCollisionDist') != .7
                     or env.get('agentCollisionTTCDiscount') != .99):
        raise ValueError('stage1 requires original TokenHSI heights and the CPA contract')


def sample_payload_height(size, probability=.5, max_top=1.2):
    """Return center height before the source-only 5 cm contact clearance.

    Sample the original U(0,1) bottom height and then clamp the top; do not
    replace this with a uniform draw over the clamped interval.
    """
    elevated = torch.rand(size.shape[:-1], device=size.device) < probability
    offset = torch.rand(size.shape[:-1], device=size.device) * elevated
    return torch.minimum(size[..., 2] / 2 + offset, max_top - size[..., 2] / 2)


def reset_stage1_platforms(task, env_ids):
    """Called after graph/RSI reset and ON_TOP support placement, before screening.

    Platform slots belong to owners, while object/goal indices follow the graph.
    Reference source poses are never lifted. ON_TOP retains its physical support.
    """
    from push_reference.utils.edge_scenario_spec import classify_templates

    task._platform_pos[env_ids] = task._platform_default_pos[env_ids]
    task._tar_platform_pos[env_ids] = task._tar_platform_default_pos[env_ids]
    for source, skills in ((True, ['loco']), (False, ['loco', 'pickUp', 'carryWith'])):
        slots = task._collect_random_slots(skills)
        if slots is None:
            continue
        env, owner = slots
        template = classify_templates(task.relation_runtime.graph, env, owner, True)
        # HOLDING_AT=3, HOLDING_ON_TOP=4; standalone tasks never get platforms.
        keep = ((template == 3) | (template == 4)) if source else template == 3
        env, owner = env[keep], owner[keep]
        if not len(env):
            continue
        payload = task._scenario_physical_object(env, owner)
        size = task._box_size[env, payload]
        z = sample_payload_height(size, task._reset_random_height_prob,
                                  task._reset_max_top_surface_height)
        if source:
            pos = task._box_states[env, payload, :3].clone()
            pos[:, 2] = z
            task._box_states[env, payload, 2] = z + .05
            platforms = task._platform_pos
        else:
            goal = task._scenario_goal(env, owner)
            pos = task._tar_pos[env, goal].clone()
            pos[:, 2] = z
            task._tar_pos[env, goal, 2] = z
            platforms = task._tar_platform_pos
        platforms[env, owner, :2] = pos[:, :2]
        # Also place ground-selected plates just below the ground, as in TokenHSI.
        platforms[env, owner, 2] = z - size[:, 2] / 2 - task._platform_height / 2
