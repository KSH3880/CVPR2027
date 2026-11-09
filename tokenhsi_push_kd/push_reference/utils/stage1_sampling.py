"""Stage1 opt-in TokenHSI distances; legacy saved configs retain their sampler."""
import torch


def uses_tokenhsi_sampling(env):
    return env['box']['reset'].get('samplingMode') == 'tokenhsi'


def sample_source_xy(human_xy, templates):
    # Template 1 is SIT; CLIMB and both carry tasks use the native 10 m limit.
    maximum = torch.where(templates == 1, 5., 10.)
    direction = torch.randn_like(human_xy)
    direction /= direction.norm(dim=-1, keepdim=True)
    radius = 1. + torch.rand_like(maximum) * (maximum - 1.)
    return human_xy + direction * radius[:, None]


def sample_goal_xy(human_xy, payload_xy, center_xy):
    """Native square U([-4.5,4.5]^2), outside both 1 m exclusion circles."""
    target = center_xy + (torch.rand_like(human_xy) * 2. - 1.) * 4.5
    for _ in range(256):
        invalid = ((target - human_xy).square().sum(-1) < 1.) | (
            (target - payload_xy).square().sum(-1) < 1.)
        if not invalid.any():
            return target
        target[invalid] = center_xy[invalid] + (
            torch.rand_like(target[invalid]) * 2. - 1.) * 4.5
    raise RuntimeError('TokenHSI goal sampler exhausted rejection attempts')


def reset_ontop_support_xy(task):
    """Match AT's random destination XY; putDown retains its reference target."""
    from push_reference.utils.edge_scenario_spec import classify_templates
    slots = task._collect_random_slots(['loco', 'pickUp', 'carryWith'])
    if slots is None:
        return
    env, owner = slots
    keep = classify_templates(task.relation_runtime.graph, env, owner, True) == 4
    env, owner = env[keep], owner[keep]
    if not len(env):
        return
    payload = task._scenario_physical_object(env, owner)
    # ON_TOP task records bind their target to a logical object, not a goal slot.
    from push_reference.utils.task_role_spec import task_graph_packet
    packet = task_graph_packet(task.relation_runtime.graph, task.num_envs).reshape(
        task.num_envs, task.num_agents, 5).long()
    support = task._logical_box_order[env, packet[env, owner, 4] - task.num_agents]
    task._box_states[env, support, :2] = sample_goal_xy(
        task._humanoid_root_states[env, owner, :2],
        task._box_states[env, payload, :2], task._env_origins[env, :2])


def check_sampling_resume(weights, env):
    if env.get('relationReward', {}).get('stage1_variant') not in ('stage1', 'stage1_v2'):
        return
    saved = weights.get('relation_experiment_config', {}).get('env')
    if saved is None:
        if uses_tokenhsi_sampling(env):
            raise ValueError('Stage1 sampling resume requires saved experiment config')
        return
    if (uses_tokenhsi_sampling(saved) != uses_tokenhsi_sampling(env)
            or saved.get('templateRsi') != env.get('templateRsi')
            or env['relationReward']['stage1_variant'] == 'stage1_v2' and (
                saved.get('sizeAwareRsi') != env.get('sizeAwareRsi')
                or saved['box']['build'].get('taskSizeRanges') != env['box']['build'].get('taskSizeRanges'))):
        raise ValueError('Stage1 sampling/RSI mismatch: use the saved env config to resume')
