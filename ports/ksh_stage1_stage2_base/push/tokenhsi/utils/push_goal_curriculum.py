"""Epoch-based PUSH goal distances, with explicit full-state checkpoint migration."""
from copy import deepcopy

STAGES = [dict(epoch=0, distance_max=5.), dict(epoch=1000, distance_max=7.),
          dict(epoch=3000, distance_max=10.)]


def validate_goal(spec):
    expected = dict(sampling='box_forward_uniform', distance_range=[1., 5.],
                    curriculum=STAGES)
    if spec != expected:
        raise ValueError('PUSH goal curriculum must be 1--5 / 1--7 / 1--10m at epochs 0/1000/3000')


def distance_range(env, epoch=0):
    spec = env['pushGoalRandomization']
    if 'curriculum' not in spec:
        return spec['distance_range']
    validate_goal(spec)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError('PUSH curriculum epoch must be a nonnegative integer')
    stage = next(stage for stage in reversed(spec['curriculum']) if epoch >= stage['epoch'])
    return [1., stage['distance_max']]


def migrate_checkpoint(weights, target_env):
    """Only change the goal law and scene spacing; preserve all training state."""
    from utils.stage1_push import validate_env
    validate_env(target_env)
    validate_goal(target_env['pushGoalRandomization'])
    saved = weights['relation_experiment_config']['env']
    expected = deepcopy(target_env)
    expected['pushGoalRandomization'] = dict(sampling='box_forward_uniform', distance_range=[1., 4.])
    expected['envSpacing'] = saved['envSpacing']
    for key in ('motion_file', 'seed', 'render_device', 'numObservations', 'numActions', 'task_plan'):
        if key in saved and key not in expected:
            expected[key] = deepcopy(saved[key])
    if saved != expected:
        raise ValueError('Migration requires the matching original 1--4m experiment; other settings must match')
    distance_range(target_env, weights['epoch'])
    result = dict(weights)
    result['relation_experiment_config'] = deepcopy(weights['relation_experiment_config'])
    migrated_env = deepcopy(saved)
    for key in ('pushGoalRandomization', 'envSpacing'):
        migrated_env[key] = deepcopy(target_env[key])
    result['relation_experiment_config']['env'] = migrated_env
    result['push_goal_migration'] = dict(source_goal=deepcopy(saved['pushGoalRandomization']),
                                         epoch=weights['epoch'])
    return result
