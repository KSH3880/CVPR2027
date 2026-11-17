import math


def validate_config(params):
    if params['algo']['name'] != 'ma_distill':
        raise ValueError('stage2_distill requires the ma_distill trainer')
    cfg = params['config']['distillation']
    if cfg['version'] != 1 or cfg['solo_teacher'] != 'before' or cfg['loss'] != 'mean_action_mse':
        raise ValueError('Unsupported distillation contract')
    for key in ('teacher_before', 'teacher_coupled'):
        if not isinstance(cfg[key], str) or not cfg[key]:
            raise ValueError('Missing distillation teacher: '+key)
    for key in ('rollout_steps', 'replay_capacity', 'minibatch_scenes', 'updates_per_iteration'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError('Invalid distillation '+key)
    for key in ('teacher_warmup_iterations', 'teacher_decay_iterations'):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError('Invalid distillation '+key)
    if not math.isfinite(params['config']['learning_rate']) or params['config']['learning_rate'] <= 0:
        raise ValueError('Distillation learning_rate must be positive')


def teacher_probability(iteration, cfg):
    warmup = cfg['teacher_warmup_iterations']
    decay = cfg['teacher_decay_iterations']
    if iteration <= warmup:
        return 1.
    return max(0., 1.-(iteration-warmup)/decay) if decay else 0.
