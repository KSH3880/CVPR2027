"""Epoch-only task curriculum for the opt-in Stage1 PUSH experiment."""
import math
import torch

TASKS = ('HOLDING', 'SIT', 'CLIMB', 'HOLDING_AT', 'HOLDING_ON_TOP', 'PUSH_AT')


def validate_schedule(env):
    spec = env['samplingSchedule']
    initial = env['relationGraph']['template_probabilities']
    final = spec['final_template_probabilities']
    if (set(spec) != {'kind', 'start_epoch', 'end_epoch', 'final_template_probabilities'}
            or spec['kind'] != 'linear_epoch'
            or type(spec['start_epoch']) is not int or type(spec['end_epoch']) is not int
            or not 0 <= spec['start_epoch'] < spec['end_epoch']):
        raise ValueError('PUSH sampling schedule requires an increasing integer epoch interval')
    for probabilities in (initial, final):
        values = list(probabilities.values())
        if (tuple(probabilities) != TASKS or not all(math.isfinite(v) for v in values)
                or values[0] != 0 or any(v <= 0 for v in values[1:])
                or not math.isclose(sum(values), 1., abs_tol=1e-9)):
            raise ValueError('PUSH schedule needs five positive task probabilities summing to one')
    if initial['PUSH_AT'] != final['PUSH_AT']:
        raise ValueError('PUSH asset slots must stay fixed throughout the schedule')


def schedule_state(env, epoch):
    if type(epoch) is not int or epoch < 0:
        raise ValueError('Sampling epoch must be a nonnegative integer')
    spec = env['samplingSchedule']
    fraction = min(1., max(0., (epoch-spec['start_epoch']) /
                          (spec['end_epoch']-spec['start_epoch'])))
    initial = env['relationGraph']['template_probabilities']
    final = spec['final_template_probabilities']
    probabilities = [initial[t] + fraction*(final[t]-initial[t]) for t in TASKS]
    return dict(epoch=epoch, fraction=fraction, probabilities=probabilities)


def check_schedule_checkpoint(weights, env):
    if 'samplingSchedule' not in env:
        if 'stage1_push_sampling_state' in weights:
            raise ValueError('Scheduled PUSH checkpoint requires its scheduled experiment')
        return
    epoch = weights.get('epoch')
    if weights.get('stage1_push_sampling_state') != schedule_state(env, epoch):
        raise ValueError('PUSH checkpoint sampling epoch/state differs from its schedule')


def balanced_templates(probabilities, count):
    """Largest-remainder counts, randomly placed once when assets are built."""
    exact = probabilities.double()*count
    counts = exact.floor().long()
    remainder = count-int(counts.sum())
    counts[(exact-counts).argsort(descending=True)[:remainder]] += 1
    ids = torch.arange(len(counts), device=counts.device).repeat_interleave(counts)
    return ids[torch.randperm(count, device=counts.device)]


def fit_task_weights(base, probabilities, strict=True):
    """Balance eligible fixed assets to a target marginal, without changing sizes.

    Only the few distinct eligibility/likelihood patterns are fitted on CPU.
    PUSH rows remain one-hot. Small viewer scenes can have insufficient assets;
    they retain size-conditioned priors instead of inventing incompatible tasks.
    """
    flat = base.reshape(-1, len(TASKS))
    push = flat[:, -1] > 0
    if push.all():
        return base.clone()
    source = flat[~push, :-1].double().cpu()
    target = torch.tensor(probabilities[:-1], dtype=torch.float64)
    target /= target.sum()
    # Same eligible tasks imply the same prior/size-volume likelihood ratios.
    patterns, inverse, counts = torch.unique(source, dim=0, return_inverse=True,
                                            return_counts=True)
    weights = patterns.clone()
    mass = counts.double()/len(source)
    for _ in range(2000):
        actual = (weights*mass[:, None]).sum(0)
        if (actual-target).abs().max() < 1e-8:
            result = flat.clone()
            result[~push, :-1] = weights[inverse].to(device=base.device, dtype=base.dtype)
            return result.reshape_as(base)
        weights *= target/actual.clamp_min(1e-30)
        weights /= weights.sum(-1, keepdim=True).clamp_min(1e-30)
    if strict:
        raise ValueError('Fixed asset sizes cannot realize the requested PUSH task schedule')
    # This is also the legacy size-conditioned interpretation for tiny viewers.
    ratios = target/source.mean(0).clamp_min(1e-30)
    source *= ratios
    source /= source.sum(-1, keepdim=True)
    result = flat.clone()
    result[~push, :-1] = source.to(device=base.device, dtype=base.dtype)
    return result.reshape_as(base)


class TaskSamplingSchedule:
    def __init__(self, env, base, strict=True):
        self.env = env
        self.initial = fit_task_weights(base, schedule_state(env, 0)['probabilities'], strict)
        final = list(env['samplingSchedule']['final_template_probabilities'].values())
        self.final = fit_task_weights(base, final, strict)
        self.epoch = None
        self.weights = None
        self.state = None

    def update(self, epoch):
        state = schedule_state(self.env, epoch)
        # Reuse tensors throughout constant phases; no per-step synchronization.
        if self.state is None or state['fraction'] != self.state['fraction']:
            self.weights = torch.lerp(self.initial, self.final, state['fraction'])
        self.epoch, self.state = epoch, state
        return self.weights
