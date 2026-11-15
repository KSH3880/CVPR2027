"""State machine for the opt-in, evaluation-driven Stage1 PUSH curriculum.

Probabilities describe agent slots (and therefore on-policy frames), not PPO
loss weights. Physical assets are rebuilt by the managed launcher at changes.
"""
from copy import deepcopy
import math

TASKS = ('AT', 'ON_TOP', 'SIT', 'CLIMB', 'PUSH')
TEMPLATES = ('HOLDING_AT', 'HOLDING_ON_TOP', 'SIT', 'CLIMB', 'PUSH_AT')
STATE_KEY = 'stage1_push_adaptive_state'
EVAL_CRITERION = 'region'


def validate_spec(env):
    spec = env['adaptiveSampling']
    keys = {'kind', 'eval_interval', 'eval_envs', 'eval_seed', 'minimum_probability',
            'focus_probability', 'minimum_focus_epochs', 'maximum_focus_epochs',
            'mastery_threshold', 'forgetting_threshold', 'confirmations', 'tie_break_order'}
    if set(spec) != keys or spec['kind'] != 'focus_blocks':
        raise ValueError('Adaptive sampling requires the focus_blocks specification')
    for key in ('eval_interval', 'eval_envs', 'minimum_focus_epochs', 'maximum_focus_epochs', 'confirmations'):
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError('Adaptive sampling requires positive integer ' + key)
    if (spec['eval_envs'] < 9 or type(spec['eval_seed']) is not int or spec['eval_seed'] not in (0, 1, 2)
            or not spec['eval_interval'] <= spec['minimum_focus_epochs'] <= spec['maximum_focus_epochs']
            or any(spec[k] % spec['eval_interval'] for k in ('minimum_focus_epochs', 'maximum_focus_epochs'))):
        raise ValueError('Adaptive evaluation/dwell settings are inconsistent')
    floor, focus = spec['minimum_probability'], spec['focus_probability']
    if (not all(type(spec[k]) in (int, float) and math.isfinite(spec[k])
                for k in ('minimum_probability', 'focus_probability', 'mastery_threshold', 'forgetting_threshold'))
            or not 0 < floor <= .2 <= focus <= 1-4*floor
            or not 0 <= spec['forgetting_threshold'] < spec['mastery_threshold'] <= 1
            or sorted(spec['tie_break_order']) != sorted(TASKS)):
        raise ValueError('Adaptive probability/threshold settings are inconsistent')
    if 'samplingSchedule' in env or list(env['relationGraph']['template_probabilities'].items()) != [
            ('HOLDING', 0.), ('SIT', .2), ('CLIMB', .2),
            ('HOLDING_AT', .2), ('HOLDING_ON_TOP', .2), ('PUSH_AT', .2)]:
        raise ValueError('Adaptive sampling requires a separate uniform scratch experiment')


def initial_state(spec):
    return dict(revision=1, last_eval_epoch=0, focus=None, focus_since=None,
                probabilities={t: .2 for t in TASKS},
                mastered={t: False for t in TASKS}, forgotten={t: False for t in TASKS}, high_streak={t: 0 for t in TASKS},
                low_streak={t: 0 for t in TASKS}, last_focus={t: -1 for t in TASKS},
                rates=None, reason='uniform_warmup')


def validate_state(state, spec, epoch=None):
    expected = initial_state(spec)
    if not isinstance(state, dict) or set(state) != set(expected) or state['revision'] != 1:
        raise ValueError('Invalid adaptive checkpoint state schema')
    last = state['last_eval_epoch']
    if type(last) is not int or last < 0 or last % spec['eval_interval']:
        raise ValueError('Invalid adaptive evaluation epoch')
    if epoch is not None and (type(epoch) is not int or epoch < last):
        raise ValueError('Adaptive state is ahead of the checkpoint epoch')
    focus, since = state['focus'], state['focus_since']
    if focus not in (None,) + TASKS or ((focus is None) != (since is None)):
        raise ValueError('Invalid adaptive focus')
    if since is not None and (type(since) is not int or not 0 < since <= last or since % spec['eval_interval']):
        raise ValueError('Invalid adaptive focus start')
    for key in ('probabilities', 'mastered', 'forgotten', 'high_streak', 'low_streak', 'last_focus'):
        if not isinstance(state[key], dict) or set(state[key]) != set(TASKS):
            raise ValueError('Missing adaptive task state: ' + key)
    p = state['probabilities']
    if (not all(type(v) in (int, float) and math.isfinite(v) and v >= spec['minimum_probability']-1e-12
                for v in p.values()) or not math.isclose(sum(p.values()), 1., abs_tol=1e-9)
            or (focus is None and any(abs(v-.2) > 1e-9 for v in p.values()))
            or (focus is not None and abs(p[focus]-spec['focus_probability']) > 1e-9)):
        raise ValueError('Invalid adaptive probability allocation')
    for t in TASKS:
        if (type(state['mastered'][t]) is not bool or type(state['forgotten'][t]) is not bool
                or any(type(state[k][t]) is not int or not 0 <= state[k][t] <= spec['confirmations']
                       for k in ('high_streak', 'low_streak'))
                or type(state['last_focus'][t]) is not int or not -1 <= state['last_focus'][t] <= last):
            raise ValueError('Invalid adaptive streak/history')
    if state['rates'] is not None:
        validate_rates(state['rates'])
    if not isinstance(state['reason'], str):
        raise ValueError('Invalid adaptive transition reason')
    return state


def validate_rates(rates):
    if (not isinstance(rates, dict) or set(rates) != set(TASKS)
            or not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in rates.values())):
        raise ValueError('Evaluation must contain five finite success rates in [0,1]')


def focus_probabilities(spec, focus, mastered):
    p = {t: spec['minimum_probability'] for t in TASKS}
    p[focus] = spec['focus_probability']
    recipients = [t for t in TASKS if t != focus and not mastered[t]]
    # When only one task remains, keep its configured 50% and give the unused
    # budget to the four maintenance tasks rather than losing probability mass.
    if not recipients:
        recipients = [t for t in TASKS if t != focus]
    extra = (1-sum(p.values()))/len(recipients)
    for t in recipients:
        p[t] += extra
    return p


def advance(state, spec, epoch, rates):
    """Consume exactly one complete evaluation; hold ratios within focus blocks."""
    validate_state(state, spec)
    validate_rates(rates)
    if type(epoch) is not int or epoch != state['last_eval_epoch'] + spec['eval_interval']:
        raise ValueError('Adaptive evaluations must advance exactly one interval')
    out = deepcopy(state)
    out['last_eval_epoch'], out['rates'] = epoch, dict(rates)
    for t in TASKS:
        out['high_streak'][t] = min(spec['confirmations'], out['high_streak'][t]+1) if rates[t] >= spec['mastery_threshold'] else 0
        out['low_streak'][t] = min(spec['confirmations'], out['low_streak'][t]+1) if rates[t] < spec['forgetting_threshold'] else 0
        if out['high_streak'][t] >= spec['confirmations']:
            out['mastered'][t] = True
            out['forgotten'][t] = False
        elif out['low_streak'][t] >= spec['confirmations']:
            if out['mastered'][t]:
                out['forgotten'][t] = True
            out['mastered'][t] = False
    if all(out['mastered'].values()):
        out.update(focus=None, focus_since=None, probabilities={t: .2 for t in TASKS}, reason='all_mastered_uniform')
        return validate_state(out, spec, epoch)
    focus = out['focus']
    elapsed = 0 if focus is None else epoch-out['focus_since']
    switch = focus is None or elapsed >= spec['maximum_focus_epochs'] or (
        elapsed >= spec['minimum_focus_epochs'] and out['mastered'][focus])
    if switch:
        candidates = [t for t in TASKS if not out['mastered'][t]]
        if len(candidates) > 1 and focus in candidates:
            candidates.remove(focus)
        order = spec['tie_break_order']
        # Confirmed forgetting gets priority at the next block boundary. Among
        # remaining candidates, rotate first, then prefer the lower eval score.
        forgotten = [t for t in candidates if out['forgotten'][t]]
        if forgotten:
            candidates = forgotten
        chosen = min(candidates, key=lambda t: (out['last_focus'][t], rates[t], order.index(t)))
        out.update(focus=chosen, focus_since=epoch, reason='focus_selected' if focus is None else 'focus_rotated')
        out['last_focus'][chosen] = epoch
        out['probabilities'] = focus_probabilities(spec, chosen, out['mastered'])
    else:
        out['reason'] = 'focus_held'
    return validate_state(out, spec, epoch)


def template_probabilities(state):
    return dict(HOLDING=0., **{template: state['probabilities'][task]
                              for task, template in zip(TASKS, TEMPLATES)})


def state_for_env(env):
    return env.get('_adaptiveSamplingState') or initial_state(env['adaptiveSampling'])


def check_checkpoint(weights, env):
    state = weights.get(STATE_KEY)
    if 'adaptiveSampling' not in env:
        if state is not None:
            raise ValueError('Adaptive PUSH checkpoint requires its own experiment')
        return
    validate_spec(env)
    if weights.get('stage1_push_adaptive_eval_criterion', EVAL_CRITERION) != EVAL_CRITERION:
        raise ValueError('Adaptive evaluation criterion differs from region')
    validate_state(state, env['adaptiveSampling'], weights.get('epoch'))
    if state != state_for_env(env):
        raise ValueError('Adaptive assets must be built from the saved sampling state')


def slot_templates(probabilities, count, device, floor, strict):
    """Integer allocation with a true floor for every action, including rounding."""
    import torch
    p = torch.tensor(list(probabilities.values()), dtype=torch.float64)
    if strict and count < math.ceil(1/floor):
        raise ValueError('Too few agent slots to guarantee the adaptive sampling floor')
    if not strict and count < math.ceil(1/floor):
        return torch.multinomial(p.to(device), count, replacement=True)
    counts = (p*count).floor().long()
    lower = torch.tensor([0 if v == 0 else math.ceil(floor*count) for v in p.tolist()])
    counts = counts.maximum(lower)
    while int(counts.sum()) != count:
        delta = p*count-counts
        if int(counts.sum()) < count:
            counts[int(delta.argmax())] += 1
        else:
            delta[counts <= lower] = float('inf')
            counts[int(delta.argmin())] -= 1
    ids = torch.arange(len(counts), device=device).repeat_interleave(counts.to(device))
    return ids[torch.randperm(count, device=device)]
