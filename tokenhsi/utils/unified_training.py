"""Task-family AMP contracts for the independent unified Stage-1 experiment."""
import torch
import torch.nn.functional as F
from utils.task_role_spec import TASK_ROLE_VARIANT, TASK_PROBS as ROLE_TASK_PROBS

FAMILIES = ('carry', 'sit', 'climb')
SKILLS = ('loco', 'sit', 'climb', 'climbNoRSI', 'omomo', 'pickUp', 'carryWith', 'putDown')
FAMILY_PROBS = (.75, .05, .20)
# Original unified's conditional expert distributions; locomotion aliases share clips.
EXPERT_PROBS = (
    (1/3, 0., 0., 0., 1/3, 1/6, 0., 1/6),
    (1/3, 2/3, 0., 0., 0., 0., 0., 0.),
    (1/3, 0., 1/3, 1/3, 0., 0., 0., 0.),
)


def family_from_templates(templates):
    # HOLDING / SIT / CLIMB / HOLDING_AT / HOLDING_ON_TOP
    return torch.where(templates == 1, 1, torch.where(templates == 2, 2, 0))


def append_family(obs, family):
    """Append one-hot family to each motion frame, preserving arbitrary prefix dims."""
    return torch.cat((obs, F.one_hot(family.long(), len(FAMILIES)).to(obs.dtype)), -1)


def amp_family_ids(obs, steps):
    return obs.reshape(-1, steps, obs.shape[-1] // steps)[:, 0, -3:].argmax(-1)


def sample_family_matched(pool, reference, steps, fallback=None):
    """Sample actual trajectories of the requested family; never relabel motion.

    Matching row-by-row also matches every shuffled PPO minibatch and AMP prefix.
    Fresh rollouts supply a missing replay family during replay warm-up only.
    """
    ids = amp_family_ids(reference, steps)
    pool_ids = amp_family_ids(pool, steps)
    result = torch.empty_like(reference)
    fallback_ids = None if fallback is None else amp_family_ids(fallback, steps)
    for family in range(len(FAMILIES)):
        dest = (ids == family).nonzero(as_tuple=False).flatten()
        if not len(dest):
            continue
        source = pool
        candidates = (pool_ids == family).nonzero(as_tuple=False).flatten()
        if not len(candidates) and fallback is not None:
            source = fallback
            candidates = (fallback_ids == family).nonzero(as_tuple=False).flatten()
        if not len(candidates):
            raise ValueError('AMP buffer is missing expert family: ' + FAMILIES[family])
        selected = candidates[torch.randint(len(candidates), (len(dest),), device=pool.device)]
        result[dest] = source[selected]
    return result


def preserve_amp_labels(raw, normalized, steps):
    """Normalise motion features, but keep task labels literal one-hot values."""
    shape = raw.shape
    raw = raw.reshape(-1, steps, shape[-1] // steps)
    normalized = normalized.reshape_as(raw)
    return torch.cat((normalized[..., :-3], raw[..., -3:]), -1).reshape(shape)


def validate_unified_env(env):
    """Fail early if the new sampler, AMP, geometry and reward contracts diverge."""
    from utils.edge_scenario_spec import UNIFIED_SAMPLER
    from utils.edge_stage1_spec import UNIFIED_VARIANTS, SIZE_RSI_VARIANTS
    unified = env['relationGraph'].get('sampler') == UNIFIED_SAMPLER
    enabled = env.get('ampTaskConditioning', False)
    variant = env['relationReward'].get('stage1_variant')
    if unified != (variant in UNIFIED_VARIANTS) or unified != (enabled is True):
        raise ValueError('Unified sampler, reward variant and task-conditioned AMP must be paired')
    if not unified:
        return
    if tuple(env['skill']) != SKILLS or env.get('goalRotation') != 'identity':
        raise ValueError('Unified requires unified skills and identity goal rotation')
    if env['box']['reset'].get('ownerLocoDistanceRange') != [1., 2.]:
        raise ValueError('Unified loco sources must start 1..2m from owner')
    from utils.size_rsi import TASK_PROBS, SIZE_RANGES, SCREEN
    task_roles = variant == TASK_ROLE_VARIANT
    if task_roles != bool(env['relationGraph'].get('policy_task_roles', False)):
        raise ValueError('Task-role packet and reward variant must be paired')
    expected = (list(ROLE_TASK_PROBS) if task_roles else list(TASK_PROBS)
                if variant in SIZE_RSI_VARIANTS else [.05, .05, .20, .35, .35])
    if list(env['relationGraph']['template_probabilities'].values()) != expected:
        raise ValueError('Unified task proportions do not match the selected variant')
    sized = env.get('sizeAwareRsi')
    if (variant in SIZE_RSI_VARIANTS) != (sized is not None):
        raise ValueError('Size-aware assets and RSI must use their dedicated variant')
    if sized is not None:
        if (env['box']['build'].get('taskSizeRanges') != SIZE_RANGES
                or env['box']['build'].get('sizeInterval') != .05
                or env['box']['reset'].get('randomAssignment') is not False
                or sized != {'cacheDirectory': 'output/rsi_cache', 'lateProbability': .7,
                             'lateFraction': .3, 'screen': SCREEN}):
            raise ValueError('Invalid size-aware asset / RSI contract')
        for name in ('HOLDING_AT', 'HOLDING_ON_TOP'):
            if env['templateRsi'][name] != [.4, 0., 0., 0., 0., .1, .4, .1]:
                raise ValueError('Size-aware placement RSI requires 40/10/40/10')


def validate_typed_bias_config(env, train):
    from utils.edge_stage1_spec import SIZE_RSI_TYPED_BIAS_VARIANT, SIZE_RSI_TYPED_MESSAGE_VARIANT
    variant = env.get('relationReward', {}).get('stage1_variant')
    selected = variant in (SIZE_RSI_TYPED_BIAS_VARIANT, SIZE_RSI_TYPED_MESSAGE_VARIANT)
    transformer = train['params']['network'].get('transformer', {})
    if selected != (transformer.get('relation_bias_mode') == 'typed_lookup'):
        raise ValueError('Size RSI typed-bias variant and typed_lookup train config must be paired')
    task_roles = variant == TASK_ROLE_VARIANT
    if task_roles != (transformer.get('relation_bias_mode') == 'task_role_lookup') or \
            task_roles != bool(env.get('relationGraph', {}).get('policy_task_roles', False)):
        raise ValueError('Task-role variant, policy packet and task_role_lookup must be paired')
    if (selected or task_roles) and (transformer.get('share_edge_encoder', False)
                     or transformer.get('relation_bias') is not True):
        raise ValueError('Typed bias requires enabled bias and separate actor/critic tables')
    message = transformer.get('relation_message', {})
    if (variant in (SIZE_RSI_TYPED_MESSAGE_VARIANT, TASK_ROLE_VARIANT)) != bool(message.get('enable', False)):
        raise ValueError('Typed relation-message variant and enabled relation_message must be paired')
    if variant in (SIZE_RSI_TYPED_MESSAGE_VARIANT, TASK_ROLE_VARIANT) and (
            message != {'enable': True, 'alpha': 1.0, 'init_std': .02}
            or transformer.get('num_features') != 64
            or transformer.get('layer_num_heads') != 2
            or transformer.get('num_layers') != 4):
        raise ValueError('Typed relation-message experiment requires alpha=1, std=0.02 and 64-D/2-head/4-layer')
