"""Task-family AMP contracts for the independent unified Stage-1 experiment."""
import torch
import torch.nn.functional as F

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
    from utils.edge_stage1_spec import UNIFIED_VARIANTS
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
    if list(env['relationGraph']['template_probabilities'].values()) != [.05, .05, .20, .35, .35]:
        raise ValueError('Unified requires independent task proportions 5/5/20/35/35')
