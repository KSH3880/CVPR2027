"""Stage1 task-conditioned AMP sampling and training contracts."""

import torch
import torch.nn.functional as F
from utils.stage1_platforms import validate_stage1_platform_config
from utils.task_role_spec import STAGE1_VARIANTS, TASK_PROBS

FAMILIES = ("carry", "sit", "climb")
SKILLS = (
    "loco",
    "sit",
    "climb",
    "climbNoRSI",
    "omomo",
    "pickUp",
    "carryWith",
    "putDown",
)
EXPERT_PROBS = (
    (1 / 3, 0.0, 0.0, 0.0, 1 / 3, 1 / 6, 0.0, 1 / 6),
    (1 / 3, 2 / 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (1 / 3, 0.0, 1 / 3, 1 / 3, 0.0, 0.0, 0.0, 0.0),
)


def family_from_templates(templates):
    return torch.where(templates == 1, 1, torch.where(templates == 2, 2, 0))


def append_family(obs, family, families=3):
    """Append one-hot family to each motion frame, preserving arbitrary prefix dims."""
    return torch.cat((obs, F.one_hot(family.long(), families).to(obs.dtype)), -1)


def amp_family_ids(obs, steps, families=3):
    return obs.reshape(-1, steps, obs.shape[-1] // steps)[:, 0, -families:].argmax(-1)


def sample_family_matched(pool, reference, steps, fallback=None, families=3):
    """Sample actual trajectories of the requested family; never relabel motion.

    Matching row-by-row also matches every shuffled PPO minibatch and AMP prefix.
    Fresh rollouts supply a missing replay family during replay warm-up only.
    """
    ids = amp_family_ids(reference, steps, families)
    pool_ids = amp_family_ids(pool, steps, families)
    result = torch.empty_like(reference)
    fallback_ids = None if fallback is None else amp_family_ids(fallback, steps, families)
    for family in range(families):
        dest = (ids == family).nonzero(as_tuple=False).flatten()
        if not len(dest):
            continue
        source = pool
        candidates = (pool_ids == family).nonzero(as_tuple=False).flatten()
        if not len(candidates) and fallback is not None:
            source = fallback
            candidates = (fallback_ids == family).nonzero(as_tuple=False).flatten()
        if not len(candidates):
            raise ValueError("AMP buffer is missing expert family: " + (FAMILIES + ("push",))[family])
        selected = candidates[
            torch.randint(len(candidates), (len(dest),), device=pool.device)
        ]
        result[dest] = source[selected]
    return result


def preserve_amp_labels(raw, normalized, steps, families=3):
    """Normalise motion features, but keep task labels literal one-hot values."""
    shape = raw.shape
    raw = raw.reshape(-1, steps, shape[-1] // steps)
    normalized = normalized.reshape_as(raw)
    return torch.cat((normalized[..., :-families], raw[..., -families:]), -1).reshape(shape)


def validate_unified_env(env):
    if env.get("stage2JointPush") is not None:
        from utils.coupled_push_spec import validate_env
        return validate_env(env)
    if env.get("jointPush") is not None and env.get("before") is not None:
        from utils.before_push_spec import validate_env
        return validate_env(env)
    if env.get("jointPush") is not None:
        from utils.stage1_jointpush import validate_env
        return validate_env(env)
    if env.get('before') is not None:
        from utils.before_spec import validate_env
        return validate_env(env)
    validate_common_env(env)
    from utils.edge_scenario_spec import validate_sampler
    validate_sampler(env['relationGraph'], env['numAgents'], env['numObjects'])
    expected = [.5, 0., 0., 0., 0., .1, .3, .1]
    if any(env['templateRsi'][key] != expected for key in ('HOLDING_AT', 'HOLDING_ON_TOP')):
        raise ValueError('Fixed Stage1 placement RSI requires 50/10/30/10')


def validate_common_env(env):
    from utils.stage1_geometry import RSI_SPEC
    from utils.size_rsi import STAGE1_RANGES
    from utils.stage1_config import uses_fixed_platforms
    from utils.relation_task_spec import validate_relation_config
    validate_relation_config(env['relationReward'])
    if env['asset']['assetFileName'] != 'mjcf/phys_humanoid_v3.xml':
        raise ValueError('Expected phys_humanoid_v3 asset')
    validate_stage1_platform_config(env)
    if not uses_fixed_platforms(env):
        raise ValueError('Fixed platform physics is required')
    if (env.get('ampTaskConditioning') is not True
            or tuple(env['skill']) != SKILLS
            or env.get('goalRotation') != 'identity'
            or env.get('policyObsMode') != 'clean_scene'):
        raise ValueError('Expected task-conditioned AMP and clean_scene identity goals')
    if list(env['relationGraph']['template_probabilities'].values()) != list(TASK_PROBS):
        raise ValueError('Task proportions differ from the contract')
    reset, build = env['box']['reset'], env['box']['build']
    if (reset.get('samplingMode') != 'tokenhsi'
            or reset.get('ownerLocoDistanceRange') != [1., 10.]
            or reset.get('randomAssignment') is not False
            or build.get('taskSizeRanges') != STAGE1_RANGES
            or build.get('sizeInterval') != .05
            or env.get('sizeAwareRsi') != RSI_SPEC):
        raise ValueError('Invalid size-aware asset / RSI contract')


def validate_stage1_network(env, train):
    if env.get("relationReward") is None:
        return
    if env["relationReward"].get("stage1_variant") not in STAGE1_VARIANTS:
        raise ValueError("Unsupported Stage1 experiment")
    validate_transformer(train["params"]["network"])
    if env.get("relationGraph", {}).get("policy_task_roles") is not True:
        raise ValueError("Stage1 requires the task packet architecture")


def validate_transformer(network):
    tp = network.get("transformer", {})
    if (
        tp.get("relation_bias_mode") != "task_shared_embedding_action"
        or tp.get("relation_bias") is not True
        or tp.get("relation_message", {}).get("enable", False)
        or tp.get("share_edge_encoder", False)
        or (network.get("separate") is not True)
        or (tp.get("num_features") != 64)
        or (tp.get("num_layers") != 4)
        or (tp.get("layer_num_heads") != 2)
        or tp.get("geometry", {}).get("enable", False)
        or (tp.get("gta", {}).get("enable") is not True)
        or (tp.get("gta", {}).get("representation") != "se3_direct_sum")
        or (tp.get("relation_attention_mode") is not None)
        or (network.get("coordination") is not None)
    ):
        raise ValueError(
            "Stage1 requires separate actor/critic, shared task bias, Action MLP and 64-D/2-head/4-layer GTA"
        )
