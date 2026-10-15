"""Stage1 task-conditioned AMP sampling and training contracts."""

import torch
import torch.nn.functional as F
from utils.stage1_platforms import validate_stage1_platform_config
from utils.task_role_spec import STAGE1_VARIANTS, STAGE1_V2_VARIANT, TASK_PROBS

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
            raise ValueError("AMP buffer is missing expert family: " + FAMILIES[family])
        selected = candidates[
            torch.randint(len(candidates), (len(dest),), device=pool.device)
        ]
        result[dest] = source[selected]
    return result


def preserve_amp_labels(raw, normalized, steps):
    """Normalise motion features, but keep task labels literal one-hot values."""
    shape = raw.shape
    raw = raw.reshape(-1, steps, shape[-1] // steps)
    normalized = normalized.reshape_as(raw)
    return torch.cat((normalized[..., :-3], raw[..., -3:]), -1).reshape(shape)


def validate_unified_env(env):
    from utils.edge_scenario_spec import validate_sampler
    from utils.stage1_sampling import uses_tokenhsi_sampling
    from utils.size_rsi import SIZE_RANGES, STAGE1_V2_RANGES, SCREEN
    from utils.stage1_v2 import RSI_SPEC

    variant = env["relationReward"].get("stage1_variant")
    if variant not in STAGE1_VARIANTS:
        raise ValueError("Only stage1 and stage1_v2 environments are supported")
    validate_stage1_platform_config(env)
    validate_sampler(env["relationGraph"], env["numAgents"], env["numObjects"])
    if (
        env.get("ampTaskConditioning") is not True
        or tuple(env["skill"]) != SKILLS
        or env.get("goalRotation") != "identity"
        or (env.get("policyObsMode") != "clean_scene")
    ):
        raise ValueError(
            "Stage1 requires task-conditioned AMP, supported skills and clean_scene identity goals"
        )
    if list(env["relationGraph"]["template_probabilities"].values()) != list(
        TASK_PROBS
    ):
        raise ValueError("Stage1 task proportions do not match the contract")
    native_sampling = uses_tokenhsi_sampling(env)
    if env["box"]["reset"].get("samplingMode") not in (None, "tokenhsi"):
        raise ValueError("Unsupported sampling mode")
    distance = [1.0, 10.0] if native_sampling else [1.0, 2.0]
    if env["box"]["reset"].get("ownerLocoDistanceRange") != distance:
        raise ValueError("Stage1 source distance range differs")
    ranges = STAGE1_V2_RANGES if variant == STAGE1_V2_VARIANT else SIZE_RANGES
    rsi = (
        RSI_SPEC
        if variant == STAGE1_V2_VARIANT
        else dict(
            cacheDirectory="output/rsi_cache",
            lateProbability=0.7,
            lateFraction=0.3,
            screen=SCREEN,
        )
    )
    if (
        env["box"]["build"].get("taskSizeRanges") != ranges
        or env["box"]["build"].get("sizeInterval") != 0.05
        or env["box"]["reset"].get("randomAssignment") is not False
        or (env.get("sizeAwareRsi") != rsi)
    ):
        raise ValueError("Invalid size-aware asset / RSI contract")
    if env["box"]["reset"].get("groundStandaloneSitClimb") is not True:
        raise ValueError("Stage1 requires grounded standalone SIT/CLIMB")
    for name in ("HOLDING_AT", "HOLDING_ON_TOP"):
        expected = (
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.1]
            if native_sampling
            else [0.4, 0.0, 0.0, 0.0, 0.0, 0.1, 0.4, 0.1]
        )
        if env["templateRsi"][name] != expected:
            raise ValueError(
                "Size-aware placement RSI requires "
                + ("50/10/30/10" if native_sampling else "40/10/40/10")
            )


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
