JOINT_PRESETS = ("random_scenario", "joint_carry_at", "joint_carry_ontop")


def validate_env(env):
    from utils.joint_carry_platform import LOCO_PLATFORM

    if env["jointCarryRsi"].get("locoPlatform") != LOCO_PLATFORM:
        raise ValueError("Unsupported joint loco platform contract")
    from utils.joint_carry_goals import GOAL360

    if env["jointCarryRsi"].get("goalSampling") != GOAL360:
        raise ValueError("Unsupported joint goal sampling contract")
    from utils.joint_carry_amp import validate_training

    validate_training(env)
    from utils.mixed_carry_spec import validate_sampler
    from utils.size_rsi import SCREEN

    validate_sampler(env["relationGraph"], env["numAgents"], env["numObjects"])
    if (
        env.get("sizeAwareRsi")
        != dict(
            cacheDirectory="output/rsi_cache",
            lateProbability=0.7,
            lateFraction=0.3,
            screen=SCREEN,
        )
        or env.get("independentCarryRsi") != [0.4, 0.0, 0.0, 0.0, 0.0, 0.1, 0.4, 0.1]
        or env["relationReward"]["joint_carry"].get("independent")
        != {
            "environment_fraction": 0.2,
            "placement_gate": "own_current_holding",
            "rsi": "stage1_size_aware",
        }
    ):
        raise ValueError(
            "Mixed carry requires size RSI and the independent holding gate"
        )
    build, reset = env["box"]["build"], env["box"]["reset"]
    if (
        build["baseSize"] != [0.52, 0.8, 0.4]
        or build["jointSupportSize"] != [0.52, 0.8, 0.3]
        or build["randomSize"]
        or build["randomDensity"]
        or reset["randomAssignment"]
        or reset["randomHeight"]
    ):
        raise ValueError(
            "Paired RSI requires fixed payload/support assets and canonical binding"
        )
    if (
        env.get("ampTaskConditioning") is not True
        or env["jointCarryRsi"]["ampHistory"] != "repeat_reset_state"
    ):
        raise ValueError(
            "Joint carry requires carry-conditioned AMP and matching reset history"
        )
    expected = [1 / 3, 0.0, 0.0, 0.0, 1 / 3, 1 / 6, 0.0, 1 / 6]
    if (
        env["skill"]
        != [
            "loco",
            "sit",
            "climb",
            "climbNoRSI",
            "omomo",
            "pickUp",
            "carryWith",
            "putDown",
        ]
        or env["skillDiscProb"] != expected
        or env["templateRsi"]["HOLDING_AT"] != [0.4, 0.0, 0.0, 0.0, 0.0, 0.1, 0.4, 0.1]
        or env["templateRsi"]["HOLDING_ON_TOP"]
        != [0.5, 0.0, 0.0, 0.0, 0.0, 0.1, 0.4, 0.0]
    ):
        raise ValueError(
            "Joint carry RSI/AMP skill probabilities do not match the experiment"
        )
