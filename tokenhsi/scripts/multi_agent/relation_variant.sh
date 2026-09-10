#!/bin/bash
# Sourced by relation train/test scripts after runtime_env.sh.
# Keep original defaults untouched; opt into isolated state2 / state2_signed experiments.
case "${RELATION_VARIANT:-v0}" in
    v0)
        RELATION_ENV_CFG=tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry_relation.yaml
        RELATION_OUTPUT_DEFAULT=output/ma_carry_relation_v0
        RELATION_EXPERIMENT=
        ;;
    state2)
        RELATION_ENV_CFG=tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry_relation_state2.yaml
        RELATION_OUTPUT_DEFAULT=output/ma_carry_relation_state2
        RELATION_EXPERIMENT=CarryRelationState2
        # Match the already-running v0 experiment, not its YAML's random-seed sentinel.
        SEED=${SEED:-9896}
        ;;
    state2_signed)
        RELATION_ENV_CFG=tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry_relation_state2_signed.yaml
        RELATION_OUTPUT_DEFAULT=output/ma_carry_relation_state2_signed
        RELATION_EXPERIMENT=CarryRelationState2Signed
        SEED=${SEED:-9896}
        ;;
    *)
        echo "Unknown RELATION_VARIANT: $RELATION_VARIANT (expected v0, state2 or state2_signed)" >&2
        return 1
        ;;
esac
