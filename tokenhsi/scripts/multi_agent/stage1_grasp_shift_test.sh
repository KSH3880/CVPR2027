#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-5}
. "$(dirname "$0")/runtime_env.sh"
cd "$TOKENHSI_ROOT"
CKPT=${1:-stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth}
export SHIFT=${SHIFT:-1}
unset TOKENHSI_INFERENCE_CONFIG
case "${TASK_GRAPH:-joint_carry_at}" in
  joint_carry_at|joint_carry_ontop) ;;
  *) echo 'Use joint_carry_at or joint_carry_ontop for this shared-object toy' >&2; exit 1 ;;
esac
set -- --task_graph "${TASK_GRAPH:-joint_carry_at}" --task_camera "${TASK_CAMERA:-stack}"
if [ "${HEADLESS:-0}" != 0 ]; then set -- "$@" --headless; fi
python tokenhsi/scripts/prepare_joint_carry.py
python tokenhsi/stage1_grasp_shift.py --task HumanoidMACarry \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml --checkpoint "$CKPT" \
  --num_agents 2 --num_envs "${NUM_ENVS:-1}" --num_objects 4 \
  --eval_repeats "${REPEATS:-10}" --eval_skills "${EVAL_SKILLS:-loco}" --eval_skill_probs 1.0 \
  --seed "${SEED:-42}" --episode_length "${EPISODE_LENGTH:-300}" \
  --output_path "${OUTPUT_PATH:-output_etc/stage1_shared_grasp_shift_${SHIFT}}" \
  --no_video --test --eval "$@"
