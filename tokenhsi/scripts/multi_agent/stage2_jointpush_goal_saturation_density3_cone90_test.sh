#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-5}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_jhh}
. "$(dirname "$0")/runtime_env.sh"
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
tokenhsi_select_output stage2_jointpush_goal_saturation_density3_cone90 eval
JOINTPUSH_CHECKPOINT=${1:-pth/Stage1JointPushDensity3Cone90Finetune_00011000.pth}
set -- --num_agents "${2:-4}" --num_envs "${3:-1}" --num_objects "${4:-8}" --eval_repeats "${5:-10}"
set -- "$@" --task_graph "${TASK_GRAPH:-random_scenario}" --task_camera "${TASK_CAMERA:-stack}" --seed "${SEED:-42}"
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
if [ "${SIM_DEVICE:-}" = cpu ]; then set -- "$@" --sim_device cpu --pipeline cpu --rl_device cpu; fi
python "${JOINTPUSH_ENTRYPOINT:-tokenhsi/run.py}" --task HumanoidMAStage2JointPush \
  --cfg_env tokenhsi/data/cfg/multi_agent/stage2_jointpush_goal_saturation_density3_cone90.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage2_jointpush_goal_saturation_density3_cone90.yaml \
  --motion_file tokenhsi/data/dataset_stage1_jointpush.yaml \
  --checkpoint "$JOINTPUSH_CHECKPOINT" --output_path "$OUTPUT_PATH" --test --eval --no_video "$@"
