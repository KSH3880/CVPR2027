#!/bin/bash
set -eu
. "$(dirname "$0")/runtime_env.sh"
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
tokenhsi_select_output stage1_jointpush_density3_cone90_finetune eval
JOINTPUSH_CHECKPOINT=${1:-pth/Stage1PushFinetune_00014000.pth}
set -- --num_agents "${2:-2}" --num_envs "${3:-1}" --num_objects "${4:-4}" --eval_repeats "${5:-10}"
set -- "$@" --task_graph "${TASK_GRAPH:-random_scenario}" --task_camera "${TASK_CAMERA:-stack}" --seed "${SEED:-42}"
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
if [ "${SIM_DEVICE:-}" = cpu ]; then set -- "$@" --sim_device cpu --pipeline cpu --rl_device cpu; fi
python tokenhsi/run.py --task HumanoidMAStage1JointPush \
  --cfg_env tokenhsi/data/cfg/multi_agent/stage1_jointpush_density3_cone90_finetune.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage1_jointpush_density3_cone90_finetune.yaml \
  --motion_file tokenhsi/data/dataset_stage1_jointpush.yaml \
  --checkpoint "$JOINTPUSH_CHECKPOINT" --output_path "$OUTPUT_PATH" --test --eval --no_video "$@"
