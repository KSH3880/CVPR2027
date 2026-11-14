#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/runtime_env.sh"
cd "$TOKENHSI_ROOT"
CKPT=${1:-output/stage2_before_total/Stage2BeforeTotal_06-22-22-14/nn/Stage2BeforeTotal_00022000.pth}
[ -f "$CKPT" ] || { echo "Checkpoint not found: $CKPT" >&2; exit 1; }
. "$SCRIPT_DIR/output_paths.sh"
tokenhsi_select_output before_chain3 eval
set -- --task HumanoidMABeforePush --cfg_train tokenhsi/data/cfg/train/rlg/stage2_before_total.yaml \
  --checkpoint "$CKPT" --num_agents 3 --num_objects 6 --num_envs 1 \
  --eval_repeats "${2:-10}" --episode_length "${EPISODE_LENGTH:-1800}" \
  --task_graph before_stack --before_box_size original --task_camera stack \
  --eval_skills loco --eval_skill_probs 1.0 --test --eval --no_video \
  --motion_file tokenhsi/data/dataset_stage1_jointpush.yaml --output_path "$OUTPUT_PATH" --seed "${SEED:-9}"
if [ "${HEADLESS:-0}" != 0 ]; then set -- "$@" --headless; fi
exec python tokenhsi/view_before_chain.py "$@"
