#!/bin/bash
# Usage: bash .../eval_retreat.sh <training-tag> <planner.pth> <eval-tag>
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TRAIN_TAG=${1:?training tag required}
PLANNER=$(realpath -- "${2:?planner checkpoint required}")
EVAL_TAG=${3:?unique evaluation tag required}
case "$TRAIN_TAG:$EVAL_TAG" in *[!A-Za-z0-9_.:-]*) echo 'invalid tag' >&2; exit 2;; esac
EVAL_GPU=${MA_GPU:-7}
EVAL_ENVS=${STACK_PLANNER_EVAL_ENVS:-64}
EVAL_SEED=${STACK_PLANNER_EVAL_SEED:-0}
EVAL_SCEN=${STACK_PLANNER_EVAL_SCEN:-free}
case "$EVAL_SCEN" in
    free|cross|parallel|solo) ;;
    *) echo "invalid STACK_PLANNER_EVAL_SCEN: $EVAL_SCEN" >&2; exit 2;;
esac
SIDECAR="$ROOT/runs/stack_planner/$TRAIN_TAG/run.env"
[ -f "$SIDECAR" ] || { echo "missing training sidecar: $SIDECAR" >&2; exit 5; }
set -a
. "$SIDECAR"
set +a
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$EVAL_GPU"
export STACK_PLANNER_EVAL_CKPT="$PLANNER"
export STACK_PLANNER_EVAL_REPLAN=${STACK_PLANNER_EVAL_REPLAN:-$STACK_PLANNER_LOW_STEPS}
export STACK_PLANNER_EVAL_OUTPUT="$ROOT/runs/stack_planner_eval/$EVAL_TAG"
# Evaluation deliberately covers small/medium/large bottom/top box pairs.
# Box geometry is fixed when Isaac Gym creates an env, so diversity must be
# assigned across the vector rather than attempted at episode reset.
export STACK_EVAL_BOX_GRID=${STACK_PLANNER_EVAL_BOX_GRID:-1}
export MS_SCEN="$EVAL_SCEN"
# Do not allow an unrelated interactive shell layout to override the explicit
# planner-evaluation scenario. HumanoidMASteerCarry derives these from MS_SCEN.
unset MA_LAYOUT MA_LAYOUT_D MA_LAYOUT_S MA_LAYOUT_L
unset STACK_FIXED_BOX_SIZE_IDS
[ ! -e "$STACK_PLANNER_EVAL_OUTPUT" ] || { echo 'evaluation output exists' >&2; exit 3; }
for file in "$executor" "$stage1" "$cfg" "$PLANNER"; do
    [ -f "$file" ] || { echo "missing: $file" >&2; exit 1; }
done
if [ -z "${CONDA_BASE:-}" ]; then CONDA_BASE=$(conda info --base); fi
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi_juan}"
. "$ROOT/TokenHSI-coord/stack_planner/physx_cuda_compat.sh"
echo "stack planner eval: scenario=$EVAL_SCEN box_grid=$STACK_EVAL_BOX_GRID envs=$EVAL_ENVS seed=$EVAL_SEED"
cd "$ROOT/TokenHSI-coord"
python -u -m stack_planner.eval_retreat \
    --test --headless --task HumanoidMAStackPlannerTrain \
    --cfg_train "$ROOT/TokenHSI-masteer/tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml" \
    --cfg_env "$cfg" --num_envs "$EVAL_ENVS" --seed "$EVAL_SEED" \
    --motion_file "$ROOT/TokenHSI-masteer/tokenhsi/data/dataset_loco_sit_carry_climb.yaml" \
    --hrl_checkpoint "$stage1" --checkpoint "$executor" \
    --output_path "$ROOT/runs/stack_planner_eval_executor_unused/$EVAL_TAG"
