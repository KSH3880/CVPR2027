#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CKPT=$1
shift
bash "$SCRIPT_DIR/evaluate.sh" stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg "$CKPT" --profile general "$@"
