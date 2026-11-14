#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/runtime_env.sh"
SOURCE=${RESUME_CHECKPOINT:-output/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg/Stage2BeforePushNear50Pair50Source5VirtualGoal5GoalRouteWaitPenaltyV2RelFusionZeroBg_05-22-58-01/nn/Stage2BeforePushNear50Pair50Source5VirtualGoal5GoalRouteWaitPenaltyV2RelFusionZeroBg_00021000.pth}
RESUME_CHECKPOINT=$(python tokenhsi/utils/before_fusion_resume.py "$SOURCE")
export RESUME_CHECKPOINT
exec bash "$SCRIPT_DIR/launch.sh" stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_human3_resume21k train "$@"
