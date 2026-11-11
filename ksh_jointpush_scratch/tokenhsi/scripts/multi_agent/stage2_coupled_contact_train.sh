#!/bin/bash
# stage2_coupled_contact: train
export RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-output/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg/Stage2CoupledPushIndividualGoalSolo5Carry5GroundRelFusionZeroBg_03-14-49-26/nn/Stage2CoupledPushIndividualGoalSolo5Carry5GroundRelFusionZeroBg_00017000.pth}
exec bash "$(dirname "$0")/launch.sh" stage2_coupled_contact train "$@"
