#!/bin/bash
# View a checkpoint trained by train_retreat_only.sh.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

export STACK_PLANNER_VIEW_TASK_CLASS=HumanoidMAStackPlannerRetreatView
export STACK_PLANNER_FRESH_START=0
export STACK_RETREAT_ONLY_CLEARANCE=${STACK_RETREAT_ONLY_CLEARANCE:-1.5}
export STACK_RETREAT_ONLY_A2_DISTANCE=${STACK_RETREAT_ONLY_A2_DISTANCE:-3.0}
export STACK_EPISODE_LENGTH=${STACK_EPISODE_LENGTH:-300}

exec bash "$ROOT/TokenHSI-coord/stack_planner/view.sh" "$@"
