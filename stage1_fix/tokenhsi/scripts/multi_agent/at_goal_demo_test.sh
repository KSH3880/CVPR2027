#!/bin/bash
set -eu
. "$(dirname "$0")/runtime_env.sh"
CKPT=${1:-distill_pth/ApproachScenarioStage1RescueAtKLClimb50_00008000.pth}
if [ ! -f "$CKPT" ]; then
    echo "checkpoint not found: $CKPT" >&2
    exit 1
fi
set --
if [ "${HEADLESS:-1}" != 0 ]; then set -- --headless; fi
python tokenhsi/at_goal_demo.py --task HumanoidMACarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation.yaml \
    --cfg_env tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_self_sum.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml --checkpoint "$CKPT" \
    --num_agents 2 --num_envs 1 --num_objects 3 \
    --eval_repeats "${REPEATS:-1}" --eval_skills loco --eval_skill_probs 1.0 \
    --episode_length "${EPISODE_LENGTH:-600}" --seed "${SEED:-42}" \
    --output_path "${OUTPUT_PATH:-output/at_goal_demo}" \
    --task_graph random_scenario --task_camera stack --no_video --test --eval "$@"
