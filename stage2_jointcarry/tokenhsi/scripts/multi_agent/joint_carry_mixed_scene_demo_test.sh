#!/bin/bash
set -eu
. "$(dirname "$0")/runtime_env.sh"
if [ $# -ne 1 ] || [ ! -f "$1" ]; then
    echo "usage: $0 <CPA-Release1-or-LocoPlatform-checkpoint.pth>" >&2
    exit 1
fi
CKPT=$1
DEMO_AGENTS=${DEMO_AGENTS:-6}
DEFAULT_OUTPUT=output/joint_carry_mixed_scene_demo
if [ "$DEMO_AGENTS" = 4 ]; then DEFAULT_OUTPUT=${DEFAULT_OUTPUT}_4agents; fi
EXPERIMENT=$(python - "$CKPT" <<'PY'
import sys
import torch

weights = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
suffix = '_locoplatform' if weights.get('joint_carry_loco_platform') is not None else ''
print('stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1' + suffix + '_task_embedding')
PY
)
echo "[mixed-scene] config: $EXPERIMENT"
set --
if [ "${HEADLESS:-1}" != 0 ]; then set -- --headless; fi
python tokenhsi/scripts/prepare_joint_carry.py
python tokenhsi/joint_carry_demo.py --task HumanoidMACarry \
    --cfg_train "tokenhsi/data/cfg/train/rlg/amp_ma_${EXPERIMENT}.yaml" \
    --cfg_env "tokenhsi/data/cfg/multi_agent/approach_${EXPERIMENT}.yaml" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml --checkpoint "$CKPT" \
    --num_agents "$DEMO_AGENTS" --num_envs 1 --num_objects 8 \
    --eval_repeats "${REPEATS:-10}" --eval_skills loco --eval_skill_probs 1.0 \
    --episode_length "${EPISODE_LENGTH:-600}" --seed "${SEED:-42}" \
    --output_path "${OUTPUT_PATH:-$DEFAULT_OUTPUT}" \
    --task_graph joint_carry_at --task_camera stack --no_video --test --eval "$@"
