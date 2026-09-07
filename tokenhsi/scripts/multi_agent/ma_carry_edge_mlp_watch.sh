#!/bin/bash
# Watch the A2 edge-embedding policy train in a viewer.
#   sh tokenhsi/scripts/multi_agent/run-gui.sh \
#     sh tokenhsi/scripts/multi_agent/ma_carry_edge_mlp_watch.sh 2 4 3 [checkpoint.pth]

. "$(dirname "$0")/runtime_env.sh"

NUM_AGENTS=${1:-2}
NUM_ENVS=${2:-4}
NUM_OBJECTS=${3:-3}
CKPT=${4:-}
if [ -n "${MAX_ITERATIONS:-}" ]; then
    set -- --max_iterations "$MAX_ITERATIONS"
else
    set --
fi
if [ -n "$CKPT" ]; then
    if [ ! -f "$CKPT" ]; then
        echo "checkpoint not found: $CKPT" >&2
        exit 1
    fi
    set -- "$@" --checkpoint "$CKPT" --resume 1
fi

python ./tokenhsi/run.py --task HumanoidMACarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_carry_watch_edge_mlp.yaml \
    --cfg_env tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --num_envs ${NUM_ENVS} \
    --num_agents ${NUM_AGENTS} \
    --num_objects ${NUM_OBJECTS} \
    --no_video \
    --output_path output/ma_carry_edge_mlp_watch "$@"
