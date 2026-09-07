#!/bin/bash
# Open the newest immutable A2 checkpoint in the noVNC viewer.

. "$(dirname "$0")/runtime_env.sh"

CHECKPOINT=$(find output/ma_carry_edge_mlp -type f -name 'HumanoidMAEdgeMLP_*.pth' \
    | sort -V | tail -1)
if [ -z "$CHECKPOINT" ]; then
    echo "no intermediate A2 checkpoint found" >&2
    exit 1
fi

export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export HEADLESS=0

exec sh tokenhsi/scripts/multi_agent/run-gui.sh \
    sh tokenhsi/scripts/multi_agent/ma_carry_edge_mlp_test.sh \
    "$CHECKPOINT" 2 1 3 20
