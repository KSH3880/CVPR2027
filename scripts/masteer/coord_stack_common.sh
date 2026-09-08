#!/bin/bash
# Sourced only by the new coord-stack scripts; no effect on existing launchers.

coord_stack_setup() {
    if [ -z "${COORD_CKPT:-}" ]; then
        local candidates=() candidate
        for candidate in "$ROOT"/runs/coord/c13*/coord_c2_latest.pth; do
            [ ! -f "$candidate" ] || candidates+=("$candidate")
        done
        if [ "${#candidates[@]}" -ne 1 ]; then
            echo "COORD_CKPT가 필요합니다. C13 자동 후보 ${#candidates[@]}개 (정확히 1개일 때만 자동 선택)." >&2
            printf '  %s\n' "${candidates[@]}" >&2
            return 2
        fi
        COORD_CKPT=${candidates[0]}
    fi
    [ -f "$COORD_CKPT" ] || { echo "coordinator PTH 없음: $COORD_CKPT" >&2; return 2; }
    export COORD_CKPT
    COORD_CKPT=$(realpath -- "$COORD_CKPT")
    export COORD_REPLAN_STEPS=${COORD_REPLAN_STEPS:-6}
    export COORD_COMMAND_ACCEL=${COORD_COMMAND_ACCEL:-0.75}
    export MS_ZERO=0
    export MS_CLIP=1
    export STACK_CARRY_REHEARSAL_PROB=0
    export STACK_END_ON_A2_RESUME=0
    export CUDA_DEVICE_ORDER=PCI_BUS_ID

    # Accept an explicit interpreter, an activated conda environment, or the
    # local TokenHSI conda environment. No host/user-specific hardcoded paths.
    if [ -z "${TOKENHSI_PYTHON:-}" ]; then
        if [ -n "${CONDA_PREFIX:-}" ]; then
            TOKENHSI_PYTHON="$CONDA_PREFIX/bin/python"
        else
            local base=${CONDA_BASE:-}
            if [ -z "$base" ] && command -v conda >/dev/null 2>&1; then
                base=$(conda info --base)
            fi
            if [ -z "$base" ]; then
                for candidate in "$HOME/anaconda3" "$HOME/miniconda3"; do
                    if [ -d "$candidate" ]; then base=$candidate; break; fi
                done
            fi
            TOKENHSI_PYTHON="$base/envs/${TOKENHSI_CONDA_ENV:-tokenhsi118}/bin/python"
        fi
    fi
    [ -x "$TOKENHSI_PYTHON" ] || {
        echo "TOKENHSI_PYTHON에 Isaac Gym 환경의 python 절대 경로를 지정하세요: $TOKENHSI_PYTHON" >&2
        return 2
    }
    export TOKENHSI_PYTHON
    local python_prefix
    python_prefix=$(cd -- "$(dirname -- "$TOKENHSI_PYTHON")/.." && pwd)
    export PATH="$python_prefix/bin:$PATH"
    export LD_LIBRARY_PATH="$python_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "coord-stack planner: $COORD_CKPT"
    echo "coord-stack python:  $TOKENHSI_PYTHON"
    # Validate the separate model contract on CPU before allocating the simulator.
    "$TOKENHSI_PYTHON" - "$ROOT/TokenHSI-coord" "$COORD_CKPT" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from coordinator.sequential_bridge import load_planner
model, payload = load_planner(sys.argv[2], "cpu")
print("coord-stack checkpoint: schema={} step={}".format(
    payload["schema_version"], payload.get("step")))
PY
}
