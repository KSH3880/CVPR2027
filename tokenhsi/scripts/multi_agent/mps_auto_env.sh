#!/bin/sh
# Sourced by runtime_env.sh. Attach only to this user's verified per-GPU MPS.
# Discovery is read-only: this file never starts, stops or repairs a daemon.
tokenhsi_attach_mps() {
    tokenhsi_mps_root=${MPS_GPU_ROOT:-/tmp/mps-test-$(id -u)}
    # Most machines do not use this helper; leave their normal CUDA setup alone.
    if [ ! -d "$tokenhsi_mps_root" ] && [ -z "${CUDA_MPS_PIPE_DIRECTORY:-}" ]; then
        return 0
    fi
    tokenhsi_mps_gpu=${TOKENHSI_GPU:-${CUDA_VISIBLE_DEVICES:-0}}
    case "$tokenhsi_mps_gpu" in
        ''|*,*)
            if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]; then
                echo '[MPS] A per-GPU MPS connection requires exactly one GPU.' >&2
                return 1
            fi
            return 0 ;;
    esac
    tokenhsi_mps_index=$(nvidia-smi -i "$tokenhsi_mps_gpu" \
        --query-gpu=index --format=csv,noheader,nounits) || return 1
    case "$tokenhsi_mps_index" in
        ''|*[!0-9]*) echo '[MPS] Could not resolve exactly one physical GPU.' >&2; return 1 ;;
    esac
    tokenhsi_mps_pipe="$tokenhsi_mps_root/gpu$tokenhsi_mps_index/pipe"
    if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] && [ "$CUDA_MPS_PIPE_DIRECTORY" != "$tokenhsi_mps_pipe" ]; then
        echo "[MPS] Inherited pipe does not match GPU $tokenhsi_mps_index; refusing to run." >&2
        echo '[MPS] Use mps_use for the intended GPU in this terminal first.' >&2
        return 1
    fi
    if [ ! -e "$tokenhsi_mps_pipe/nvidia-cuda-mps-control.pid" ] && \
       [ ! -L "$tokenhsi_mps_pipe/nvidia-cuda-mps-control.pid" ] && \
       [ -z "${CUDA_MPS_PIPE_DIRECTORY:-}" ]; then
        return 0
    fi
    # Validate before exporting anything to the learner. Even a responding
    # global or differently scoped daemon must not be accepted here.
    tokenhsi_mps_uuid=$(bash -c '
        source "$1" || exit
        _mps_gpu_env "$2" && _mps_validate_daemon && _mps_is_up || exit
        printf "%s\n" "$TOKENHSI_GPU"
    ' -- "$TOKENHSI_ROOT/mps/shell.sh" "$tokenhsi_mps_index") || {
        echo "[MPS] GPU $tokenhsi_mps_index daemon could not be verified; training was not started." >&2
        return 1
    }
    export TOKENHSI_GPU="$tokenhsi_mps_uuid"
    export CUDA_VISIBLE_DEVICES="$tokenhsi_mps_uuid"
    export CUDA_MPS_PIPE_DIRECTORY="$tokenhsi_mps_pipe"
    export CUDA_MPS_LOG_DIRECTORY="$tokenhsi_mps_root/gpu$tokenhsi_mps_index/log"
    echo "[MPS] GPU $tokenhsi_mps_index auto-connect: $CUDA_MPS_PIPE_DIRECTORY"
}

tokenhsi_attach_mps || return 1
