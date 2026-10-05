#!/usr/bin/env bash
# Source in Bash. GPU-specific MPS helpers; no sudo, GPU reset, or global pkill.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Load with: source $0" >&2
    exit 2
fi

# Replace conflicting aliases in this shell only; no startup files are edited.
unalias mps_start mps_stop mps_status mps_use mps_run 2>/dev/null || true

_mps_gpu_env() {
    local gpu="${1:-}" uuid tool root
    if [[ ! "$gpu" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo 'Specify one physical GPU index, e.g. 6.' >&2
        return 2
    fi
    for tool in nvidia-smi nvidia-cuda-mps-control timeout flock; do
        command -v "$tool" >/dev/null || return 127
    done
    uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader) || return
    if [[ ! "$uuid" =~ ^GPU-[[:xdigit:]]{8}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{12}$ ]]; then
        echo "Could not resolve one UUID for GPU $gpu." >&2
        return 1
    fi
    root="${MPS_GPU_ROOT:-/tmp/mps-test-${UID}}"
    if [[ "$root" != /* || "$root" == / || "$root" == /tmp || "$root" == /tmp/nvidia-mps ]]; then
        echo 'MPS_GPU_ROOT must be a dedicated absolute directory owned by you.' >&2
        return 2
    fi
    export CUDA_VISIBLE_DEVICES="$uuid"
    export TOKENHSI_GPU="$uuid"
    export CUDA_MPS_PIPE_DIRECTORY="${root}/gpu${gpu}/pipe"
    export CUDA_MPS_LOG_DIRECTORY="${root}/gpu${gpu}/log"
}

_mps_check_paths() {
    local gpu_dir="${CUDA_MPS_PIPE_DIRECTORY%/pipe}" path
    for path in "${gpu_dir%/*}" "$gpu_dir" "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; do
        if [[ -L "$path" || ( -e "$path" && ( ! -d "$path" || ! -O "$path" ) ) ]]; then
            echo "Refusing MPS path not owned by you or using a symlink: $path" >&2
            return 1
        fi
    done
}

_mps_control() {
    # Explicit on every call: never fall back to a global/default pipe or GPU set.
    command env CUDA_VISIBLE_DEVICES="$TOKENHSI_GPU" \
        CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
        CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_LOG_DIRECTORY" \
        nvidia-cuda-mps-control "$@"
}

_mps_validate_daemon() {
    local file="$CUDA_MPS_PIPE_DIRECTORY/nvidia-cuda-mps-control.pid" pid entry
    local daemon_gpu='' daemon_pipe='' daemon_log=''
    _mps_check_paths || return
    if [[ ! -f "$file" || -L "$file" || ! -O "$file" ]]; then
        echo "No verified MPS daemon at $CUDA_MPS_PIPE_DIRECTORY; run mps_start GPU." >&2
        return 1
    fi
    pid=$(<"$file")
    if [[ ! "$pid" =~ ^[1-9][0-9]*$ || ! -O "/proc/$pid" || ! -r "/proc/$pid/environ" ]]; then
        echo 'MPS PID is stale or belongs to another user; left untouched.' >&2
        return 1
    fi
    while IFS= read -r -d '' entry; do
        case "$entry" in
            CUDA_VISIBLE_DEVICES=*) daemon_gpu=${entry#*=} ;;
            CUDA_MPS_PIPE_DIRECTORY=*) daemon_pipe=${entry#*=} ;;
            CUDA_MPS_LOG_DIRECTORY=*) daemon_log=${entry#*=} ;;
        esac
    done < "/proc/$pid/environ"
    if [[ "$daemon_gpu" != "$TOKENHSI_GPU" || "$daemon_pipe" != "$CUDA_MPS_PIPE_DIRECTORY" || "$daemon_log" != "$CUDA_MPS_LOG_DIRECTORY" ]]; then
        echo "MPS GPU/path mismatch for PID $pid; refusing to use or stop it." >&2
        return 1
    fi
}

_mps_timed_control() {
    command env CUDA_VISIBLE_DEVICES="$TOKENHSI_GPU" \
        CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
        CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_LOG_DIRECTORY" \
        timeout 5 nvidia-cuda-mps-control "$@"
}

_mps_is_up() {
    printf 'get_server_list\n' | _mps_timed_control >/dev/null 2>&1
}

_mps_ensure_server() {
    local servers attempt
    servers=$(printf 'get_server_list\n' | _mps_timed_control) || return
    if [[ -z "$servers" ]]; then
        # Launch on the already-validated daemon's ONE visible GPU, as this UID.
        printf 'start_server -uid %s\n' "$UID" | _mps_timed_control || return
    fi
    for attempt in {1..20}; do
        servers=$(printf 'get_server_list\n' | _mps_timed_control) || return
        if [[ "$servers" =~ ^[0-9]+([[:space:]][0-9]+)*$ ]]; then
            echo "MPS GPU server ready: $servers"
            return 0
        fi
        sleep .1
    done
    echo 'MPS control daemon is up, but its GPU server did not start.' >&2
    return 1
}

_mps_archive_stale_pipe() {
    # NVIDIA can leave socket/lock/log FIFO files after a normal quit. Recover
    # known layout, without a PID, reachable daemon or live kernel socket.
    local path backup probe_status
    _mps_check_paths || return
    [[ ! -e "$CUDA_MPS_PIPE_DIRECTORY/nvidia-cuda-mps-control.pid" &&
       ! -L "$CUDA_MPS_PIPE_DIRECTORY/nvidia-cuda-mps-control.pid" &&
       -r /proc/net/unix ]] || return 1
    if _mps_is_up; then
        echo 'MPS endpoint still responds; left untouched.' >&2
        return 1
    else
        probe_status=$?
        [[ "$probe_status" == 1 ]] || return 1
    fi
    if awk -v path="$CUDA_MPS_PIPE_DIRECTORY/" \
        'index($0, " " path) { found=1 } END { exit !found }' /proc/net/unix; then
        echo 'Live MPS socket exists; left untouched.' >&2
        return 1
    fi
    for path in "$CUDA_MPS_PIPE_DIRECTORY"/* "$CUDA_MPS_PIPE_DIRECTORY"/.[!.]* "$CUDA_MPS_PIPE_DIRECTORY"/..?*; do
        [[ -e "$path" || -L "$path" ]] || continue
        if [[ -L "$path" || ! -O "$path" ]]; then
            echo 'Unverified MPS pipe contents; left untouched.' >&2
            return 1
        fi
        case "${path##*/}" in
            control|control_privileged) [[ -S "$path" ]] || return 1 ;;
            control_lock) [[ -f "$path" ]] || return 1 ;;
            log) [[ -p "$path" ]] || return 1 ;;
            *) echo "Unknown MPS pipe file: $path; left untouched." >&2; return 1 ;;
        esac
    done
    # Preserve the stale files for inspection; never unlink a running endpoint.
    backup=$(mktemp -d "${CUDA_MPS_PIPE_DIRECTORY%/pipe}/pipe.stale.XXXXXXXX") || return
    mv -T -- "$CUDA_MPS_PIPE_DIRECTORY" "$backup" || return
    mkdir -m 700 -- "$CUDA_MPS_PIPE_DIRECTORY" || return
    echo "Archived inactive MPS sockets: $backup"
}

mps_use() {
    if [[ "$#" != 1 ]]; then
        echo 'Usage: mps_use GPU' >&2
        return 2
    fi
    if ! ( _mps_gpu_env "$1" && _mps_validate_daemon && _mps_is_up ); then
        echo "Cannot connect to MPS for GPU $1; check mps_status or start it first." >&2
        return 1
    fi
    _mps_gpu_env "$1" || return
    echo "Selected GPU $1: $CUDA_MPS_PIPE_DIRECTORY"
}

mps_start() {
    if [[ "$#" != 1 ]]; then
        echo 'Usage: mps_start GPU' >&2
        return 2
    fi
    (
        _mps_gpu_env "$1" || exit
        _mps_check_paths || exit
        nvidia-smi -i "$1" --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || exit
        umask 077
        mkdir -p -- "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" || exit
        _mps_check_paths || exit
        local lock="${CUDA_MPS_PIPE_DIRECTORY%/pipe}/helper.lock"
        if [[ -L "$lock" || ( -e "$lock" && ( ! -f "$lock" || ! -O "$lock" ) ) ]]; then
            echo 'Unsafe MPS lock; refusing to start.' >&2
            exit 1
        fi
        exec 9>"$lock" || exit
        flock -w 5 9 || exit
        if [[ -e "$CUDA_MPS_PIPE_DIRECTORY/nvidia-cuda-mps-control.pid" ]]; then
            _mps_validate_daemon && _mps_is_up || exit
            echo "MPS for GPU $1 is already running."
        else
            if [[ -e "$CUDA_MPS_PIPE_DIRECTORY/control" ||
                  -e "$CUDA_MPS_PIPE_DIRECTORY/control_privileged" ||
                  -e "$CUDA_MPS_PIPE_DIRECTORY/control_lock" ||
                  -L "$CUDA_MPS_PIPE_DIRECTORY/nvidia-cuda-mps-control.pid" ]]; then
                _mps_archive_stale_pipe || exit
            fi
            _mps_control -d 9>&- || exit
            _mps_validate_daemon && _mps_is_up || exit
        fi
        _mps_ensure_server 9>&- || exit
    ) || return
    mps_use "$1"
}

mps_stop() (
    if [[ "$#" != 1 && !( "$#" == 2 && "$2" == --force ) ]]; then
        echo 'Usage: mps_stop GPU [--force]' >&2
        exit 2
    fi
    _mps_gpu_env "$1" || exit
    _mps_validate_daemon && _mps_is_up || exit
    if [[ "${2:-}" == --force ]]; then
        printf 'quit -t 1\n' | _mps_control
    else
        printf 'quit\n' | _mps_control
    fi
)

mps_status() (
    if [[ "$#" != 1 ]]; then
        echo 'Usage: mps_status GPU' >&2
        exit 2
    fi
    _mps_gpu_env "$1" || exit
    _mps_validate_daemon && _mps_is_up || exit
    local output
    output=$(printf 'ps\n' | _mps_timed_control) || exit
    if [[ -z "$output" || "$output" == 'Server not found' ]]; then
        echo "MPS for GPU $1 is ready; no connected CUDA clients."
    else
        printf '%s\n' "$output"
    fi
)

mps_run() (
    if [[ "$#" -lt 2 ]]; then
        echo 'Usage: mps_run GPU COMMAND [ARG...]' >&2
        exit 2
    fi
    mps_use "$1" || exit
    shift
    "$@"
)
