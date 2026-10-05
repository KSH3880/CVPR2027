# Codex에게 전달하는 GPU별 MPS 단축 명령 설치 안내

**이 MD 파일 하나만 다른 서버의 Codex에게 전달하면 된다. 아래에 설치할 전체 코드가 포함되어 있다. 이 저장소나 conda 환경은 필요 없다.**

## Codex에게 보낼 요청

> 이 문서에 따라 GPU별 MPS 단축 명령을 설치해줘. 개인 설치 디렉터리와 ~/.bashrc의 자동 로드 설정 수정은 허용한다. 기존 설정은 보존하고 백업해줘. 기존 학습이나 MPS를 종료·재시작하지 말고, GPU compute mode·드라이버·시스템 서비스는 변경하지 마. 설치 후 새 터미널에서 mps_start 6, mps_stop 6처럼 사용할 수 있게 해줘. 현재 터미널에서 필요한 마지막 한 줄도 알려줘.

## 설치 후 사용자가 쓰는 명령

| 하고 싶은 일 | 명령 |
| --- | --- |
| GPU 6 MPS 시작 + 현재 터미널에서 선택 | `mps_start 6` |
| GPU 4 MPS 시작 + 현재 터미널에서 선택 | `mps_start 4` |
| GPU 6 MPS 정상 종료 | `mps_stop 6` |
| GPU 4 MPS 강제 종료 | `mps_stop 4 --force` |
| GPU 6에 연결된 작업 확인 | `mps_status 6` |
| 새 터미널에서 이미 켜진 GPU 6 MPS 선택 | `mps_use 6` |
| 한 명령만 GPU 6 MPS로 실행 | `mps_run 6 python train.py` |

GPU 번호는 서버에 맞게 바꾼다. `mps_start`는 지정 GPU의 제어 데몬과 GPU 서버를 학습 없이도 즉시 시작한다. 이미 같은 경로에 켜져 있으면 재시작하지 않고 재사용한다. 정상 종료는 연결된 작업이 끝날 때까지 기다리고, `--force`는 1초 뒤 해당 MPS를 강제로 종료한다.

`mps_start 6` 또는 `mps_use 6`을 실행한 **그 터미널에서 이후 실행하는 CUDA 작업**에 GPU·MPS 설정이 전달된다. 새 터미널은 `mps_use 6`으로 선택한다. 프로그램이 환경변수를 덮어쓰거나 자체적으로 GPU를 선택하면 별도 확인이 필요하다. 이미 실행 중인 작업은 자동으로 MPS에 옮겨지지 않는다.

예를 들어 같은 터미널에서 다음처럼 사용한다. `train.py`는 사용자의 실제 학습 프로그램으로 바꾼다.

```bash
mps_start 6
python train.py
# 학습이 끝난 뒤
mps_stop 6
```

여러 GPU를 켠 뒤에는 마지막으로 선택한 GPU가 현재 터미널에 적용된다. 정지·상태 조회는 현재 터미널의 GPU 선택을 바꾸지 않는다. 특정 작업만 명시적으로 지정하고 싶으면 `mps_run`을 사용한다. MPS를 멈춘 뒤에는 다음 작업 전에 다시 시작·선택한다.

## Codex 설치 절차

1. 사용자 셸과 운영체제를 확인한다. 이 코드는 **Linux의 Bash**용이다. 다른 셸의 설정 파일에 Bash 코드를 그대로 넣지 않는다. Bash에서 사용할 수 있게 설치하고 그 한계를 알린다.
2. `nvidia-smi`, `nvidia-cuda-mps-control`, `timeout`, `flock`의 존재를 확인하고 GPU 목록·점유·현재 MPS 프로세스를 읽기 전용으로 확인한다. 사용 가능한 NVIDIA GPU와 MPS 지원 드라이버가 필요하다. 도구가 없거나 관리형 클러스터에서 직접 실행이 제한되면 이유를 보고한다. 임의로 드라이버 설치·sudo·GPU reset·compute mode 변경을 하지 않는다.
3. 기존 `mps_start`, `mps_stop`, `mps_status` alias·함수와 MPS 관련 환경변수를 확인한다. 기존 MPS를 종료하거나 경로를 변경하지 않는다. 이 코드가 기본으로 쓰는 경로는 `/tmp/mps-test-사용자UID/gpu번호/pipe`다. 기존 데몬이 다른 경로에 있으면 자동으로 관리 대상으로 간주하지 않는다.
4. 개인 설치 디렉터리 `~/.local/share/gpu-mps/`를 만든다. 기존 설치 파일과 `~/.bashrc`가 있으면 수정 전 내용을 이 설치 디렉터리 안에 날짜를 붙여 백업한다. 아래 **설치할 전체 코드**를 정확히 `~/.local/share/gpu-mps/shell.sh`로 저장한다. 코드에 이 레포의 절대 경로를 넣지 않는다.
5. 사용자가 위 설치 요청문처럼 개인 셸 자동 로드를 허용했으면, 아래 자동 로드 블록을 `~/.bashrc`의 기존 alias 선언보다 뒤에 한 번만 추가한다. 같은 표시 블록이 이미 있으면 그 블록만 갱신한다. 관련 없는 설정은 보존한다. 기존 MPS 전용 alias는 백업 후 제거해 helper 로드 실패 시 전체 GPU용 옛 명령으로 돌아가지 않게 한다. **사용자가 특정 폴더 안에서만 수정하라고 제한했다면 그 제한을 우선하고, 설정 파일을 건드리지 않은 채 수동 source 한 줄을 안내한다.**
6. `bash -n`으로 helper와 변경한 Bash 설정 파일 문법을 확인한다. 깨끗한 Bash에서 helper를 source하고 5개 공개 함수가 정의되는지 확인한다. alias 충돌, 잘못된 GPU·인자 거부, 명령 인자·종료 코드 전달, GPU별 접속 경로, 정상·강제 종료 명령은 가짜 NVIDIA 명령으로 검증한다. 실제 MPS를 켜거나 끄는 검증은 사용자가 요청한 경우에만 진행한다. 설치만으로 GPU 작업을 시작하지 않는다.
7. 새 터미널의 자동 로드 설정과 기존 설정 보존을 확인하고 결과를 보고한다. Codex의 자식 셸에서 source한 설정은 사용자의 이미 열린 부모 셸에 전달되지 않는다. **현재 터미널에서는 아래 source 한 줄을 사용자가 실행해야 한다.** 새 Bash 터미널에서는 자동으로 사용할 수 있다.

자동 로드 블록:

```bash
# BEGIN GPU MPS SHORTCUTS
if [ -n "${BASH_VERSION:-}" ] && [ -r "$HOME/.local/share/gpu-mps/shell.sh" ]; then
    . "$HOME/.local/share/gpu-mps/shell.sh"
fi
# END GPU MPS SHORTCUTS
```

이미 열린 터미널에서 한 번 적용:

```bash
source "$HOME/.local/share/gpu-mps/shell.sh"
```

## 설치할 전체 코드

아래 코드 블록 전체를 `shell.sh`로 저장한다. 직접 실행하는 스크립트가 아니라 Bash에서 source하는 함수 모음이다. 같은 이름의 alias는 source한 셸 안에서만 해제해 함수가 호출되도록 한다.

```bash
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
```

## 동작 원리와 이식 조건

MPS 자체는 NVIDIA 드라이버가 제공한다. 위 함수는 GPU UUID 조회, GPU마다 다른 접속·로그 경로 지정, NVIDIA 제어 명령 호출을 묶은 것이다. `mps_use`는 데몬 연결을 확인한 뒤 GPU UUID와 MPS 경로를 현재 셸에 export한다. 일반 CUDA용 `CUDA_VISIBLE_DEVICES`와 이 저장소용 `TOKENHSI_GPU`를 같은 UUID로 설정한다.

- 관리 명령에는 Python·PyTorch·conda가 필요 없다. 실제 작업은 해당 서버의 GPU·드라이버와 호환되는 CUDA 환경에서 실행한다.
- 기본 경로는 서버 로컬 `/tmp` 아래이며 사용자 UID와 GPU 번호별로 나뉜다. `MPS_GPU_ROOT`로 바꿀 수 있지만 시작·작업·종료를 실행하는 모든 터미널에서 같은 값을 사용해야 한다.
- 같은 사용자로 MPS와 작업을 실행한다. Slurm·Kubernetes 등에서 GPU를 할당받는 환경은 그 관리 정책과 가시 GPU 설정을 확인한 후 적용한다. 이 예제는 사용자가 물리 GPU 번호를 직접 지정할 수 있는 서버용이다.
- GPU별 MPS를 나눠도 같은 GPU 안의 작업 간 완전한 장애 격리나 드라이버 전체 장애 방지는 보장되지 않는다.
- `mps_stop GPU --force`는 해당 접속 경로에 연결된 모든 작업에 영향을 줄 수 있다. `pkill`로 전체 MPS를 종료하는 기능은 없다.
- `mps_status`에 실제 작업의 PID가 나오는지 확인해야 해당 작업의 MPS 연결을 확인한 것이다. 단순한 데몬 시작 성공과 학습의 MPS 연결 성공은 별도다.

설치 취소 시에는 먼저 사용자가 관리하는 MPS 작업을 정상 정리한 뒤, `.bashrc`의 표시 블록과 설치한 helper만 제거한다. 옛 alias는 백업에만 남긴다. 전체 GPU용 alias를 다시 활성화하지 않는다. 다른 사람이나 다른 접속 경로의 MPS는 종료하지 않는다.

공식 문서: [NVIDIA 명령·환경변수](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html), [MPS 사용 조건·장애 격리](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html).
