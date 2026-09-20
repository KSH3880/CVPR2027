# GPU 하나에만 MPS 사용하기

## 짧은 명령으로 사용하기

현재 서버는 사용자 요청으로 `~/.local/share/gpu-mps/shell.sh`를 설치하고 `~/.bashrc`에 자동 로드를 등록했다. **새 Bash 터미널에서는 바로 `mps_start 6`을 실행한다.** 전체 GPU로 실행될 수 있던 옛 MPS alias는 백업 후 제거했다.

설정 전에 이미 열려 있던 터미널에서만 한 번 적용한다:

```bash
source "$HOME/.local/share/gpu-mps/shell.sh"
```

그다음 `mps_start 6`, `mps_stop 6`으로 시작·종료한다. `mps_start`는 학습 없이도 지정 GPU의 MPS 서버까지 바로 띄우며 현재 터미널의 GPU·MPS 경로도 선택한다. `mps_start 4`, `mps_stop 4 --force`도 사용할 수 있다. 자동 로드 자체는 MPS를 시작하거나 GPU를 선택하지 않는다.

원래 `.bashrc` 백업은 `~/.local/share/gpu-mps/bashrc.backup.*`에 있다. 개인 설치본은 이 저장소 `shell.sh`의 복사본이며, helper 코드를 수정하면 설치본도 갱신해야 한다.

공유 서버 보호: GPU 번호 하나를 UUID로 변환하고 UID/GPU별 전용 경로만 사용한다. 기존 데몬의 PID 소유자·GPU UUID·접속 및 로그 경로를 확인한 뒤에만 재사용·조회·종료한다. 불일치, 오래된 PID, 확인할 수 없는 socket, 다른 사용자 경로 또는 symlink는 거부한다. 정상 종료 후 알려진 소켓 파일만 남은 경우에는 PID 없음·데몬 응답 없음·커널의 살아 있는 소켓 없음·파일 소유자를 확인하고 `pipe.stale.*`에 보관한 뒤 새 pipe를 만든다. 동시 시작은 GPU별 lock으로 직렬화한다. 기본 `/tmp/nvidia-mps`로 돌아가는 fallback은 없다.

**이 레포의 train/test/VNC는 `mps_run` 없이도 지정 GPU의 켜진 MPS에 자동 연결한다.** 새 터미널에서도 `TOKENHSI_GPU=6`으로 실행하면 해당 GPU의 내 MPS만 검증해서 연결한다. MPS가 꺼져 있으면 새 데몬을 만들지 않으며, 기존 접속 설정과 GPU가 충돌하거나 검증에 실패하면 실행을 중단한다. 이미 실행 중인 학습은 자동 전환되지 않는다.

```bash
mps_start 6
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_success_train.sh
```

다른 프로젝트의 일반 프로그램은 그 프로그램의 실행 환경에 MPS 경로가 전달돼야 한다. 같은 터미널에서 `mps_start`/`mps_use` 후 실행하거나 `mps_run 6 <command>`를 쓴다. 이 레포의 자동 연결은 `runtime_env.sh` → `mps_auto_env.sh`에서 수행한다.

실제 연결은 `mps_status 6`의 client PID로 확인한다. `mps_stop 6`은 GPU 6 전용 MPS만 정상 종료하며, 연결된 작업이 있으면 종료를 기다린다. GPU 자체나 MPS를 사용하지 않는 학습을 끄는 명령이 아니다. `--force`는 연결된 작업을 중단시킬 수 있으므로 정상 종료와 구분한다.

보호 장치 검증은 `python mps/test_shell.py`로 실행한다. 가짜 NVIDIA 도구와 CPU 프로세스만 사용하며 실제 MPS/GPU에 접근하지 않는다.

**다른 서버에는 [CODEX_SETUP.md](CODEX_SETUP.md) 하나를 전달한다.** 필요한 전체 코드와 Codex 설치 요청문이 들어 있다. 개인 `.bashrc` 자동 로드를 허용해 설치하면 그 서버의 새 Bash 터미널에서는 source 없이 단축 명령을 바로 쓸 수 있다.

`mps_start`·`mps_stop`은 기본 NVIDIA 명령이 아니라 [shell.sh](shell.sh)에 정의한 단축 함수다. 아래에는 단축 함수를 사용하지 않는 직접 명령도 남겨둔다.

**시작·정상 종료·강제 종료 명령만 보려면 [GPU별 MPS 명령 모음](commands.md)을 사용한다.** GPU 번호만 바꾸는 형태이며 다른 서버에서의 사용 조건도 정리했다.

GPU 4·6에서 계속 연산을 돌리고 GPU 4만 끊어보려면 [GPU 4·6 분리 테스트](gpu4_gpu6_test.md)를 따른다. 테스트 프로그램은 [heartbeat.py](heartbeat.py)다.

**MPS는 NVIDIA의 `nvidia-cuda-mps-control -d` 명령으로 켠다. 별도 `.sh`나 `mps_gpu` 함수는 필요 없다.**

GPU별로 따로 운영하려면 시작할 때 사용할 GPU와 접속 경로를 지정하고, 작업에도 같은 접속 경로를 전달한다. 아래 예제는 GPU 6에만 MPS를 켠다. 다른 GPU는 MPS 없이 사용하거나 비워둘 수 있다.

## 1. GPU 6 MPS 시작

서버의 Bash 터미널에서 실행한다. GPU 6에 기존 작업이 있으면 점유를 먼저 확인한다.

```bash
nvidia-smi -i 6

MPS_UUID=$(nvidia-smi -i 6 --query-gpu=uuid --format=csv,noheader)
mkdir -p "/tmp/mps-test-${UID}/gpu6/pipe" "/tmp/mps-test-${UID}/gpu6/log"

CUDA_VISIBLE_DEVICES="$MPS_UUID" \
CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu6/pipe" \
CUDA_MPS_LOG_DIRECTORY="/tmp/mps-test-${UID}/gpu6/log" \
nvidia-cuda-mps-control -d
```

이 명령은 데몬을 백그라운드로 시작한다. 이미 같은 경로에 MPS를 켰다면 시작 명령을 반복할 필요 없다. `(base)` 환경에서도 MPS 자체를 켤 수 있다.

## 2. 학습을 해당 MPS에 연결

이 저장소에서는 기존 학습 스크립트 앞에 **GPU UUID와 MPS 접속 경로**를 붙이면 된다. 스크립트 파일은 수정하지 않는다.

아래는 **2048환경·2회 학습 확인용**이다. 결과는 `mps/output/` 아래에 저장한다.

```bash
cd /home/hwanhee/ksh/approach_distance_success
nvidia-smi -i 6
MPS_UUID=$(nvidia-smi -i 6 --query-gpu=uuid --format=csv,noheader)

TOKENHSI_GPU="$MPS_UUID" \
CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu6/pipe" \
TOKENHSI_CONDA_ENV=tokenhsi \
RESUME_CHECKPOINT= \
MAX_ITERATIONS=2 \
OUTPUT_PATH="mps/output/gpu6_$(date +%Y%m%d_%H%M%S)_check" \
bash tokenhsi/scripts/multi_agent/approach_distance_success_train.sh 2 2048 3
```

같은 GPU에서 여러 학습을 공유하려면 각 터미널에서 동일한 GPU UUID와 접속 경로를 사용한다. 작업마다 출력 폴더는 다르게 지정한다. SSH 연결 종료에도 학습을 유지하려면 학습을 `tmux` 안에서 실행한다.

## 3. MPS 연결 확인

학습이 실행 중일 때 다른 터미널에서 확인한다. 아래 명령은 새 터미널에서도 그대로 사용할 수 있다.

```bash
echo ps | CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu6/pipe" \
    nvidia-cuda-mps-control
```

학습 Python이 클라이언트 목록에 나오면 MPS 연결을 확인한 것이다. 데몬만 켜고 아직 작업을 실행하지 않았다면 목록이 비어 있을 수 있다.

## 4. GPU 6 MPS 종료

GPU 6의 작업들이 끝난 뒤 실행한다.

```bash
echo quit | CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu6/pipe" \
    nvidia-cuda-mps-control
```

일반 `quit`은 연결된 작업이 끝날 때까지 기다린다. 이 접속 경로에 연결된 MPS만 종료한다.

## 다른 GPU에도 MPS를 켜려면

GPU 1에도 켜고 싶다면 위 명령의 GPU 선택 `-i 6`을 `-i 1`로, 경로의 `gpu6`을 `gpu1`로 바꿔 별도로 실행한다. GPU마다 **보이는 GPU 한 개 + 서로 다른 접속 경로**를 사용한다.

| GPU | 접속 경로 |
| --- | --- |
| 1 | `/tmp/mps-test-${UID}/gpu1/pipe` |
| 6 | `/tmp/mps-test-${UID}/gpu6/pipe` |

기존 기본 경로 `/tmp/nvidia-mps` 또는 이전 테스트의 `/tmp/mps-사용자UID.임시문자/`에서 시작한 MPS는 위 경로와 별개다. **시작·작업·확인·종료에 같은 경로를 사용한다.**

## 앞서 수행한 분리 테스트 결과

2026-09-19 사용자 실행 로그를 확인했다.

- GPU 0 MPS 강제 종료 후 GPU 0 테스트 연산이 중단됐다.
- GPU 1의 테스트 연산은 계속됐고 `DONE`으로 정상 종료했다.
- GPU 1에서 실제 학습이 MPS에 연결됐고 checkpoint를 저장했다.

이는 GPU별 MPS 종료 범위가 분리되는지 확인한 결과다. 같은 GPU를 공유하는 작업의 치명적인 CUDA 오류나 드라이버 전체 장애까지 격리한다는 뜻은 아니다. 이 문서의 새 GPU 6 명령은 문법만 확인했으며 다시 실행하지 않았다.

## 참고

- [NVIDIA MPS 명령·환경변수](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html)
- [NVIDIA MPS 장애 격리 범위](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)
- 이 폴더는 MPS 안내 전용이다. 기존 학습 코드·설정·문서는 수정하지 않는다.
