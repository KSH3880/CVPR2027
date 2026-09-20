# GPU 4 MPS 종료 후 GPU 6 생존 확인

## 예전에는 왜 `mps_start` 한 줄이었나

MPS 자체의 시작 명령은 `nvidia-cuda-mps-control -d` 한 줄이다. 이 서버의 `~/.bashrc`를 확인해 보니 `mps_start`는 NVIDIA 기본 명령이 아니라, 디렉터리 준비·데몬 확인·시작 명령을 묶어둔 **alias(단축 명령)**였다.

GPU별로 독립 실행하려면 그 한 줄 앞에 다음 설정을 붙인다.

| 설정 | 역할 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | 해당 MPS가 사용할 GPU 한 개 선택 |
| `CUDA_MPS_PIPE_DIRECTORY` | GPU 4와 GPU 6의 접속 경로 분리 |
| `CUDA_MPS_LOG_DIRECTORY` | GPU별 로그 저장 위치 지정 |

아래 명령이 긴 이유는 이 설정을 생략하지 않고 보여주기 때문이다. **별도 관리 스크립트는 필요 없다.** 연산 테스트용 Python만 사용한다.

현재 `~/.bashrc`의 MPS 경로 export 두 줄은 주석 처리돼 있다. `mps_stop`은 `/home/hwanhee/Autonomous_Driving_26_cost/stop_mps.sh`를 가리키지만, 확인 시점에 해당 파일이 없었다. 따라서 아래 테스트는 기존 단축 명령 대신 NVIDIA 명령을 직접 사용한다. 셸 설정 파일은 수정하지 않았다.

## 복사해서 실행하는 순서

**터미널 A = GPU 4 연산, 터미널 B = GPU 6 연산, 터미널 C = MPS 제어.**

`heartbeat.py`는 매초 실제 CUDA 덧셈·동기화를 완료한 뒤 `CUDA_OK`를 출력한다. 직접 종료할 때까지 반복한다. MPS 관리용 `.sh`나 함수는 사용하지 않는다.

## 1. 터미널 C: GPU 4·6의 MPS 시작

`nvidia-smi`에서 두 GPU의 점유를 먼저 확인한다. 아래는 이번 테스트만 사용하는 접속 경로다. 같은 테스트 MPS를 이미 시작했다면 시작 명령을 반복하지 않는다.

```bash
nvidia-smi -i 4,6

for MPS_GPU in 4 6; do
    MPS_UUID=$(nvidia-smi -i "$MPS_GPU" --query-gpu=uuid --format=csv,noheader) || break
    mkdir -p "/tmp/mps-test-${UID}/gpu${MPS_GPU}/pipe" "/tmp/mps-test-${UID}/gpu${MPS_GPU}/log"
    CUDA_VISIBLE_DEVICES="$MPS_UUID" \
    CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu${MPS_GPU}/pipe" \
    CUDA_MPS_LOG_DIRECTORY="/tmp/mps-test-${UID}/gpu${MPS_GPU}/log" \
        nvidia-cuda-mps-control -d || break
done
```

## 2. 터미널 A: GPU 4 연산 시작

```bash
cd /home/hwanhee/ksh/approach_distance_success
TOKENHSI_CONDA_ENV=tokenhsi source tokenhsi/scripts/multi_agent/runtime_env.sh

CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 4 --query-gpu=uuid --format=csv,noheader)" \
CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu4/pipe" \
    python -u mps/heartbeat.py --label GPU4
```

## 3. 터미널 B: GPU 6 연산 시작

```bash
cd /home/hwanhee/ksh/approach_distance_success
TOKENHSI_CONDA_ENV=tokenhsi source tokenhsi/scripts/multi_agent/runtime_env.sh

CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 6 --query-gpu=uuid --format=csv,noheader)" \
CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu6/pipe" \
    python -u mps/heartbeat.py --label GPU6
```

양쪽 화면에서 `step=1 CUDA_OK`, `step=2 CUDA_OK`처럼 숫자가 계속 증가해야 한다. 가벼운 연산이므로 `nvidia-smi`의 GPU 사용률은 낮아도 정상이다.

## 4. 터미널 C: 양쪽 MPS 연결 확인

```bash
echo ps | CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu4/pipe" nvidia-cuda-mps-control
echo ps | CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu6/pipe" nvidia-cuda-mps-control
```

**각각의 목록에 테스트 Python PID가 있어야 다음으로 진행한다.** 해당 PID는 A·B 화면의 `pid=`와 일치해야 한다. 목록이 비어 있거나 MPS 접속 오류가 나면 연결을 확인한 것이 아니다.

## 5. 터미널 C: 직접 GPU 4 MPS만 끊기

테스트 GPU 4 MPS를 강제로 종료한다. 일반 `quit`은 작업이 끝나기를 기다리므로 여기서는 `quit -t 1`을 사용한다.

```bash
echo "quit -t 1" | CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu4/pipe" \
    nvidia-cuda-mps-control
```

| 화면 | 기대 결과 |
| --- | --- |
| 터미널 A / GPU 4 | MPS 통신·CUDA 오류가 나며 연산 중단 |
| 터미널 B / GPU 6 | 시간과 `step`이 계속 증가하며 `CUDA_OK` 출력 |

GPU 6 화면을 5~10초 더 지켜보고, 연결도 다시 확인한다.

```bash
echo ps | CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu6/pipe" nvidia-cuda-mps-control
```

터미널 A에서 `Ctrl+C`를 누르는 것은 테스트 프로그램의 정상 종료다. **MPS 자체가 끊어져도 다른 GPU가 살아 있는지 확인하려면 위 제어 명령을 사용한다.** GPU 하드웨어나 서버 전체를 끄는 테스트는 아니다.

## 6. 정리

터미널 B에서 `Ctrl+C`를 누르면 CUDA 작업을 마친 뒤 `STOP GPU6`이 출력된다. 그다음 터미널 C에서 남은 GPU 6 MPS를 종료한다.

```bash
echo quit | CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu6/pipe" nvidia-cuda-mps-control
```

5번을 건너뛰었다면 터미널 A도 `Ctrl+C`로 종료한 후 GPU 4 MPS를 정리한다.

```bash
echo quit | CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu4/pipe" nvidia-cuda-mps-control
```

MPS 로그는 `/tmp/mps-test-${UID}/gpu4/log/`, `/tmp/mps-test-${UID}/gpu6/log/`에 남는다.

## 검증 범위

이 테스트는 GPU별 MPS의 종료 범위를 확인한다. 같은 GPU 안의 작업 간 장애 격리나 드라이버 전체 장애까지 보장하지 않는다. 예제 Python·Bash 문법만 확인했으며, 이 GPU 4·6 테스트의 실행과 GPU 4 종료는 사용자가 수행한다.
