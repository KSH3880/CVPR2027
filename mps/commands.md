# GPU별 MPS 시작·종료 명령

짧게 `mps_start 6`·`mps_stop 6`으로 쓰려면 [README의 단축 명령](README.md)을 따른다. 다른 서버에서 Codex에게 설치를 맡길 때는 전체 코드가 포함된 [CODEX_SETUP.md](CODEX_SETUP.md) 하나를 전달한다. 아래는 직접 실행하는 원래 NVIDIA 명령이다.

**GPU 번호만 바꾸면 된다. 별도 `.sh`, 함수, 이 저장소 설치는 필요 없다.**

아래는 Bash 기준이며 GPU 4·6 테스트에서 사용한 `/tmp/mps-test-사용자UID/gpu번호/pipe` 경로를 그대로 사용한다. `mps-test`는 디렉터리 이름일 뿐이며 일반 작업에도 사용할 수 있다.

## 시작

`MPS_GPU=6`을 `MPS_GPU=4`로 바꾸면 GPU 4용 명령이다. 이미 해당 경로에서 실행 중이면 다시 시작하지 않는다.

```bash
MPS_GPU=6
nvidia-smi -i "$MPS_GPU"

(
    MPS_UUID=$(nvidia-smi -i "$MPS_GPU" --query-gpu=uuid --format=csv,noheader) || exit
    export CUDA_VISIBLE_DEVICES="$MPS_UUID"
    export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu${MPS_GPU}/pipe"
    export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-test-${UID}/gpu${MPS_GPU}/log"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" || exit
    nvidia-cuda-mps-control -d
)
```

괄호 안 설정은 시작 명령에만 적용되므로 현재 터미널의 다른 작업에 GPU 설정을 남기지 않는다. 실제 GPU 점유를 확인한 뒤 사용한다.

## 정상 종료

지정한 접속 경로의 MPS만 종료한다. 연결된 작업이 있으면 끝날 때까지 기다린다. 새 터미널에서도 아래 블록을 그대로 실행할 수 있다.

```bash
MPS_GPU=6
echo quit | CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu${MPS_GPU}/pipe" \
    nvidia-cuda-mps-control
```

## 강제 종료

GPU 4를 끊었던 테스트처럼, 작업이 돌고 있어도 1초 뒤 해당 MPS를 강제로 종료한다. **그 MPS에 연결된 작업은 CUDA 오류로 중단될 수 있다.**

```bash
MPS_GPU=4
echo "quit -t 1" | CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu${MPS_GPU}/pipe" \
    nvidia-cuda-mps-control
```

## 연결된 작업 확인

```bash
MPS_GPU=6
echo ps | CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu${MPS_GPU}/pipe" \
    nvidia-cuda-mps-control
```

## 작업 실행 시 필요한 설정

GPU별 접속 경로를 따로 만들었으므로 **작업에도 같은 접속 경로를 전달한다.** MPS를 켜는 것만으로 임의의 Python 작업이 이 경로에 자동 연결되는 것은 아니다. 이미 실행 중인 작업은 자동으로 MPS에 옮겨지지 않는다.

일반 CUDA 프로그램의 형태는 다음과 같다. `your_program.py`는 실행할 실제 프로그램으로 바꾼다.

```bash
MPS_GPU=6
CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$MPS_GPU" --query-gpu=uuid --format=csv,noheader)" \
CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-test-${UID}/gpu${MPS_GPU}/pipe" \
    python your_program.py
```

이 저장소의 학습 스크립트는 `TOKENHSI_GPU`로 CUDA GPU를 설정하므로 학습 시에는 [README의 학습 예제](README.md)를 따른다. 복사해 실행할 수 있는 연산 테스트 명령은 [GPU 4·6 테스트](gpu4_gpu6_test.md)에 있다.

## 무엇을 만든 건가

- **MPS 자체:** NVIDIA 드라이버가 제공하는 기능이다. 새로 구현한 것이 아니다.
- **GPU별 분리:** 각 데몬에 GPU 한 개만 보이게 하고, 접속 경로를 서로 다르게 지정했다.
- **`heartbeat.py`:** 매초 CUDA 연산이 완료됐는지 표시하는 테스트 프로그램이다. MPS를 켜는 데 필요하지 않다.
- **예전 `mps_start`, `mps_stop`:** 이 서버의 셸 단축 명령이다. 다른 서버에 자동으로 존재하지 않는다. 여기서는 NVIDIA 명령을 직접 사용한다.

## 다른 서버에서도 쓸 수 있나

**MPS를 지원하는 NVIDIA GPU·드라이버가 설치된 Linux 서버라면 같은 방식을 사용할 수 있다.** Bash에서 먼저 다음 명령이 동작하는지 확인한다.

```bash
nvidia-smi
command -v nvidia-cuda-mps-control
```

- 서버에 존재하는 GPU 번호를 선택한다. 위 명령은 UUID를 해당 서버에서 조회하므로 UUID를 복사할 필요 없다.
- 자신의 MPS 데몬과 작업은 같은 사용자로 실행한다. GPU·MPS를 스케줄러나 관리자가 관리하는 서버는 그 운영 방식에 맞춰 사용한다.
- MPS 시작·종료에는 conda나 PyTorch가 필요 없다. `heartbeat.py` 테스트에는 해당 GPU에서 동작하는 CUDA 지원 PyTorch 환경이 필요하다.
- 이 저장소 경로·`tokenhsi` conda 환경은 서버마다 다를 수 있다. 테스트 실행 명령의 환경 준비 부분은 대상 서버에 맞춘다.

GPU별 MPS 분리는 같은 GPU 안의 작업 간 완전한 장애 격리나 서버·드라이버 전체 장애 방지를 보장하지 않는다.

공식 설명: [NVIDIA 명령·환경변수](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html), [사용 조건·장애 격리](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html).
