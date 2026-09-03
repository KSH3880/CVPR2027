# TokenHSI 머신 포팅 가이드

다른 서버에서 복사한 TokenHSI repo를 현재 머신에 맞출 때 LLM 에이전트에게 이
파일과 repo 경로를 제공한다. 네트워크, 모델, 학습·테스트 파이프라인은 변경하지
말고 경로·실행 환경·시스템 의존성만 최소 수정한다.

## 실행 환경 규약

스크립트에는 절대 경로나 특정 머신의 환경 이름이 들어가지 않습니다. repo root는
스크립트 자기 위치에서 계산하므로 checkout이 어디에 있든 동작하고, 나머지는 아래
환경 변수로 조정합니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `TOKENHSI_CONDA_ENV` | 이미 활성화된 env, 없으면 `tokenhsi` | 사용할 conda 환경 이름 |
| `CONDA_BASE` | 자동 탐색 (`conda info --base`, `~/anaconda3`, `~/miniconda3`, `/opt/conda` …) | conda 설치 prefix |
| `TOKENHSI_GPU` | `0` | `CUDA_VISIBLE_DEVICES` 값 (`CUDA_DEVICE_ORDER=PCI_BUS_ID` 고정) |
| `X11VNC` / `VNC_DIR` | `PATH`의 `x11vnc`, 없으면 `$VNC_DIR/usr/bin/x11vnc` | x11vnc 위치 |
| `NOVNC_DIR` | `/usr/share/novnc`, `~/opt/novnc` 등 자동 탐색 | `vnc.html`이 있는 디렉터리 |
| `WEBSOCKIFY` | `PATH` → conda 환경 → noVNC 번들 순 탐색 | websockify 실행 파일 |
| `PORT` | `6080` | noVNC 웹 포트 |

Isaac Gym은 활성화된 conda 환경에서 import되는 설치본을 사용합니다.

## 포팅 절차

1. 기존 작업을 보존하도록 `git status --short`를 먼저 확인한다.
2. 실행 스크립트와 직접 참조하는 YAML만 읽고, 절대 경로·conda 환경명·GPU
   설정·체크포인트 경로를 찾는다.
3. `tokenhsi/scripts/multi_agent/runtime_env.sh`에서 repo root를 스크립트 위치로부터
   계산하고 conda 환경을 활성화한다. 환경 이름은 `TOKENHSI_CONDA_ENV`로,
   GPU는 `TOKENHSI_GPU`로 지정한다 — 스크립트에 이름을 하드코딩하지 않는다.
4. `ldconfig -p`에 `libcuda.so.1`만 있고 `libcuda.so`가 없으면 임시 디렉터리에
   `libcuda.so` 심볼릭 링크를 만들고 `LD_LIBRARY_PATH`에 추가한다.
5. motion YAML의 `file`, `obj_file`을 YAML 파일 위치 기준으로 해석해 누락 파일을
   검사한다. asset YAML의 `assetRoot`, `assetFileName`도 실제 파일과 대조한다.
6. `run-gui.sh`의 Xvfb, x11vnc, noVNC, websockify 경로와 포트 점유 여부를
   확인한다. 영구 서비스가 실제로 설치되어 있지 않으면 systemd 서비스를
   가정하지 말고 임시 Xvfb 세션을 사용한다.
7. 아래 스모크 테스트를 순서대로 실행한다. 오류가 나면 처음 발생한 시스템·경로
   의존성만 수정하고 전체를 다시 검증한다.

## 스모크 테스트

```bash
# 환경·GPU: Isaac Gym을 torch보다 먼저 import한다.
# ${TOKENHSI_CONDA_ENV}는 본인 환경 이름으로. 이미 activate 했다면 conda run 없이 실행해도 된다.
CUDA_VISIBLE_DEVICES=${TOKENHSI_GPU:-0} conda run -n "${TOKENHSI_CONDA_ENV:-tokenhsi}" python -c \
  "import isaacgym, torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"

# 학습 1 epoch: 스모크 전용 환경변수이며 기본 학습 설정은 바꾸지 않는다.
MAX_ITERATIONS=1 sh tokenhsi/scripts/multi_agent/ma_carry_train.sh 1 256 0

# headless 테스트
sh tokenhsi/scripts/multi_agent/ma_carry_test.sh \
  output/ma_carry/<run>/nn/HumanoidMA.pth 1 1 0

# GUI + noVNC
sh tokenhsi/scripts/multi_agent/run-gui.sh \
  sh tokenhsi/scripts/multi_agent/ma_carry_test.sh \
  output/ma_carry/<run>/nn/HumanoidMA.pth 1 1 0

# GUI 학습 1 epoch
MAX_ITERATIONS=1 sh tokenhsi/scripts/multi_agent/run-gui.sh \
  sh tokenhsi/scripts/multi_agent/ma_carry_watch.sh 1 4 0
```

학습은 최소 1 epoch의 rollout과 update가 완료되어야 한다. 테스트는 체크포인트 로드 후
시뮬레이션 step이 진행되어야 한다. GUI는 noVNC HTTP 응답만으로 판정하지 말고
X display에 Isaac Gym viewer 창이 실제로 생성되는지도 확인한다.

## 변경 금지 범위

- 모델 아키텍처, 레이어 크기, observation/action 구성
- reward, reset, physics, dataset semantics
- optimizer, loss, rollout 로직
- 학습·테스트 파이프라인

이 범위의 변경이 필요해 보이면 포팅 작업을 멈추고 사용자에게 먼저 확인한다.
