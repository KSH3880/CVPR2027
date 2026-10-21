# 부채꼴 센서 학습·테스트 실행 명령

두 실험이 있다. 현재 실험은 바로 아래 [두 무대·연결다리 실험](#두-무대연결다리-실험-1024개가림-없음)이다. 1–8절은 기존 단일 무대 **576개·100도·10m** 실험이다. 모두 에이전트 2명·상자 3개이며, 준비 명령을 실행한 뒤 원하는 실행 블록 하나를 선택한다.

## 두 무대·연결다리 실험 (1024개·가림 없음)

reset마다 50%는 기존 단일 무대 고립 장면이다. 나머지 50%는 두 무대 사이의 간격(env별 1.5–2.5m)을 외력으로 움직이는 다리 상자가 채우는 장면이며, 과제 상자는 건너편 무대 위에 놓인다. 센서는 머리 기준 전방 100도·10m·**1024개**(32고리)이며 가림 판정 없이 수직 조회로 높이를 읽는다. 전체 관측은 2655차원이고 Stage 1에서 새로 전이한다. 장면·시간·RSI 설정은 [실행 가이드](config.md#stage-2-두-무대연결다리)에 있다.

에이전트 간 `collision`은 root 상대 XY 위치·속도로 예측한 CPA 위험도다(계수 0.5, 안전거리 0.7m, 시간 할인율 0.99). 가까워도 서로 멀어지면 0이며, 바닥 추락은 별도 실패 종료다. 출력 경로는 이전 거리 보상 실행과 구분해 `_cpa`를 사용한다.

| 목적 | 실행 파일 | 환경 수 |
| --- | --- | ---: |
| 본학습, 화면 없음 | `approach_stage2_fan_bridge_train.sh` | 2048 |
| 학습하면서 noVNC로 보기 | `approach_stage2_fan_bridge_watch_vnc.sh` | 1 또는 4 |
| 저장된 정책을 noVNC로 테스트 | `approach_stage2_fan_bridge_vnc.sh` | 1 |
| 로컬 학습 / 테스트 시각화 | `approach_stage2_fan_bridge_watch.sh` / `approach_stage2_fan_bridge_test.sh` | 1 또는 4 / 1 |

### 준비

```bash
cd /home/user/jhh/Projects/TokenHSI_total_sensor
nvidia-smi
GPU=0

unset MAX_ITERATIONS RESUME_CHECKPOINT OUTPUT_PATH
unset SENSOR_MASK EPISODE_LENGTH BRIDGE_PROBABILITY

STAGE1='output/stage1/28-17-01-48/nn/stage1.pth'
```

`STAGE1`은 이 저장소의 Stage 1 distillation epoch 14500 checkpoint(schema 9)다. GPU 점유를 확인하고 필요하면 `GPU` 값을 바꾼다.

### 2048환경 본학습

```bash
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_bridge_train.sh 2 2048 3
```

- 출력: `output/approach_stage2_fan_bridge_cpa/<run>/`. Checkpoint는 `<run>/nn/*.pth`, TensorBoard는 `<run>/summaries/`.
- 다리 지표는 TensorBoard `cliff/bridge/*`에 있다: `falls_before_arrival`, `falls_after_arrival`, `waiting_near_gap_steps`(도착 전 간격 1m 이내 정지), `crossed_agents`/`episodes`(건넌 에이전트 수/다리 에피소드 수).

짧은 실행 확인만 할 때는 환경 수 2048을 유지하고 반복 횟수만 줄인다.

```bash
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" MAX_ITERATIONS=1 \
OUTPUT_PATH=output/approach_stage2_fan_bridge_cpa_check/train \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_bridge_train.sh 2 2048 3
```

### 적은 환경으로 학습하며 보기

실제 PPO/AMP 학습과 센서 표시를 함께 실행한다. 기본 4환경, 최소 1환경이다.

```bash
# 서버 noVNC
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_bridge_watch_vnc.sh 2 4 3

# 로컬 viewer
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_bridge_watch.sh 2 1 3
```

- 출력: `output/approach_stage2_fan_bridge_cpa_watch/<run>/`. 소수 환경용 PPO 배치를 사용하므로 본학습과 학습 조건이 다르다.
- 명령 앞에 `BRIDGE_PROBABILITY=1`을 붙이면 다리 장면만, `0`을 붙이면 단일 무대만 나온다. 이렇게 하면 학습 분포가 5:5와 달라지므로 화면 확인용으로만 쓴다.
- noVNC 접속은 3절과 같다(GPU 0에서 5802 포워딩 후 <http://localhost:5802/>).

### checkpoint 선택

```bash
rg --files --no-ignore -g '*.pth' output/approach_stage2_fan_bridge_cpa output/approach_stage2_fan_bridge_cpa_watch
CKPT='output/approach_stage2_fan_bridge_cpa/<run>/nn/ApproachStage2FanBridge.pth'
```

`<run>`과 파일명은 실제 경로로 바꾼다. **다리 실험 checkpoint만 사용할 수 있다.** 576개 부채꼴, 정사각형 센서, Stage 1 checkpoint는 sensor metadata가 달라 테스트·resume가 거부된다.

기존 거리 패널티로 학습한 다리 checkpoint도 로드할 수 있다. 새 프로세스의 보상은 CPA이며, 이전 checkpoint가 CPA 회피를 학습했다는 뜻은 아니다. 실행 중인 프로세스는 기존 보상으로 계속 동작한다.

### 저장된 정책의 테스트 시각화

```bash
# 서버 noVNC: 다리 장면만
TOKENHSI_GPU="$GPU" HEADLESS=0 SENSOR_VIS=1 BRIDGE_PROBABILITY=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_bridge_vnc.sh "$CKPT" 2 1 3 10

# 로컬 viewer: 다리 장면만
TOKENHSI_GPU="$GPU" HEADLESS=0 SENSOR_VIS=1 BRIDGE_PROBABILITY=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_bridge_test.sh "$CKPT" 2 1 3 10
```

- `2 1 3 10`은 에이전트 2명, 환경 1개, 상자 3개, 평가 반복 10회다.
- `BRIDGE_PROBABILITY`: `1`은 다리 장면만, `0`은 단일 무대만 보여준다. 생략하면 학습과 같이 5:5로 섞는다. checkpoint 호환에는 영향이 없다.
- 센서 표시: 하늘색 십자는 수직 조회한 표면 높이다. 가림 판정이 없으므로 노랑·자홍 표시는 나오지 않는다. 다리는 갈색, 무대는 회색이다.
- 센서 특징을 0으로 만든 비교 평가는 명령 앞에 `SENSOR_MASK=1`을 붙인다.

### 화면 없는 짧은 테스트

```bash
TOKENHSI_GPU="$GPU" HEADLESS=1 EPISODE_LENGTH=64 \
OUTPUT_PATH=output/approach_stage2_fan_bridge_cpa_check/eval \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_bridge_test.sh "$CKPT" 2 16 3 1
```

16환경·최대 64 step(약 2초)의 실행 경로 확인이다. 이 시간 안에는 일부 다리만 도착하므로, 건너기를 평가하려면 `EPISODE_LENGTH`를 지정하지 않는다(600 step, 20초).

### 학습 재개

```bash
# 2048환경 본학습 재개
TOKENHSI_GPU="$GPU" RESUME_CHECKPOINT="$CKPT" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_bridge_train.sh 2 2048 3

# 소수 환경에서 재개하며 noVNC 시각화
TOKENHSI_GPU="$GPU" RESUME_CHECKPOINT="$CKPT" SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_bridge_watch_vnc.sh 2 1 3
```

`RESUME_CHECKPOINT`가 설정되어 있으면 `STAGE1_CHECKPOINT`보다 우선한다. 새 전이를 시작할 때는 준비의 `unset`을 실행한다.

---

아래 1–8절은 기존 단일 무대 576개·100도·10m 실험이다.

| 목적 | 실행 파일 | 환경 수 |
| --- | --- | ---: |
| 본학습, 화면 없음 | `approach_stage2_fan_sensor_train.sh` | 2048 |
| 학습하면서 noVNC로 보기 | `approach_stage2_fan_sensor_watch_vnc.sh` | 1 또는 4 |
| 저장된 정책을 noVNC로 테스트 | `approach_stage2_fan_sensor_vnc.sh` | 1 |
| 로컬 학습 / 테스트 시각화 | `watch.sh` / `test.sh` | 1 또는 4 / 1 |

`watch`는 가중치를 갱신하는 실제 학습이다. `test`·`vnc`는 저장된 checkpoint로 추론하며 가중치를 갱신하지 않는다. 실행 중인 본학습 화면을 watch가 연결해서 보여주는 것은 아니며, 각각 별도 프로세스다.

## 1. 공통 준비

```bash
cd /home/user/jhh/Projects/TokenHSI_vision_sensor
nvidia-smi
GPU=0

unset MAX_ITERATIONS RESUME_CHECKPOINT OUTPUT_PATH
unset SENSOR_MASK EPISODE_LENGTH

STAGE1='output/approach_scenario_stage1_self_sum_distill/ApproachScenarioStage1SelfSumDistill_28-17-01-48/nn/ApproachScenarioStage1SelfSumDistill.pth'
```

GPU 점유를 확인하고 필요하면 `GPU` 값을 바꾼다. 스크립트는 기본 `tokenhsi_sm120` conda 환경과 `runtime_env.sh`를 사용한다. 새 터미널에서는 준비 명령을 다시 실행한다.

## 2. 2048환경 본학습

Stage 1 checkpoint에서 현재 센서 구성으로 새 전이 학습을 시작한다.

```bash
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_train.sh 2 2048 3
```

- 인자 `2 2048 3`: 에이전트 수, 환경 수, 상자 수.
- 출력: `output/approach_stage2_fan_sensor_576_100_10m/<run>/`.
- Checkpoint: `<run>/nn/*.pth`, TensorBoard: `<run>/summaries/`.
- 기본 최대 1,000,000 epoch, checkpoint 저장 주기 500 epoch.

짧은 실행 확인이 필요할 때만 다음 명령을 사용한다. 환경 수는 2048을 유지한다.

```bash
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" MAX_ITERATIONS=1 \
OUTPUT_PATH=output/approach_stage2_fan_sensor_576_10m_check/train \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_train.sh 2 2048 3
```

## 3. 학습하면서 noVNC로 보기

4환경에서 새 전이 학습과 센서 시각화를 함께 실행한다.

```bash
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_watch_vnc.sh 2 4 3
```

1환경만 보려면 마지막 인자를 `2 1 3`으로 바꾼다. 출력은 `output/approach_stage2_fan_sensor_576_100_10m_watch/<run>/`이다. 소수 환경용 PPO 배치를 사용하므로 2048환경 본학습과 학습 조건이 다르다. PPO 업데이트 중에는 화면이 잠깐 멈출 수 있다.

이 서버의 GPU 0에서 기존 GUI 서비스를 이용하면 VS Code **PORTS에서 5802를 포워딩**하고 <http://localhost:5802/>를 연다. 다른 GPU나 별도 GUI 서비스를 사용하면 터미널에 출력된 noVNC 주소·포트에 따른다.

## 4. 테스트하거나 이어 학습할 checkpoint 선택

먼저 현재 10m 실험에서 저장된 파일을 확인한다.

```bash
rg --files --no-ignore -g '*.pth' output/approach_stage2_fan_sensor_576_100_10m
# watch 결과를 사용할 경우
rg --files --no-ignore -g '*.pth' output/approach_stage2_fan_sensor_576_100_10m_watch
```

아래 예시의 `<run>`과 파일명은 실제 출력된 경로로 바꾼다. Stage 1 checkpoint를 테스트용 `CKPT`에 넣지 않는다.

```bash
CKPT='output/approach_stage2_fan_sensor_576_100_10m/<run>/nn/ApproachStage2FanSensor.pth'
```

**현재 576개·100도·10m 설정과 같은 checkpoint만 사용할 수 있다.** 이전 정사각형, 1024개 부채꼴, 20m 부채꼴 checkpoint는 직접 테스트하거나 resume할 수 없다.

## 5. 저장된 정책의 테스트 시각화 — noVNC

```bash
TOKENHSI_GPU="$GPU" HEADLESS=0 SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_vnc.sh "$CKPT" 2 1 3 10
```

Checkpoint 뒤의 `2 1 3 10`은 에이전트 2명, 환경 1개, 상자 3개, 평가 반복 10회를 뜻한다. noVNC 접속 방법은 3절과 같다.

센서 표시 색상은 하늘색=직접 관측, 노랑=뒤쪽 관측 높이로 추정, 자홍색=미관측이다. 자홍색은 머리 높이 위에 띄운 표시이며 실제 지형 높이가 아니다. 표시를 끄려면 `SENSOR_VIS=0`으로 실행한다.

## 6. 로컬 화면에서 보기

그래픽 데스크톱이 있는 환경에서 사용한다. 서버 브라우저 접속은 앞의 noVNC 명령을 사용한다.

```bash
# 새 전이 학습과 화면 표시
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_watch.sh 2 1 3

# 저장된 정책 테스트와 화면 표시
TOKENHSI_GPU="$GPU" HEADLESS=0 SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_test.sh "$CKPT" 2 1 3 10
```

## 7. 화면 없는 짧은 테스트

```bash
TOKENHSI_GPU="$GPU" HEADLESS=1 EPISODE_LENGTH=64 \
OUTPUT_PATH=output/approach_stage2_fan_sensor_576_10m_check/eval \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_test.sh "$CKPT" 2 16 3 1
```

16환경·최대 64 step의 실행 확인용 평가다. 장기 학습 성능을 판단하는 평가와 구분한다.

## 8. checkpoint에서 학습 재개

```bash
# 2048환경 본학습 재개
TOKENHSI_GPU="$GPU" RESUME_CHECKPOINT="$CKPT" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_train.sh 2 2048 3

# 소수 환경에서 학습 재개와 noVNC 시각화
TOKENHSI_GPU="$GPU" RESUME_CHECKPOINT="$CKPT" SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_watch_vnc.sh 2 1 3
```

`RESUME_CHECKPOINT`가 설정되어 있으면 `STAGE1_CHECKPOINT`보다 우선한다. 새 전이를 시작할 때는 공통 준비의 `unset`을 실행한다. 각 명령은 실행한 터미널에서 `Ctrl+C`로 종료한다.

현재 낙하 실험은 상자를 표면·가림 판정에 포함한다. **향후 carryWith에서는 운반 상자 높이가 학습을 방해할 수 있으므로 상자 제외가 필요하다.** 센서 의미와 제외 범위의 후속 결정은 [센서 설계 문서](fan_sensor.md), 전체 실험 목록은 [실행 가이드](config.md)를 참고한다.
