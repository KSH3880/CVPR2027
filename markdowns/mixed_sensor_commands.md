# 혼합 시나리오 실행 명령

고정 지형·RSI 발 보정이 적용된 코드 기준이다. 실행 중인 프로세스에는 자동 반영되지 않는다. 기존 pickup 50% 혼합 checkpoint도 사용할 수 있으며, 새 학습 출력은 벽 CPA 교체(2026-10-06)에 따라 `_cpa_static_wall`로 분리된다.

## 공통 준비

```bash
cd /home/user/jhh/Projects/TokenHSI_total_sensor
nvidia-smi

GPU=0
STAGE1=output/stage1/8000.pth  # train/watch 기본값이라 STAGE1_CHECKPOINT는 생략 가능
# pickup_cross_probability: 0.5 설정으로 학습한 혼합 checkpoint 경로로 변경
CKPT="output/approach_stage2_fan_mixed_cpa_static_wall/<run>/nn/ApproachStage2FanMixed.pth"

unset MAX_ITERATIONS RESUME_CHECKPOINT OUTPUT_PATH EPISODE_LENGTH RELATION_GRAPH
export TASK_GRAPH=random_scenario SENSOR_MASK=0
```

## 1. 학습

```bash
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_mixed_train.sh 2 2048 3
```

## 2. 테스트 · 화면 없음

```bash
TOKENHSI_GPU="$GPU" HEADLESS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_mixed_test.sh "$CKPT" 2 96 3 3
```

## 3. 학습 시각화 · 4환경 실제 학습

```bash
# 로컬 viewer
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_mixed_watch.sh 2 4 3

# 서버 VNC
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_mixed_watch_vnc.sh 2 4 3
```

## 4. 테스트 시각화

```bash
# 로컬 viewer
TOKENHSI_GPU="$GPU" HEADLESS=0 SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_mixed_test.sh "$CKPT" 2 1 3 10

# 서버 VNC
TOKENHSI_GPU="$GPU" HEADLESS=0 SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_mixed_vnc.sh "$CKPT" 2 1 3 10
```
