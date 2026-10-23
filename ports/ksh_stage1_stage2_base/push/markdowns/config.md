# 실행 가이드

모든 명령은 저장소 루트에서 실행한다.

## 공통 규칙

- GPU6 두 5과제 물성의 1000회 저장본에서 PUSH 100% 재개는 [명령·샘플링 범위](reference/push100_resume1000.md)를 따른다. 학습 wrapper에 `TASK_GRAPH=push_at`, 별도 `OUTPUT_PATH`, `RESUME_CHECKPOINT`를 함께 지정한다.

- 5과제 PUSH 접촉·목표 진척/에피소드 이동량 로그와 기존 두 학습의 500회 재개는 [지표·재개 명세](reference/push_training_metrics.md)를 따른다. 보상과 모델 계약은 유지한다.

- PUSH RSI 신규/변경 작업은 [PUSH RSI 기준](../../../../markdowns/push_rsi_guidelines.md)을 먼저 따른다.

- **재선별 RSI 새 학습 두 설정**: `stage1_push_depth180_h100_rsi1m_at5deg_phase`·`stage1_push_depth180_h080_rsi1m_phase`; 커리큘럼 없이 목표1~4m, loco50%·선별 PUSH50%. [학습·로컬·VNC 명령](reference/push_rsi_phase_selection.md).

- **PUSH RSI 재선별 준비**: 손 높이≤1m와 실제 양팔 밀기 자세를 함께 만족하는5모션·1196프레임의 [별도 캐시·검수·재현 명령](reference/push_rsi_phase_selection.md). 기존 활성 학습 설정은 변경하지 않았다.

- 거리 커리큘럼 두 실험의 상태보존 이관·resume은 [별도 명세](reference/push_goal_curriculum.md)를 따른다.

- Y1.8m·20kg·원본RSI손중점≤1m 선별의 1.0m/AT5° 및0.8m/무제한 비교는 [명세](reference/push_depth180_rsi_le1m.md)를 따른다.

- 손 목표 높이1.0m 고정은 `stage1_push_depth180_safe_h100_relaxed`이며 [높이 명세](reference/push_depth180_height_compare.md#손-목표-높이10m-고정)를 따른다.

- Y1.8m 높은 손 위치의 비학습 전도 검사는 [사용법](USAGE.md#y18m-높은-손-위치-전도-검사)을 따른다.

- Y1.8m·손 높이0.65m/AMP 비교의 state 감쇠·RSI 높이는 [명세](reference/push_depth180_height_compare.md)를 따른다. 성공은 state>0.9와 0.5N 반력이며 별도 수평·높이 gate는 없다.

- safe PUSH 3종의 기본 CONTACT 힘 조건과 기존 checkpoint 구분은 [명세](reference/push_safe_start.md#contact-힘-조건-복구)를 따른다. train/test/VNC 명령은 동일하며 실행 중 학습에는 자동 반영되지 않는다.

- GPU5/GPU6 시작 collider 관통 수정은 [명세·캐시 준비](reference/push_safe_start.md)를 따른다. 새 `*_safe` checkpoint만 사용한다.

- X만가변·Y1.8m깊이고정 PUSH 준비는 [명세](reference/stage1_push_depth180_native_rsi.md)를 따른다.

- 중점 전용 CONTACT·20kg·X=Y·목표1~4m native RSI 준비는 [명세](reference/stage1_push_midpoint_native_rsi.md)를 따른다.

- 비학습 XYZ/동일질량·기울기 검사는 [분석 명세](reference/push_rect_stability.md)와 [USAGE](USAGE.md#직육면체-push-기울기-분석)를 따른다.

- 직육면체 PUSH 점 설정은 [별도 명세](reference/stage1_push_rect_point.md)를 따른다.

- GPU6 물성의 다섯 과제 두 설정·최종 목표1~5m·360° 시작/전신 표면 간격은 [명세](reference/five_task_gpu6_variants.md)를 따른다.

- 다섯 과제(PUSH_AT=4)·AMP4-family/133D·Stage2 PUSH 전용 coupled의 설정/호환성은 [명세](reference/five_task_push_at.md)를 따른다. Stage2의 `$STAGE1_CKPT`는 새 다섯 과제 Stage1 checkpoint 경로다.

- PUSH 추가 설정은 [별도 명세](reference/stage1_push.md)를 따른다. 이 사본에서 실행하며 원본 보관본 `../upstream/`은 변경하지 않는다.

- Config·스크립트·실행 옵션 변경 시 아래 표와 학습·로컬 추론·VNC 명령을 함께 갱신한다. 새 컨피그는 표와 각 명령 목록의 맨 아래에 추가한다. 실험별 설명·별도 섹션을 늘리지 않는다.
- 실행 전 `nvidia-smi`로 점유를 확인하고 `TOKENHSI_GPU`를 지정한다. 기존 프로세스는 임의 종료·재시작하지 않는다.
- 학습 인자: `[agents] [envs] [objects]`. stage1 계열은 **2명·2048환경·4물체**, stage2는 **4명·1024환경·8물체**를 사용한다. 공유 받침 두 실험은 **4명·1024환경**이며 마지막 인자는 총 obj 토큰 수(기본·최소 `12`)다. 본학습 전 `CHECK`, `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 해제한다.
- 추론 인자: `<checkpoint.pth> [agents] [envs] [objects] [repeats]`. 해당 실험의 checkpoint를 사용한다. test/VNC는 학습 당시 저장된 `relation_config.yaml`을 기본으로 읽는다.
- 본학습은 `output/`, 확인 학습·test·VNC 결과는 **`output_etc/check/`**에 저장한다. 짧은 학습은 `CHECK=1`을 붙이고 해당 실험의 환경 수(stage1 계열 2048, stage2 계열 1024)를 유지한다.
- 화면 없는 평가는 로컬 추론 명령의 `HEADLESS=0`을 `HEADLESS=1`로 바꾼다. 추가 옵션·resume·checkpoint 예외는 [USAGE](USAGE.md)를 따른다.

## 현재 컨피그

| 환경 Config | 학습 Config |
| --- | --- |
| [stage1.yaml](../tokenhsi/data/cfg/multi_agent/stage1.yaml) | [stage1.yaml](../tokenhsi/data/cfg/train/rlg/stage1.yaml) |
| [stage1_v2.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2.yaml) | [stage1_v2.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2.yaml) |
| [stage1_v2_fixed.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml) | [stage1_v2_fixed.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed.yaml) |
| [stage2.yaml](../tokenhsi/data/cfg/multi_agent/stage2.yaml) | [stage2.yaml](../tokenhsi/data/cfg/train/rlg/stage2.yaml) |
| [stage2_shared_full_success.yaml](../tokenhsi/data/cfg/multi_agent/stage2_shared_full_success.yaml) | [stage2_shared_full_success.yaml](../tokenhsi/data/cfg/train/rlg/stage2_shared_full_success.yaml) |
| [stage2_shared_region_success.yaml](../tokenhsi/data/cfg/multi_agent/stage2_shared_region_success.yaml) | [stage2_shared_region_success.yaml](../tokenhsi/data/cfg/train/rlg/stage2_shared_region_success.yaml) |
| [stage1_push.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push.yaml) | [stage1_push.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push.yaml) |
| [stage1_push_no_force.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_no_force.yaml) | [stage1_push_no_force.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_no_force.yaml) |
| [stage1_push_rect_point.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_rect_point.yaml) | [stage1_push_rect_point.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_rect_point.yaml) |
| [stage1_push_rect_native_rsi.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_rect_native_rsi.yaml) | [stage1_push_rect_native_rsi.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_rect_native_rsi.yaml) |
| [stage1_push_midpoint_native_rsi.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_midpoint_native_rsi.yaml) | [stage1_push_midpoint_native_rsi.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_midpoint_native_rsi.yaml) |
| [stage1_push_depth180_native_rsi.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_depth180_native_rsi.yaml) | [stage1_push_depth180_native_rsi.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_depth180_native_rsi.yaml) |
| [stage1_push_midpoint_safe.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_midpoint_safe.yaml) | [stage1_push_midpoint_safe.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_midpoint_safe.yaml) |
| [stage1_push_depth180_safe.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_depth180_safe.yaml) | [stage1_push_depth180_safe.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_depth180_safe.yaml) |
| [stage1_push_midpoint_safe_at5deg.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_midpoint_safe_at5deg.yaml) | [stage1_push_midpoint_safe_at5deg.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_midpoint_safe_at5deg.yaml) |
| [stage1_push_depth180_safe_h065_relaxed.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_depth180_safe_h065_relaxed.yaml) | [stage1_push_depth180_safe_h065_relaxed.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_depth180_safe_h065_relaxed.yaml) |
| [stage1_push_depth180_safe_hamp_relaxed.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_depth180_safe_hamp_relaxed.yaml) | [stage1_push_depth180_safe_hamp_relaxed.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_depth180_safe_hamp_relaxed.yaml) |
| [stage1_push_depth180_safe_h100_relaxed.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_depth180_safe_h100_relaxed.yaml) | [stage1_push_depth180_safe_h100_relaxed.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_depth180_safe_h100_relaxed.yaml) |
| [stage1_push_depth180_h100_rsi1m_at5deg.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg.yaml) | [stage1_push_depth180_h100_rsi1m_at5deg.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_depth180_h100_rsi1m_at5deg.yaml) |
| [stage1_push_depth180_h080_rsi1m.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_depth180_h080_rsi1m.yaml) | [stage1_push_depth180_h080_rsi1m.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_depth180_h080_rsi1m.yaml) |
| [stage1_push_depth180_h100_rsi1m_at5deg_goal_curriculum.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_goal_curriculum.yaml) | [stage1_push_depth180_h100_rsi1m_at5deg_goal_curriculum.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_depth180_h100_rsi1m_at5deg_goal_curriculum.yaml) |
| [stage1_push_depth180_h080_rsi1m_goal_curriculum.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_depth180_h080_rsi1m_goal_curriculum.yaml) | [stage1_push_depth180_h080_rsi1m_goal_curriculum.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_depth180_h080_rsi1m_goal_curriculum.yaml) |
| [stage1_push_depth180_h100_rsi1m_at5deg_phase.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_phase.yaml) | [stage1_push_depth180_h100_rsi1m_at5deg_phase.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_depth180_h100_rsi1m_at5deg_phase.yaml) |
| [stage1_push_depth180_h080_rsi1m_phase.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_depth180_h080_rsi1m_phase.yaml) | [stage1_push_depth180_h080_rsi1m_phase.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_depth180_h080_rsi1m_phase.yaml) |
| [stage1_push_depth180_h080_rsi1m_phase_face_xy.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_depth180_h080_rsi1m_phase_face_xy.yaml) | [stage1_push_depth180_h080_rsi1m_phase_face_xy.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_depth180_h080_rsi1m_phase_face_xy.yaml) |
| [stage1_push_depth180_h100_rsi1m_at5deg_phase_face_xy.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_phase_face_xy.yaml) | [stage1_push_depth180_h100_rsi1m_at5deg_phase_face_xy.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_depth180_h100_rsi1m_at5deg_phase_face_xy.yaml) |
| [stage1_push_face_xy_depth180_20kg.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_face_xy_depth180_20kg.yaml) | [stage1_push_face_xy_depth180_20kg.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_face_xy_depth180_20kg.yaml) |
| [stage1_push_face_xy_depth200_25kg.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_face_xy_depth200_25kg.yaml) | [stage1_push_face_xy_depth200_25kg.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_face_xy_depth200_25kg.yaml) |
| [stage1_push_face_xy_depth200_20kg_friction04.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push_face_xy_depth200_20kg_friction04.yaml) | [stage1_push_face_xy_depth200_20kg_friction04.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push_face_xy_depth200_20kg_friction04.yaml) |
| [stage1_v2_fixed_five_task.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_five_task.yaml) | [stage1_v2_fixed_five_task.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed_five_task.yaml) |
| [stage2_shared_full_success_five_task.yaml](../tokenhsi/data/cfg/multi_agent/stage2_shared_full_success_five_task.yaml) | [stage2_shared_full_success_five_task.yaml](../tokenhsi/data/cfg/train/rlg/stage2_shared_full_success_five_task.yaml) |
| [stage1_v2_fixed_five_task_depth180_20kg.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_five_task_depth180_20kg.yaml) | [stage1_v2_fixed_five_task_depth180_20kg.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed_five_task_depth180_20kg.yaml) |
| [stage1_v2_fixed_five_task_depth200_20kg_friction04.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_five_task_depth200_20kg_friction04.yaml) | [stage1_v2_fixed_five_task_depth200_20kg_friction04.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed_five_task_depth200_20kg_friction04.yaml) |


## 학습

```bash
# stage1
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_train.sh 2 2048 4

# stage1_v2
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_v2_train.sh 2 2048 4

# stage1_v2_fixed
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_train.sh 2 2048 4

# stage2
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_train.sh 4 1024 8

# stage2_shared_full_success
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_shared_full_success_train.sh 4 1024 12

# stage2_shared_region_success
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_shared_region_success_train.sh 4 1024 12

# stage1_push
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_train.sh 2 2048 4

# stage1_push_no_force
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_no_force_train.sh 2 2048 4

# stage1_push_rect_point
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_rect_point_train.sh 2 2048 4

# stage1_push_rect_native_rsi
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_rect_native_rsi_train.sh 2 2048 4

# stage1_push_midpoint_native_rsi
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_midpoint_native_rsi_train.sh 2 2048 4

# stage1_push_depth180_native_rsi
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_native_rsi_train.sh 2 2048 4

# stage1_push_midpoint_safe
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage1_push_midpoint_safe_train.sh 2 2048 4

# stage1_push_depth180_safe
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_train.sh 2 2048 4

# stage1_push_midpoint_safe_at5deg
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage1_push_midpoint_safe_at5deg_train.sh 2 2048 4

# stage1_push_depth180_safe_h065_relaxed
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_h065_relaxed_train.sh 2 2048 4

# stage1_push_depth180_safe_hamp_relaxed
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_hamp_relaxed_train.sh 2 2048 4

# stage1_push_depth180_safe_h100_relaxed
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_h100_relaxed_train.sh 2 2048 4

# stage1_push_depth180_h100_rsi1m_at5deg
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_train.sh 2 2048 4

# stage1_push_depth180_h080_rsi1m
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_train.sh 2 2048 4

# stage1_push_depth180_h100_rsi1m_at5deg_goal_curriculum
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_goal_curriculum_train.sh 2 2048 4

# stage1_push_depth180_h080_rsi1m_goal_curriculum
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_goal_curriculum_train.sh 2 2048 4

# stage1_push_depth180_h100_rsi1m_at5deg_phase
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_phase_train.sh 2 2048 4

# stage1_push_depth180_h080_rsi1m_phase
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_phase_train.sh 2 2048 4

# stage1_push_depth180_h080_rsi1m_phase_face_xy
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_phase_face_xy_train.sh 2 2048 4

# stage1_push_depth180_h100_rsi1m_at5deg_phase_face_xy
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_phase_face_xy_train.sh 2 2048 4

# stage1_push_face_xy_depth180_20kg
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth180_20kg_train.sh 2 2048 4

# stage1_push_face_xy_depth200_25kg
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth200_25kg_train.sh 2 2048 4

# stage1_push_face_xy_depth200_20kg_friction04
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth200_20kg_friction04_train.sh 2 2048 4

# stage1_v2_fixed_five_task
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_train.sh 2 2048 4

# stage2_shared_full_success_five_task
TOKENHSI_GPU=6 STAGE1_CHECKPOINT="$STAGE1_CKPT" bash tokenhsi/scripts/multi_agent/stage2_shared_full_success_five_task_train.sh 4 1024 12

# stage1_v2_fixed_five_task_depth180_20kg
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth180_20kg_train.sh 2 2048 4

# stage1_v2_fixed_five_task_depth200_20kg_friction04
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth200_20kg_friction04_train.sh 2 2048 4
```

## 로컬 추론

```bash
CKPT='/absolute/path/to/checkpoint.pth'

# stage1
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_test.sh "$CKPT" 2 1 4 10

# stage1_v2
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_v2_test.sh "$CKPT" 2 1 4 10

# stage1_v2_fixed
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_test.sh "$CKPT" 2 1 4 10

# stage2
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_test.sh "$CKPT" 4 1 8 10

# stage2_shared_full_success
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_shared_full_success_test.sh "$CKPT" 4 1 12 10

# stage2_shared_region_success
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_shared_region_success_test.sh "$CKPT" 4 1 12 10

# stage1_push
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_test.sh "$CKPT" 2 1 4 10

# stage1_push_no_force
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_no_force_test.sh "$CKPT" 2 1 4 10

# stage1_push_rect_point
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_rect_point_test.sh "$CKPT" 2 1 4 10

# stage1_push_rect_native_rsi
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_rect_native_rsi_test.sh "$CKPT" 2 1 4 10

# stage1_push_midpoint_native_rsi
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_midpoint_native_rsi_test.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_native_rsi
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_native_rsi_test.sh "$CKPT" 2 1 4 10

# stage1_push_midpoint_safe
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_midpoint_safe_test.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_safe
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_test.sh "$CKPT" 2 1 4 10

# stage1_push_midpoint_safe_at5deg
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_midpoint_safe_at5deg_test.sh /absolute/path/to/checkpoint.pth 2 1 4 10

# stage1_push_depth180_safe_h065_relaxed
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_h065_relaxed_test.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_safe_hamp_relaxed
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_hamp_relaxed_test.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_safe_h100_relaxed
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_h100_relaxed_test.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_h100_rsi1m_at5deg
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_test.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_h080_rsi1m
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_test.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_h100_rsi1m_at5deg_goal_curriculum
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_goal_curriculum_test.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_h080_rsi1m_goal_curriculum
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_goal_curriculum_test.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_h100_rsi1m_at5deg_phase
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_phase_test.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_h080_rsi1m_phase
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_phase_test.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_h080_rsi1m_phase_face_xy
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_phase_face_xy_test.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_h100_rsi1m_at5deg_phase_face_xy
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_phase_face_xy_test.sh "$CKPT" 2 1 4 10

# stage1_push_face_xy_depth180_20kg
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth180_20kg_test.sh "$CKPT" 2 1 4 10

# stage1_push_face_xy_depth200_25kg
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth200_25kg_test.sh "$CKPT" 2 1 4 10

# stage1_push_face_xy_depth200_20kg_friction04
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth200_20kg_friction04_test.sh "$CKPT" 2 1 4 10

# stage1_v2_fixed_five_task
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_test.sh "$CKPT" 2 1 4 10

# stage2_shared_full_success_five_task
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_shared_full_success_five_task_test.sh "$CKPT" 4 1 12 10

# stage1_v2_fixed_five_task_depth180_20kg
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth180_20kg_test.sh "$CKPT" 2 1 4 10

# stage1_v2_fixed_five_task_depth200_20kg_friction04
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth200_20kg_friction04_test.sh "$CKPT" 2 1 4 10
```

## 서버 VNC 추론

```bash
CKPT='/absolute/path/to/checkpoint.pth'

# stage1
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_vnc.sh "$CKPT" 2 1 4 10

# stage1_v2
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_v2_vnc.sh "$CKPT" 2 1 4 10

# stage1_v2_fixed
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_vnc.sh "$CKPT" 2 1 4 10

# stage2
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_vnc.sh "$CKPT" 4 1 8 10

# stage2_shared_full_success
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_shared_full_success_vnc.sh "$CKPT" 4 1 12 10

# stage2_shared_region_success
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_shared_region_success_vnc.sh "$CKPT" 4 1 12 10

# stage1_push
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_no_force
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_no_force_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_rect_point
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_rect_point_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_rect_native_rsi
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_rect_native_rsi_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_midpoint_native_rsi
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_midpoint_native_rsi_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_native_rsi
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_native_rsi_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_midpoint_safe
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage1_push_midpoint_safe_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_safe
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_midpoint_safe_at5deg
TOKENHSI_GPU=5 PORT=6115 bash tokenhsi/scripts/multi_agent/stage1_push_midpoint_safe_at5deg_vnc.sh /absolute/path/to/checkpoint.pth 2 1 4 10

# stage1_push_depth180_safe_h065_relaxed
TOKENHSI_GPU=6 PORT=6125 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_h065_relaxed_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_safe_hamp_relaxed
TOKENHSI_GPU=6 PORT=6126 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_hamp_relaxed_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_safe_h100_relaxed
TOKENHSI_GPU=6 PORT=6127 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_h100_relaxed_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_h100_rsi1m_at5deg
TOKENHSI_GPU=6 PORT=6131 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_h080_rsi1m
TOKENHSI_GPU=6 PORT=6132 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_h100_rsi1m_at5deg_goal_curriculum
TOKENHSI_GPU=6 PORT=6133 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_goal_curriculum_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_h080_rsi1m_goal_curriculum
TOKENHSI_GPU=6 PORT=6133 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_goal_curriculum_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_h100_rsi1m_at5deg_phase
TOKENHSI_GPU=6 PORT=6134 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_phase_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_h080_rsi1m_phase
TOKENHSI_GPU=6 PORT=6134 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_phase_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_h080_rsi1m_phase_face_xy
TOKENHSI_GPU=6 PORT=6140 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_phase_face_xy_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_depth180_h100_rsi1m_at5deg_phase_face_xy
TOKENHSI_GPU=6 PORT=6141 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_phase_face_xy_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_face_xy_depth180_20kg
TOKENHSI_GPU=6 PORT=6142 bash tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth180_20kg_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_face_xy_depth200_25kg
TOKENHSI_GPU=6 PORT=6143 bash tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth200_25kg_vnc.sh "$CKPT" 2 1 4 10

# stage1_push_face_xy_depth200_20kg_friction04
TOKENHSI_GPU=6 PORT=6144 bash tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth200_20kg_friction04_vnc.sh "$CKPT" 2 1 4 10

# stage1_v2_fixed_five_task
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_vnc.sh "$CKPT" 2 1 4 10

# stage2_shared_full_success_five_task
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_shared_full_success_five_task_vnc.sh "$CKPT" 4 1 12 10

# stage1_v2_fixed_five_task_depth180_20kg
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth180_20kg_vnc.sh "$CKPT" 2 1 4 10

# stage1_v2_fixed_five_task_depth200_20kg_friction04
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth200_20kg_friction04_vnc.sh "$CKPT" 2 1 4 10
```

## 추론 옵션

아래 옵션을 test/VNC 명령의 `bash` 앞에 붙인다. 기본값은 `TASK_GRAPH=random_scenario EVAL_SKILLS=loco`다.

5과제 Stage1 `TASK_GRAPH=push_at EVAL_SKILLS=loco`에서 `PUSH_EVAL_SIZE_X=0.9,1.8 PUSH_EVAL_SIZE_Y=1.7,2.3 PUSH_EVAL_SIZE_Z=1.0,1.6`으로 실제 PUSH 상자의 XYZ를 독립 샘플링하는 크기 제로샷을 실행한다. 위치 전환 옵션은 해제한다. [명령·샘플링/물성 규칙](USAGE.md#push-크기-제로샷).

5과제 Stage1 PUSH에서 `PUSH_BOX_SWITCH_MODES=across_goal,behind_agent`는 목표를 고정하고 기본60step(2초) 뒤 상자를 목표 반대편/현재 사람 뒤쪽으로 반복별 교대 이동한다. 목표 전환 옵션과 동시에 사용하지 않는다. [로컬·headless·VNC 명령](USAGE.md#push-중간-상자-이동-제로샷).

5과제 Stage1의 `TASK_GRAPH=push_at` 평가에서 `PUSH_GOAL_SWITCH_ANGLES=90,180`을 붙이면 반복마다 90°/180°를 번갈아 적용한다. 기본150step(5초) 후 현재 상자에서2m 떨어진 새 목표로 전환한다. [로컬·headless·VNC 명령과 추가 옵션](USAGE.md#push-중간-목표-전환-제로샷).

**TASK_GRAPH — 수행할 과제**

다섯 과제 설정은 Stage1 `random_scenario`/`push_at` 및 기존 네 과제 preset을 지원한다. 새 Stage2는 `random_scenario`/`all_solo`/`all_shared`/`push_at`/`shared_push` 및 기존 네 과제 solo preset을 지원한다. `shared_push`는 전원이 같은 PUSH 상자와 목표를 공유한다.

| 값 | 동작 |
| --- | --- |
| `random_scenario` | 사람마다 과제 무작위 선택 |
| `sit` | 모두 SIT |
| `climb` | 모두 CLIMB |
| `sit_climb` | 사람 순서대로 SIT·CLIMB 교대 |
| `carry_at` | 각자 물체를 자기 목표 위치로 운반 |
| `carry_ontop` | 각자 물체를 받침 위로 운반 |

`holding_at`은 `carry_at`, `holding_ontop`은 `carry_ontop`과 같다.

**EVAL_SKILLS — 시작할 참조 동작**

| 값 | 초기 동작 |
| --- | --- |
| `loco` | 걷기·접근 |
| `push` | 5과제 Stage1의 PUSH 슬롯에서 선별 PUSH RSI 자세로 시작 |
| `sit` | 앉기 |
| `climb` | 오르기 |
| `pickUp` | 물체 집기 |
| `carryWith` | 물체를 들고 운반 |
| `putDown` | 물체 내려놓기 |
| `climbNoRSI` | 별도 Climb 참조 모션 묶음 |
| `omomo` | OMOMO 운반 참조 모션 묶음 |

여러 skill은 쉼표로 나열하고 `EVAL_SKILL_PROBS`를 같은 순서·개수로 지정한다. 하나만 지정하면 기본 확률은 `1.0`이다. skill은 사람별로 추첨하며 과제별 자동 배정이나 정확한 모션 프레임 고정은 아니다.

```bash
# AT 운반 — 접근부터 시작
TASK_GRAPH=carry_at EVAL_SKILLS=loco

# AT 운반 — 들고 있는 동작에서 시작
TASK_GRAPH=carry_at EVAL_SKILLS=carryWith

# ON_TOP 운반 — 접근 50%, 운반 50%
TASK_GRAPH=carry_ontop EVAL_SKILLS=loco,carryWith EVAL_SKILL_PROBS=0.5,0.5

# 5과제 Stage1 PUSH — 선별 PUSH RSI 100%, 가까운 사람 카메라
TASK_GRAPH=push_at EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 TASK_CAMERA=agent
```

Stage2의 `TASK_GRAPH`는 `random_scenario`, `two_pairs_at`, `two_pairs_ontop`, `one_pair_at`, `one_pair_ontop`, `all_solo`, `sit`, `climb`, `carry_at`, `carry_ontop`을 지원한다. `carry_at`·`carry_ontop`은 전원 단독 운반이며, 공동 구성을 고정하려면 pair preset을 쓴다.

공동 그룹의 `EVAL_SKILLS`는 `loco`, `pickUp`, `carryWith`, `putDown`을 지원한다. Stage2 ON_TOP에는 `putDown`을 지정하지 않는다. SIT/CLIMB 참조 동작은 해당 전원 단독 preset에서 사용한다.

공유 받침 두 실험은 `random_scenario`, `all_solo`, `all_shared`, `sit`, `climb`, `carry_at`, `carry_ontop`, `shared_sit`, `shared_climb`, `shared_ontop`을 지원하며, `EVAL_SKILLS=loco`만 사용한다. `shared_*`는 전원이 하나의 받침을 공유한다.
