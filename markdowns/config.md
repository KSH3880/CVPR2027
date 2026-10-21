# Multi-Agent Carry 실행 가이드

현재 실행 가능한 실험은 원본 1번과 Stage 1 **27·28·30·31·32·33·34·35번 및 34번 distillation**, Stage 2 **29번·SIT plane·무대 센서·두 무대 다리·4면 벽 room·3종 혼합·3종 혼합 정사각형 센서**이다. 과거 9~26번 config와 전용 실행 스크립트는 정리했다. 모든 명령은 저장소 루트에서 실행한다.

## 공통 규칙

- 학습 인자는 `[num_agents] [num_envs] [num_objects]`; Stage 1 기본값은 `2 2048 3`이다. 짧은 확인도 환경 수 2048을 유지하고 `MAX_ITERATIONS`만 줄인다. 사용자가 요청한 센서 Stage 2 시각화용 `watch`는 별도로 1환경 이상을 지원한다.
- 실행 전 `nvidia-smi`로 점유를 확인하고 명령 앞에 `TOKENHSI_GPU`를 지정한다. 실행 중인 프로세스는 임의 종료하지 않는다.
- 본학습 전 셸의 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 확인한다.
- 학습은 기본적으로 scratch다. 기존 실험의 reward/checkpoint 계약을 섞지 않는다. Stage 2 전이는 아래의 `STAGE1_CHECKPOINT`를 사용한다.
- 평가·VNC 인자는 `<checkpoint.pth> [agents] [envs] [objects] [repeats]`다. `HEADLESS=0`은 로컬 viewer, `HEADLESS=1`은 화면 없는 평가다.
- 로컬 viewer와 서버 VNC의 상자 색은 매 reset의 과제 배정을 따른다. 단독 대상은 해당 에이전트 색, 공동 대상은 노란색, 나머지는 회색이다. 실행 중인 뷰어는 다시 시작해야 반영된다.
- 데이터 원본 `/home/user/jhh/Projects/TokenHSI`는 읽기 전용이며, 이 저장소는 심링크로 사용한다.

## 현재 실험

| 번호 | Config | 역할 |
| ---: | --- | --- |
| 1 | [amp_humanoid_ma_carry.yaml](../tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml) | 원본 multi-agent carry 기준 |
| 27 | [approach_scenario_independent_with_climb.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_independent_with_climb.yaml) | 독립 물체·goal binding, 점 기준 성공 |
| 28 | [approach_scenario_stage1_plane.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_plane.yaml) | 27번의 ON_TOP·CLIMB을 상판 영역 성공으로 변경; Stage 2 기본 출발점 |
| 29 | [approach_stage2_coordination.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_coordination.yaml) | Stage 1 checkpoint에서 협력 graph 학습 |
| Stage 2 plane | [approach_stage2_coordination_sit_plane.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_coordination_sit_plane.yaml) | 34번 기반 SIT plane·보상·독립 과제 분포 |
| Stage 2 sensor | [approach_stage2_cliff_sensor.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_cliff_sensor.yaml) | 무대 위 두 에이전트, 높이 센서 late fusion, loco RSI·HOLDING/SIT/CLIMB |
| Stage 2 fan sensor | [approach_stage2_fan_sensor.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_fan_sensor.yaml) | 머리 시점·전방 100도/10m·576개 균일 고리·가림 및 연결된 뒤쪽 관측 채움 |
| Stage 2 fan sensor watch | [approach_stage2_fan_sensor_watch.sh](../tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_watch.sh) | 부채꼴 센서의 소수 환경 학습 시각화; 기본 4환경, 최소 1환경 |
| Stage 2 fan bridge | [approach_stage2_fan_bridge.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_fan_bridge.yaml) | 단일 무대/두 무대+외력 다리 5:5, 가림 없는 1024개 부채꼴 수직 조회 |
| Stage 2 fan bridge watch | [approach_stage2_fan_bridge_watch.sh](../tokenhsi/scripts/multi_agent/approach_stage2_fan_bridge_watch.sh) | 다리 실험의 소수 환경 학습 시각화; 기본 4환경, 최소 1환경 |
| Stage 2 fan room | [approach_stage2_fan_room.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_fan_room.yaml) | 4면 벽·실내 두 에이전트/객체 3개, 1024개 가림 없는 센서, 사람·벽 CPA를 단일 collision 항으로 통합 |
| Stage 2 fan room partition | [approach_stage2_fan_room_partition.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_fan_room_partition.yaml) | 현재 room 확장: 내부 칸막이·양 끝 통로, HOLDING/SIT/CLIMB/HOLDING_AT 각각 25%, 반대편 운반 목표 |
| Stage 2 fan mixed | [approach_stage2_fan_mixed.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_fan_mixed.yaml) | 학습·평가 reset마다 단일 무대/연결다리/칸막이 room 각 1/3, 모든 장면에서 네 과제 각 25% |
| Stage 2 fan mixed watch | [approach_stage2_fan_mixed_watch.sh](../tokenhsi/scripts/multi_agent/approach_stage2_fan_mixed_watch.sh) | 혼합 실험의 소수 환경 학습 viewer; 기본 4환경·PPO/AMP minibatch 64·별도 output |
| Stage 2 square mixed | [approach_stage2_square_mixed.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_square_mixed.yaml) | fan mixed와 같은 장면·과제·보상; 센서만 root 중심·heading 회전 14×14m 정사각형 48×48(에이전트당 2304점) |
| Stage 2 sensor watch | [amp_ma_cliff_sensor_watch.yaml](../tokenhsi/data/cfg/train/rlg/amp_ma_cliff_sensor_watch.yaml) | 같은 무대·정책의 소수 환경 학습 시각화; 기본 4환경, 최소 1환경 |
| 30 | [approach_scenario_stage1_sit_plane_normalized.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_sit_plane_normalized.yaml) | 28번 + SIT 상판 성공·유효 edge 평균 보상 |
| 31 | [approach_scenario_stage1_skill_curriculum.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_skill_curriculum.yaml) | 30번 + 가까운 시작·후반 CLIMB RSI·어려운 과제 증량·보상 shaping |
| 32 | [approach_scenario_stage1_skill_curriculum_reward_preserved.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved.yaml) | 31번의 시작 상태·RSI·샘플링을 유지하고 state/progress는 30번 식 |
| 33 | [approach_scenario_stage1_at_goal_fix.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_at_goal_fix.yaml) | 32번 + AT 목표 slot reset 오류 수정 |
| 34 | [approach_scenario_stage1_self_sum.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_self_sum.yaml) | 33번 + 자기 edge 보상 합계·동료 edge 평균, 과제 비율 10/10/10/35/35 |
| 34 distill | [approach_scenario_stage1_self_sum_distill.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_self_sum_distill.yaml) | 34번의 graph·보상 유지, 동결한 원본 통합 Stage 1 teacher 행동 분포 KL 추가 |
| 35 | [approach_scenario_stage1_slow_near_start.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_slow_near_start.yaml) | 34번 + 가까운 시작 80%를 60만 step 유지하고 120만 step까지 30%로 감소 |

27~35번 Stage 1은 2-agent/3-object 독립 graph와 schema 9 semantic 관찰을 쓴다. ON_TOP 두 개를 동시에 샘플하지 않는다. 28~35번의 팀 보상은 자기 0.9, 동료 0.1이다. 30~33번은 유효 edge를 agent별로 평균 내어 단일 edge와 두 edge의 최대 task 보상을 0.6으로 맞춘다. 34·35번은 자기 edge를 합산하고 동료 edge만 평균 내며 state·progress·성공 식은 33번과 같다.

33번은 32번의 보상·샘플링·RSI를 유지한다. AT graph가 두 번째 목표 slot을 가리킬 때 reset도 해당 목표를 배치하도록 공통 코드의 slot 선택을 수정했다. 수정 후 새로 시작하는 27~32번 실행에도 적용되지만, 이미 실행 중인 프로세스와 checkpoint는 바뀌지 않는다. 비교 실험은 33번 전용 output에 scratch로 시작한다.

## 학습

아래 `GPU`를 사용할 장치 번호로 바꾼다. 27·28·30·31·32·33·34·35번은 각각 별도 output에 scratch로 시작한다.

```bash
GPU=5
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_independent_with_climb_train.sh 2 2048 3  # 27
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_plane_train.sh 2 2048 3  # 28
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_sit_plane_normalized_train.sh 2 2048 3  # 30
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_train.sh 2 2048 3  # 31
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved_train.sh 2 2048 3  # 32
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_at_goal_fix_train.sh 2 2048 3  # 33
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_self_sum_train.sh 2 2048 3  # 34
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_slow_near_start_train.sh 2 2048 3  # 35
```

한 번에 필요한 실험 **한 줄만** 실행한다. 짧은 확인은 별도 output을 쓴다.

```bash
MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_scenario_stage1_skill_curriculum_reward_preserved_check \
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved_train.sh 2 2048 3
```

정기 checkpoint 저장 간격은 학습 설정에서 500 epoch다. `.pth`는 각 run의 `nn/`, TensorBoard는 `summaries/`, edge 진단은 `diagnostics/`에 저장된다.

### 34번 distillation

34번과 같은 graph·보상·AMP·PPO를 유지한다. 원본 통합 Stage 1 checkpoint를 동결하고, 학생 rollout의 각 상태에서 teacher 분포를 계산해 `KL(teacher || student)`를 PPO loss에 더한다. SIT은 Sit, CLIMB은 Climb, HOLDING/HOLDING_AT/HOLDING_ON_TOP은 Carry teacher를 쓴다. AT 목표는 graph의 goal, ON_TOP 목표는 받침 상자 상판 위의 source 중심이다. 목표가 없는 단독 HOLDING은 현재 source 상자 중심을 Carry 목표로 준다. SIT의 상자 방향은 상자 로컬 +X를 사용한다. 이 두 proxy의 행동 적합성과 장기 성공률은 별도 평가가 필요하다.

이 서버의 `tokenhsi` 환경은 CPU 전용 PyTorch여서 GPU 실행에는 `tokenhsi_sm120`을 사용한다. 아래 명령은 한 GPU만 지정하며, 실행 전 `nvidia-smi`로 점유를 확인한다. Teacher 기본 경로는 `../TokenHSI/output/tokenhsi/ckpt_stage1.pth`이며 `TEACHER_CHECKPOINT`로 바꿀 수 있다.

```bash
GPU=0
TOKENHSI_CONDA_ENV=tokenhsi_sm120 TOKENHSI_GPU="$GPU" \
  bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_self_sum_distill_train.sh 2 2048 3
```

`TEACHER_KL_COEF` 기본값은 `0.001`, `TEACHER_GRAD_CHECKS` 기본값은 `0`이다. `RESUME_CHECKPOINT`는 **같은 distillation 실험**의 checkpoint를 이어갈 때 지정한다. 확인 실행은 별도 output으로 분리한다.

```bash
TOKENHSI_CONDA_ENV=tokenhsi_sm120 TOKENHSI_GPU="$GPU" \
MAX_ITERATIONS=1 TEACHER_GRAD_CHECKS=1 \
OUTPUT_PATH=output/approach_scenario_stage1_self_sum_distill_check \
  bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_self_sum_distill_train.sh 2 2048 3
```

`MAX_ITERATIONS=1`은 현재 rl_games 종료 조건에서 2 epoch를 실행한다. 학생 checkpoint만으로 평가하며 teacher checkpoint는 필요하지 않다. 아래 `CKPT`에는 이 실험의 실제 `.pth` 경로를 넣는다.

```bash
CKPT='/absolute/path/to/ApproachScenarioStage1SelfSumDistill.pth'
TOKENHSI_CONDA_ENV=tokenhsi_sm120 TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at \
  bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_self_sum_distill_test.sh "$CKPT" 2 1 3 10
TOKENHSI_CONDA_ENV=tokenhsi_sm120 TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop \
  bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_self_sum_distill_vnc.sh "$CKPT"
```

이 서버의 VNC wrapper는 실행 중인 `tokenhsi-gui.service`의 화면을 사용한다. 서비스는 저장소 밖에서 `5802`(noVNC)·`5902`(VNC)를 사용하므로, VS Code **PORTS**에서 원격 포트 `5802`를 포워딩하고 `http://127.0.0.1:5802/`를 연다. 저장소의 `run-gui.sh`가 별도 Xvfb 화면을 띄울 때 기본값은 `5803`(noVNC)·`5903`(VNC)이지만, 이 서버에는 Xvfb가 없어 현재 서비스 재사용 경로가 적용된다. 기존 `tokenhsi` 셸에서 실행해도 명령 앞의 `TOKENHSI_CONDA_ENV=tokenhsi_sm120`을 유지한다.

화면 없는 분포 평가는 test 명령에 `HEADLESS=1 TASK_GRAPH=random_scenario`와 평가 환경 수 `64`를 지정한다. TensorBoard `distill/`에는 KL, 행동 평균 오차, actor/critic loss, PPO clip 비율과 다섯 template label 수가 기록된다. Stage 2 SIT plane 전이에는 이 실험의 schema-9 checkpoint를 `STAGE1_CHECKPOINT`로 지정한다. 기존 34번 및 실행 중인 Stage 2 output은 별도다.

## 로컬 평가와 서버 VNC

학습과 **같은 번호의 checkpoint와 스크립트**를 사용한다. `CKPT`를 실제 `.pth` 절대 경로로 설정한다.

```bash
GPU=5
CKPT='/absolute/path/to/checkpoint.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_independent_with_climb_test.sh "$CKPT" 2 1 3 10  # 27
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_plane_test.sh "$CKPT" 2 1 3 10  # 28
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=sit bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_sit_plane_normalized_test.sh "$CKPT" 2 1 3 10  # 30
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_test.sh "$CKPT" 2 1 3 10  # 31
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved_test.sh "$CKPT" 2 1 3 10  # 32
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_at_goal_fix_test.sh "$CKPT" 2 1 3 10  # 33
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_self_sum_test.sh "$CKPT" 2 1 3 10  # 34
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_slow_near_start_test.sh "$CKPT" 2 1 3 10  # 35
```

화면 없는 학습 분포 평가는 해당 test 명령에서 `HEADLESS=1 TASK_GRAPH=random_scenario`와 평가 환경 수 `64`를 지정한다. 로컬 test와 서버 VNC는 모두 기본적으로 `output/<실험명>/metrics/`와 `diagnostics/`에 평가 결과를 쓴다. 반복 평가 시 진단 CSV는 같은 파일에 이어 쓰므로 실행별 분석은 `OUTPUT_PATH`를 따로 지정한다. 학습 checkpoint는 별도 run 디렉터리의 `nn/`에 저장된다. 아래는 서버 viewer 실행 명령이다.

```bash
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_independent_with_climb_vnc.sh "$CKPT"  # 27
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_plane_vnc.sh "$CKPT"  # 28
TOKENHSI_GPU="$GPU" TASK_GRAPH=sit bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_sit_plane_normalized_vnc.sh "$CKPT"  # 30
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_vnc.sh "$CKPT"  # 31
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved_vnc.sh "$CKPT"  # 32
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_at_goal_fix_vnc.sh "$CKPT"  # 33
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_self_sum_vnc.sh "$CKPT"  # 34
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_slow_near_start_vnc.sh "$CKPT"  # 35
```

## Stage 2 무대·높이 센서

동기·구성·학습 결과와 직접 시각 확인은 [2026-09-29 실험 보고서](stage2_cliff_sensor_experiment_report_20260929.md)에 정리했다.

`HumanoidMACliff`는 기존 협력 Stage 2와 별개다. Stage 1 schema-9 checkpoint의 actor/critic/AMP와 관찰 통계를 이식한다. Teacher는 사용하지 않는다. 기존 607차원 scene 뒤에 에이전트별 1024개 높이 관찰을 붙여 총 **2655차원**을 사용한다.

- **무대:** 환경마다 가로·세로를 독립적으로 4–8m, 0.5m 간격에서 샘플링한다. 높이는 1.5m. 크기는 환경 생성 시 정해지고 episode reset에서는 유지된다. 실제 고정 충돌 박스는 환경의 **world 원점**에 생성한다.
- **배치:** 두 명은 같은 무대 위, 세 객체는 무대 밖 지면에 놓인다. 객체의 회전된 전체 footprint가 무대에서 최소 0.8m 떨어지도록 보수적인 외접 반경으로 배치한다. 안전한 하강 경로는 없다.
- **과제/RSI:** HOLDING·SIT·CLIMB를 각각 1/3 확률로 샘플링한다. 모든 RSI는 loco만 사용하고, AMP 참조 모션 분포와 기존 34번 reward 식·가중치는 유지한다.
- **센서:** heading을 따라 회전하는 ±2m, 32×32 그리드. 무대/지면 높이에서 pelvis 높이를 빼고 `[-3, 3]m`로 clip한 뒤 3으로 나눈다. 움직이는 객체는 기존 O 토큰으로 관찰한다.
- **정책:** H/O/T tokenizer·Transformer·기존 edge encoder·actor 관찰 RMS는 동결한다. 각 agent의 H64와 sensor64를 concat하여 action head에 전달한다. 새 `OWN_SENSOR` edge는 없다. Sensor encoder·action head·critic·AMP discriminator는 학습한다.
- **전이:** actor와 critic 첫 head 가중치를 `[W_old, 0]`으로 확장한다. 기존 입력이 같으면 초기 action/value가 동일하다. Sensor encoder 자체는 무작위 초기화이며 balancing scalar는 없다. 최초 backward에서 센서 encoder gradient가 0일 수 있어 후속 업데이트까지 검증한다.
- **종료:** 무대 밖 지면은 **발을 포함한 어느 신체 부위든 닿으면 실패**한다. 같은 부위의 접촉력과 지면 상대 높이를 함께 검사하므로 두 발로 정상 착지해도 종료한다. 무대 위에서는 기존의 발 제외 접촉·낮은 신체 높이 낙상 판정을 유지한다. 무대 위 정상 발 접촉과 무대 밖 공중 상태는 허용한다. Reset 직후 두 step의 유예는 유지한다. 한 명이 실패하면 전체 환경을 reset하지만 정상 동료의 value bootstrap은 유지한다. `cliff/*/falls`에는 바닥 접촉 실패도 포함된다.

### 좌표계 계약

| 데이터 | 좌표계/변환 |
| --- | --- |
| 물리 무대 중심 | `env_origin + [0, 0, height/2]`; 고정 actor 생성 때 적용 |
| RSI·humanoid 물리 상태 | motion을 root 중심으로 yaw 회전하고 위치를 무대 위 world 좌표로 이동; body pose와 속도도 함께 회전 |
| H223의 root 높이 | `root_world_z - env_origin_z - stage_height`; 기존 평면 정책의 높이 분포 유지 |
| H/O/T GTA pose | 모든 위치에서 **env_origin만** 뺀 미터 단위 좌표; 객체와 무대 위 humanoid 사이 실제 높이차 유지 |
| 센서 XY | heading-local grid → yaw 회전 → root world XY 더하기 |
| 지형 높이·낙상 높이 | world XY로 해당 환경의 footprint 조회; world 신체 높이에서 해당 지형 높이 빼기 |
| 객체 인덱스 | 물리 actor 순서와 graph의 논리 순서를 `_logical_box_order`로 연결 |

### 학습·로컬 추론·서버 VNC

새 스크립트는 이 서버에서 CUDA 사용이 확인된 `tokenhsi_sm120`을 기본으로 한다. `TOKENHSI_CONDA_ENV`로 바꿀 수 있다. 실행 전 GPU 점유를 확인한다.

```bash
STAGE1='output/approach_scenario_stage1_self_sum_distill/ApproachScenarioStage1SelfSumDistill_28-17-01-48/nn/ApproachScenarioStage1SelfSumDistill.pth'
TOKENHSI_GPU=0 STAGE1_CHECKPOINT="$STAGE1" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_cliff_sensor_train.sh 2 2048 3

# 짧은 학습 검증: 환경 수는 유지하고 반복 횟수만 제한
TOKENHSI_GPU=0 STAGE1_CHECKPOINT="$STAGE1" MAX_ITERATIONS=1 \
OUTPUT_PATH=output/approach_stage2_cliff_sensor_check \
  bash tokenhsi/scripts/multi_agent/approach_stage2_cliff_sensor_train.sh 2 2048 3

# 같은 센서 실험 checkpoint의 학습 재개
TOKENHSI_GPU=0 RESUME_CHECKPOINT="$CKPT" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_cliff_sensor_train.sh 2 2048 3

# CKPT에는 새 센서 실험에서 저장한 checkpoint를 지정
TOKENHSI_GPU=0 HEADLESS=0 SENSOR_VIS=1 TASK_GRAPH=random_scenario \
  bash tokenhsi/scripts/multi_agent/approach_stage2_cliff_sensor_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU=0 HEADLESS=1 EPISODE_LENGTH=64 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_cliff_sensor_test.sh "$CKPT" 2 16 3 1
TOKENHSI_GPU=0 SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_cliff_sensor_vnc.sh "$CKPT" 2 1 3 10
```

평가 `TASK_GRAPH`는 `random_scenario`, `holding`, `sit`, `climb`을 지원한다. VNC의 기존 서비스 재사용·포트 규칙은 아래 공통 설명을 따른다. `SENSOR_VIS=1`은 viewer의 0번 환경에 두 명의 높이 샘플을 표시한다.

TensorBoard `cliff/`와 평가 `metrics/cliff.json`에 과제별·agent별 낙상, **동료 낙상으로 인한 중단**, 센서의 단차 관찰 step, 똑바로 선 상태에서 가장자리 근처 정지·회전 step, 평균 episode 길이를 기록한다. Timeout은 동료 낙상 중단 횟수에 포함하지 않는다. 정지/회전 step은 행동 진단이며 과제 성공이나 센서에 대한 인과적 반응을 입증하는 지표는 아니다. 안전한 경로가 없으므로 정지·회전의 습득은 장기 학습으로 따로 평가한다.

### 학습 체크포인트의 센서 제거 평가

`SENSOR_MASK=1`은 **평가에서만** actor·critic 센서 인코더가 낸 64차원 특징을 action/value head 직전에 0으로 마스킹한다. 물리 지형, 보상·종료, 실제 센서 진단값과 checkpoint 가중치는 유지한다. `SENSOR_MASK`를 지정하지 않거나 0이면 기존 평가다. 마스킹 평가의 기본 출력은 별도 `output/approach_stage2_cliff_sensor_masked`이며 `metrics/cliff.json`과 `edge_context_*.json`에 `sensor_masked`가 기록된다. 센서 선의 viewer 표시 여부는 `SENSOR_VIS`로 별도 제어한다.

```bash
CKPT='output/approach_stage2_cliff_sensor/ApproachStage2CliffSensor_29-15-31-44/nn/ApproachStage2CliffSensor_00000500.pth'

# 같은 checkpoint·시드·환경 수·평가 횟수로 비교
TOKENHSI_GPU=0 HEADLESS=1 SEED=29 SENSOR_MASK=0 \
OUTPUT_PATH=output/approach_stage2_cliff_sensor_ablation_normal \
  bash tokenhsi/scripts/multi_agent/approach_stage2_cliff_sensor_test.sh "$CKPT" 2 16 3 10
TOKENHSI_GPU=0 HEADLESS=1 SEED=29 SENSOR_MASK=1 \
OUTPUT_PATH=output/approach_stage2_cliff_sensor_ablation_masked \
  bash tokenhsi/scripts/multi_agent/approach_stage2_cliff_sensor_test.sh "$CKPT" 2 16 3 10

# noVNC에서 마스킹 정책을 시각화
TOKENHSI_GPU=0 SENSOR_MASK=1 SEED=29 SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_cliff_sensor_vnc.sh "$CKPT" 2 1 3 1
```

두 평가의 `metrics/cliff.json`에서 실패·생존 길이·가장자리 행동을 비교한다. 마스킹은 학습된 센서 특징의 필요성을 시험하지만, 특징 0은 학습 중 보지 않은 입력일 수 있으므로 차이만으로 센서 인과성을 완전히 입증하지 않는다.
학습 중 갱신되는 비번호 checkpoint 대신 같은 번호 checkpoint 또는 미리 복사한 고정 파일을 두 평가에 사용한다.

### 소수 환경으로 학습하며 보기

`watch`는 실제 PPO/AMP 업데이트를 수행한다. 에이전트 2명·객체 3개, 기본 4환경이며 환경 수는 1 이상으로 지정한다. 전용 학습 config의 PPO/AMP minibatch는 64, AMP buffer는 16384다. 본학습 config의 2048환경 설정은 그대로다. 소수 환경 실행은 배치 분포가 좁으므로 시각적 동작 확인에 사용한다.

```bash
STAGE1='output/approach_scenario_stage1_self_sum_distill/ApproachScenarioStage1SelfSumDistill_28-17-01-48/nn/ApproachScenarioStage1SelfSumDistill.pth'

# 서버 noVNC: 실제 학습 + 센서 표시
TOKENHSI_GPU=0 STAGE1_CHECKPOINT="$STAGE1" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_cliff_sensor_watch_vnc.sh 2 4 3

# 한 환경만 보기: 두 사람과 세 객체가 있는 무대 하나
TOKENHSI_GPU=0 STAGE1_CHECKPOINT="$STAGE1" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_cliff_sensor_watch_vnc.sh 2 1 3

# 로컬 viewer
TOKENHSI_GPU=0 STAGE1_CHECKPOINT="$STAGE1" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_cliff_sensor_watch.sh 2 4 3

# 센서 Stage 2 checkpoint를 이어 학습하며 보기
TOKENHSI_GPU=0 RESUME_CHECKPOINT="$CKPT" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_cliff_sensor_watch_vnc.sh 2 4 3
```

기본 출력은 `output/approach_stage2_cliff_sensor_watch`다. `OUTPUT_PATH`로 분리하고 `MAX_ITERATIONS`로 반복 수를 제한할 수 있다. `SENSOR_VIS=1`이 기본이며 0번 환경의 agent별 높이 샘플을 하늘색/주황색으로 표시한다. `SENSOR_VIS=0`으로 숨긴다. Rollout 사이 PPO 업데이트 중에는 화면이 잠시 멈출 수 있다.

이 서버의 GPU 0에서는 기존 서비스의 **5802** 포트를 재사용한다. VS Code에서 5802를 포워딩하고 `http://localhost:5802/`에 접속한다. 저장소의 독립 화면 기본값 5803/5903과는 별도이며, watch는 기존 서비스를 재시작하지 않는다. 종료는 실행 터미널의 `Ctrl+C`로 한다.

### 좌표계·물리 검증 재실행

```bash
export TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_sm120
source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python -c 'import isaacgym; import pytest; raise SystemExit(pytest.main(["-q", "tokenhsi/tests/test_cliff_sensor.py", "tokenhsi/tests/test_cliff_coordinates.py"]))'
PYTHONPATH=tokenhsi python tokenhsi/tests/cliff_physics_smoke.py \
  --task HumanoidMACliff --headless --no_video --test --eval \
  --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_cliff_sensor.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_cliff_sensor.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --num_agents 2 --num_objects 3 --num_envs 16 \
  --output_path output/approach_stage2_cliff_sensor_physics_check
```

## Stage 2 부채꼴 비전 센서

복사해서 실행할 본학습·watch·테스트·resume 명령은 [부채꼴 센서 실행 명령 모음](fan_sensor_commands.md)에 정리했다.

상세 계약·가림 처리·향후 carryWith 제한은 [fan_sensor.md](fan_sensor.md)를 따른다. 기존 정사각형 실험과 별도이며, 같은 Stage 1에서 새로 전이한다. 센서 **576차원/agent**, 전체 관측 **1759차원**이다. 머리에서 전방 **100도·10m**를 보며, 24개 고리에 1·3·…·47개를 배치해 면적 밀도를 균일하게 한다. 가림 뒤 높이는 다음 고리의 가장 가까운 각도를 연결한 경로에서 채운다. 현재 무대·지면·회전된 상자의 표면과 가림을 반영한다. 미관측은 정규화 후 `2.0`이다.

**현재 낙하 실험에 한해 상자를 포함한다. 향후 carryWith에서는 운반 상자 높이가 학습을 방해할 수 있으므로 상자 제외 설계로 변경해야 한다.** 높이 관측과 가림 판정에서의 제외 범위를 별도로 정해야 하며, 현재 상자 제외 옵션은 없다.

이전 정사각형/1024개 부채꼴 checkpoint와 20m 부채꼴 checkpoint는 관측 계약이 달라 로드하지 않는다. 현재는 metadata version 3을 사용한다. 기존 sensor의 보상·종료·Stage 1 동결 및 네트워크 구조를 유지하며 PPO 설정은 `amp_ma_cliff_sensor.yaml`을 공유한다. 아래 명령은 이 서버의 CUDA 환경 `tokenhsi_sm120`을 기본으로 사용한다.

```bash
STAGE1='output/approach_scenario_stage1_self_sum_distill/ApproachScenarioStage1SelfSumDistill_28-17-01-48/nn/ApproachScenarioStage1SelfSumDistill.pth'
# 2048환경 본학습: 이전 smoke/resume 환경변수 해제
unset MAX_ITERATIONS RESUME_CHECKPOINT OUTPUT_PATH
TOKENHSI_GPU=0 STAGE1_CHECKPOINT="$STAGE1" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_train.sh 2 2048 3

# 짧은 학습 검증
TOKENHSI_GPU=0 STAGE1_CHECKPOINT="$STAGE1" MAX_ITERATIONS=1 \
OUTPUT_PATH=output/approach_stage2_fan_sensor_576_10m_check/train \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_train.sh 2 2048 3

# CKPT는 현재 576개/100도/10m 부채꼴 실험의 checkpoint
TOKENHSI_GPU=0 RESUME_CHECKPOINT="$CKPT" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_train.sh 2 2048 3
TOKENHSI_GPU=0 HEADLESS=0 SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU=0 HEADLESS=1 EPISODE_LENGTH=64 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_test.sh "$CKPT" 2 16 3 1
TOKENHSI_GPU=0 SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_vnc.sh "$CKPT" 2 1 3 10
```

본학습 기본 출력은 `output/approach_stage2_fan_sensor_576_100_10m`이며, `max_epochs=1000000`, checkpoint 저장 주기는 500 epoch다. 고리 간격은 약 0.417m, 접선 방향 간격은 약 0.364m다.

센서 표시: 하늘색=직접 관측, 노랑=뒤쪽 관측 높이로 추정, 자홍색=미관측. 미관측 십자는 지형 높이를 뜻하지 않는 머리 높이 위의 표시다. `SENSOR_MASK=1` 평가도 지원한다. `sensor_edge_steps`는 미관측 특수값을 제외한 높이 차이며, 상자 단차도 포함하므로 절벽만의 지표로 해석하지 않는다.

### 부채꼴 센서의 소수 환경 학습 시각화

`approach_stage2_fan_sensor_watch.sh`는 부채꼴 설정으로 **실제 PPO/AMP 학습과 viewer를 함께 실행**한다. 기본 4환경, 최소 1환경이며 기존 `amp_ma_cliff_sensor_watch.yaml`의 작은 배치 설정을 공유한다. 센서는 기본 표시된다. 로컬 viewer는 `watch.sh`, 서버 noVNC는 `watch_vnc.sh`를 사용한다.

```bash
STAGE1='output/approach_scenario_stage1_self_sum_distill/ApproachScenarioStage1SelfSumDistill_28-17-01-48/nn/ApproachScenarioStage1SelfSumDistill.pth'

# noVNC: 1환경에서 새 전이 학습과 센서 보기
TOKENHSI_GPU=0 STAGE1_CHECKPOINT="$STAGE1" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_watch_vnc.sh 2 1 3

# noVNC: 4환경 학습
TOKENHSI_GPU=0 STAGE1_CHECKPOINT="$STAGE1" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_watch_vnc.sh 2 4 3

# 로컬 viewer 학습
TOKENHSI_GPU=0 STAGE1_CHECKPOINT="$STAGE1" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_watch.sh 2 1 3

# 같은 부채꼴 센서 checkpoint에서 학습 재개
TOKENHSI_GPU=0 RESUME_CHECKPOINT="$CKPT" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_watch_vnc.sh 2 1 3
```

기본 출력은 `output/approach_stage2_fan_sensor_576_100_10m_watch`다. `OUTPUT_PATH`로 바꾸고, 짧은 확인에는 `MAX_ITERATIONS`를 지정한다. `RESUME_CHECKPOINT`가 설정되어 있으면 Stage 1 새 전이보다 우선하므로 새 전이 시 해당 변수를 해제한다. 이 서버에서는 5802 포트를 포워딩하고 `http://localhost:5802/`를 연다. PPO 업데이트 중에는 화면이 잠깐 멈출 수 있다. 종료는 실행 터미널에서 `Ctrl+C`를 누른다. 소수 환경의 학습 진행 확인이며 본학습 2048환경과 동등한 학습 결과를 보장하지 않는다.

### 통제된 가림 장면 촬영

정책과 독립된 가림 장면의 실제 viewer 스크린샷은 다음 명령으로 생성한다. 기존 GUI 서비스를 유지하며 출력 PNG/NPZ/JSON은 별도 디렉터리에 쓴다.

```bash
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_sm120 \
  bash tokenhsi/scripts/multi_agent/run-gui.sh bash -c '
    source tokenhsi/scripts/multi_agent/runtime_env.sh
    PYTHONPATH=tokenhsi python tokenhsi/tests/fan_sensor_capture.py \
      --task HumanoidMACliff --no_video --test --eval \
      --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_fan_sensor.yaml \
      --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_cliff_sensor.yaml \
      --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
      --num_agents 2 --num_objects 3 --num_envs 1 \
      --output_path output/approach_stage2_fan_sensor_576_10m_check/screenshots'
```

## Stage 2 두 무대·연결다리

학습·watch·테스트 시각화·resume 명령은 [실행 명령 모음](fan_sensor_commands.md#두-무대연결다리-실험-1024개가림-없음)에 있다. `HumanoidMABridge`(`humanoid_ma_bridge.py`)가 부채꼴 센서 task를 확장하며, Stage 1 schema-9 checkpoint에서 새로 전이한다. 에이전트 간 `collision`은 아래 CPA 식을 사용한다.

연결다리의 접촉 마찰계수는 `BRIDGE_FRICTION=0.3`처럼 지정한다(기본 `1.0`, 유한한 0 이상 값). 다른 actor의 마찰은 유지하며, 아래 fan/square 혼합 실험에도 적용된다.

- **장면:** reset마다 50%는 기존 단일 무대(물체는 무대 밖 지면), 50%는 두 무대 장면이다. 무대 B(4–5m, `far_size_range`)는 A의 +x쪽 간격 건너편에 놓이며 y 오프셋은 무작위다. 과제 상자 3개는 무대 B 위, 간격 쪽 변에서 1–2m 띠(`object_band`)의 3개 차선에 놓인다. 무대 A 크기는 단일 무대 장면과 공유되므로 줄이지 않는다. 대신 일반 시작 위치를 A의 간격 쪽 변에서 0.8–3.0m(`spawn_depth`), y는 두 무대가 겹치는 구간으로 제한한다. 이렇게 하면 에피소드 시작 시 에이전트–자기 상자 거리가 p99 7.7m로, Stage 1 시작 거리(p99 7.8m) 범위에 든다.
- **목표(T 노드):** HOLDING_AT·ON_TOP을 샘플하지 않으므로 목표는 보상에 쓰이지 않고 관측에만 들어간다. 단일 무대 장면에서는 기존처럼 env 원점(무대 A 중앙의 바닥 높이)에 둔다. 두 무대 장면에서는 두 에이전트의 목표를 모두 무대 B 윗면, 상자 띠 중앙(간격 쪽 변에서 1.5m)의 한 점에 둔다. Stage 1도 비활성 목표를 한 점에 모았다.
- **다리:** 간격 1.5–2.5m와 폭 1.0–2.0m(0.1m 단위)는 env마다 생성 시 고정한다. 다리 길이는 간격 − 2×1cm이고, 윗면은 무대보다 1cm 낮다. 다리는 중력 없는 동적 상자(밀도 500, 회전 금지)이며, `apply_rigid_body_force_tensors`의 PD 힘(2Hz, 최대 6m/s²)으로 quintic 궤적을 따라 채널로 들어온다. 정지 위치는 두 무대가 겹치는 변 구간에서 무작위로 정한다.
- **도착 시각:** 20%는 reset 때부터 연결된 상태다. 나머지는 도착 시각을 1–18초 로그균등(`arrival_time_range`)으로 먼저 정한다. 2초 이내 약 24%, 중앙값 약 4.2초, 8초 이상 약 28%다. 초기 학습의 짧은 에피소드에서도 연결 순간을 겪고, 긴 대기도 충분히 겪게 하려는 분포다.
  - 다리는 도착 시각에 맞춘 평균 속도(0.3–1.5m/s)로 출발점에서 계속 움직인다(`bridge_schedule`).
  - 최저 속도보다 느려야 하면 출발점에서 기다린 뒤 출발한다.
  - 최고 속도로도 늦으면 경로 중간에서 그 지점의 속도로 이미 움직이는 상태로 시작한다.
  - 이전 설정(0–6초 대기 후 이동)에서는 에피소드가 1–2초에 끝나, 다리가 도착한 접근 에피소드가 0개였다. 그래서 바꿨다.
- **무대 B와 추가 actor:** 무대 B는 reset마다 옮겨야 하므로 중력 없는 고밀도(1e7) 동적 body로 만든다. 단일 무대 장면에서는 무대 B와 다리를 z=20 위에 띄우고 센서·지형 조회에서 제외한다. 추가 actor의 root state는 carry의 단일 indexed 호출(`_extra_reset_actor_ids`)에 합친다. 따로 호출하면 앞선 상자 배치가 무효가 된다.
- **센서:** 머리 기준 전방 100도·10m, 32고리 **1024개**이며 `sensor_occlusion: none`이다. 가림 판정 없이 무대 A·B, 다리, 과제 상자의 yaw 발자국과 최상단 모서리 높이를 수직 조회한다(`vertical_height`). 관측은 2655차원, sensor metadata는 version 4다. 576개 부채꼴, 정사각형 센서, Stage 1 checkpoint는 resume·평가가 거부된다.
- **RSI:** 동작은 loco 100%다. 두 무대 장면에서 두 에이전트는 무대 A에서 시작한다. 에이전트마다 near-start 확률로 다리 착지점 근처(A 가장자리에서 0.8–2.0m, 접선 ±1m)에 두고, 나머지는 A 내부에 균일하게 둔다. near-start 확률은 0.5에서 0.2로 30만 step에 걸쳐 줄고, 평가에서는 0.2다. heading은 목표 방향에 jitter를 더한다.
- **종료:** 지면 높이 조회에 무대 B와 다리를 포함한다. 다리 위는 무대 위와 같이 취급하고, 간격 바닥에 닿으면 기존 바닥 접촉 실패로 종료한다.
- **충돌 패널티:** 이 task의 단일 무대·다리 장면 모두에서 기존 root XY 거리 패널티를 CPA로 교체했다. 현재 상대 XY 위치·속도로 최근접 시각과 거리를 예측하고, 연속 접근 정도 × 예상 거리 위험도 × `agentCollisionTTCDiscount ** (t_cpa / dt)`에 음수 계수를 곱한다. 계수는 0.5, 예상 안전거리는 0.7m, 시간 할인율은 0.99다. 0.7m 안에서도 서로 멀어지거나 상대속도가 0이면 패널티가 0이다. Z·신체 contact는 쓰지 않으며, 바닥 접촉 실패와 별개다. TensorBoard `reward_terms/collision`과 CSV의 기존 `collision` 항에 기록한다.
- **출력·checkpoint:** 기본 학습·평가 output은 `output/approach_stage2_fan_bridge_cpa`, watch는 `_cpa_watch`, 센서 마스킹 평가는 `_cpa_masked`다. 관측·모델·sensor metadata는 같아 기존 다리 checkpoint도 로드할 수 있지만, 새 실행에서는 CPA 보상을 쓴다. 이전 가중치는 CPA로 학습된 정책이 아니며, 실행 중인 프로세스에는 소급 적용되지 않는다.
- **진단·확인:** 진단은 TensorBoard `cliff/bridge/*`와 평가 `cliff.json`에 남는다. `BRIDGE_PROBABILITY=1/0`으로 장면을 고정할 수 있으며, metadata는 바뀌지 않아 시각 확인용으로 쓴다.

```bash
export TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_sm120
source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python -c 'import isaacgym; import pytest; raise SystemExit(pytest.main(["-q", "tokenhsi/tests/test_bridge_collision.py", "tokenhsi/tests/test_bridge_scenario.py"]))'
PYTHONPATH=tokenhsi python tokenhsi/tests/bridge_physics_smoke.py \
  --task HumanoidMABridge --headless --no_video --test --eval \
  --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_fan_bridge.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_cliff_sensor.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --num_agents 2 --num_objects 3 --num_envs 16 \
  --output_path output/approach_stage2_fan_bridge_physics_check
```

## Stage 2 room · 4면 벽

현재 요청한 내부 칸막이·HOLDING_AT 실험은 아래의 **room partition**이다. 이 절의 room은 칸막이 없는 비교용 초기 구성이다.

[실제 Isaac Gym 화면·검증 보고서](room_scenario_report_20260930.md). `HumanoidMARoom`은 기존 센서 경로를 사용하며, 무대·다리와 별도 room 전용 실험이다.

- 내부 가로·세로 각각 6–8m(0.5m 단위, 환경 생성 시 고정), 바닥 0.1m, 바닥 위 벽 높이 2.5m·두께 0.4m, 천장 없음. reset마다 두 사람과 작업 객체 3개를 모두 실내에 겹치지 않게 배치한다. 비활성 목표 2개도 실내 바닥 중앙에 둔다.
- 관측은 머리 기준 100도·10m·1024개 가림 없는 높이 센서와 기존 scene 관측, 총 2655차원이다. 네 벽은 물리 actor와 센서 표면에 포함하며 graph 토큰에는 넣지 않는다.
- `collision = -0.5 × max(사람 CPA 위험도, 벽 CPA 위험도)`. 벽마다 안쪽 면 선분에서 root와 가장 가까운 점을 속도 0의 가상 에이전트로 두고 사람 간 CPA 함수(`compute_agent_cpa_collision_penalty`)를 그대로 적용한다. 0.7m·계수는 공유하고 시간 할인은 벽 전용 `wallCollisionTTCDiscount: 0.95`(사람 간 0.99)를 사용한다. 추가 보상 항은 없다. 이탈·평행 이동·정지하거나 최근접점에서 0.7m 넘게 비껴가는 방향이면 그 벽의 위험도는 0이다. 이전의 선분 궤적 CPA(할인 0.99)는 닫힌 방에서 어느 방향 보행이든 벽 위험도가 생겨 정지를 유도했으므로 2026-10-06에 교체했다.
- 벽 접촉 자체로 종료하지 않는다. 발밑 지형은 방 바닥이며, 벽 상단을 낙상 판정의 지형으로 사용하지 않는다. 넘어짐과 시간 제한 종료는 유지한다.
- HOLDING/SIT/CLIMB 균등, loco RSI, Stage 1 actor encoder·관측 RMS 동결은 다리 실험과 같다. 별도 학습 config·네트워크 변경은 없다.
- sensor metadata version 5와 `room` 형상으로 checkpoint를 구분한다. 기존 bridge/cliff checkpoint는 직접 resume·평가할 수 없다. schema-9 Stage 1에서 새로 전이한다. 기본 output은 `output/approach_stage2_fan_room_cpa_wall`, 센서 마스킹 평가는 `_cpa_wall_masked`다.

학습(2048환경):

```bash
GPU=0
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT=output/stage1/28-17-01-48/nn/stage1.pth \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_room_train.sh 2 2048 3
```

짧은 확인은 `MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_stage2_fan_room_cpa_check/train`을 위 명령에 추가한다. 동일 room 실험의 resume은 `STAGE1_CHECKPOINT` 대신 `RESUME_CHECKPOINT=/absolute/path/to/ApproachStage2FanRoom.pth`를 지정한다.

로컬 추론·화면 없는 평가·서버 VNC(이번 smoke checkpoint를 지정한 예시):

```bash
CKPT=output/approach_stage2_fan_room_cpa_check/train/ApproachStage2FanRoom_30-14-02-55/nn/ApproachStage2FanRoom.pth
TOKENHSI_GPU="$GPU" HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_stage2_fan_room_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" HEADLESS=1 bash tokenhsi/scripts/multi_agent/approach_stage2_fan_room_test.sh "$CKPT" 2 16 3 3
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_stage2_fan_room_vnc.sh "$CKPT" 2 1 3 10
```

실제 방 초기 배치·비전센서 viewer 캡처 재현(정책 checkpoint 불필요):

```bash
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_sm120 \
  bash tokenhsi/scripts/multi_agent/run-gui.sh bash -c '
    source tokenhsi/scripts/multi_agent/runtime_env.sh
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    PYTHONPATH=tokenhsi python tokenhsi/tests/room_capture.py \
      --task HumanoidMARoom --no_video --test --eval \
      --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_fan_room.yaml \
      --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_cliff_sensor.yaml \
      --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
      --num_agents 2 --num_objects 3 --num_envs 1 \
      --output_path output/approach_stage2_fan_room_cpa_check/screenshots'
```

CPU 검사는 `test_room_scenario.py`; 물리 검사는 같은 task/config에 `room_physics_smoke.py --headless`를 사용하고 `--num_envs 64`, `--output_path output/approach_stage2_fan_room_cpa_check/physics`로 실행한다. GPU 실행은 이 서버에서 `tokenhsi_sm120`을 사용한다.

## Stage 2 세 시나리오 혼합 · 단일 무대·연결다리·room

현재 통합 실험은 `HumanoidMAMixedSensor`와 `approach_stage2_fan_mixed.yaml`이다. 기존 개별 실험은 유지한다.

**[학습·테스트·학습 시각화·테스트 시각화 명령만 보기](mixed_sensor_commands.md)**

- **학습과 테스트 모두** 환경 reset마다 `scenarioMixture`의 단일 무대/연결다리/room 확률 **각 1/3**로 다시 선택한다. 두 에이전트는 같은 장면을 공유한다. 장면과 별도로 agent별 HOLDING/SIT/CLIMB/HOLDING_AT **각 25%**, ON_TOP 0%다. 시나리오 확률 자체를 매번 무작위로 바꾸지는 않는다.
- **단일 무대:** 기존 4–8m·높이 1.5m 무대와 바깥 지면의 상자 3개를 유지한다. AT 목적지는 무대 A 위다. 과제를 완료하지 못하더라도 낙하하지 않도록 배우는 의도된 장면이다. 성공률과 별도로 낙하율·에피소드 생존 길이를 확인한다.
- **연결다리:** 사람은 A, 상자는 B에서 시작하며 AT는 B의 상자를 A의 목적지로 운반한다. 간격·다리 폭·도착 일정·near-start는 기존 연결다리 실험과 같다. 혼합 실험에서는 `BRIDGE_PROBABILITY` 대신 `scenarioMixture`가 장면을 결정한다.
- **Room:** 6–8m 내부·바닥 0.1m·높이 2.5m 벽·중앙 칸막이·양 끝 1.8m 통로를 유지한다. **HOLDING/HOLDING_AT은 agent별 50% 확률로 자기 상자를 칸막이 반대편에서 집으러 가도록** 시작한다(`room.partition.pickup_cross_probability: 0.5`). 나머지 50%와 SIT/CLIMB은 자기 상자와 같은 쪽에서 시작한다. 반대편 시작은 사람–상자의 직선 경로가 칸막이를 지나도록 제한하며, AT 목적지는 항상 상자의 반대편이다. 목표는 실제 graph goal slot과 운반 상자 높이를 따른다.
- **센서·정책:** 1024개·100도·10m·가림 없음, 관측 2655차원, graph 노드 7개를 유지한다. 시나리오 ID·벽/다리 토큰을 정책에 추가하지 않는다. 학습·평가 모두 loco 모션/시점 랜덤 RSI 100%다.
- **물리 전환:** 단일 무대·연결다리·room을 Y 방향으로 32m씩 떨어진 고정 공간에 생성한다(`sceneSpacing: 32`, `envSpacing: 48`). 무대 3개와 room 바닥·벽 6개는 실제 고정 actor다. reset마다 사람·상자·목표·움직이는 다리만 선택된 공간으로 배치한다. 관측 원점은 활성 공간을 따르고 `observationArenaScale: 16`을 유지하므로 공간 간 이동량이 정책 입력에 섞이지 않는다. 비활성 지형은 센서·종료·벽 CPA 계산에서 제외한다. 두 무대의 횡방향 상대 위치는 환경 생성 때 랜덤으로 정해 고정하며, 사람·상자·다리의 출발/도착 위치와 일정은 reset마다 계속 샘플링한다.
- **RSI 발 배치:** 랜덤 모션/시점을 유지하면서 현재 발 메시의 회전·최저점을 계산해, 바닥과 겹치는 경우 몸 전체를 최소한으로 올린다. 최소 발 여유는 5mm이며 root와 관측용 신체 위치를 함께 보정한다. 자세·관절 상태·RSI 속도는 유지한다. 별도의 보상 항을 추가하지 않는다.
- **보상·종료:** 기존 과제/AMP 보상을 유지하고, collision은 사람 CPA와 활성 room 벽 CPA의 최댓값에 `-0.5`를 곱한 항 하나다. 벽 CPA는 room 절과 같이 벽 최근접점을 정지 가상 에이전트로 둔 사람 CPA이며 시간 할인 0.95다. 단일 무대 밖/다리 틈 바닥 접촉은 실패, room 벽 접촉만으로는 종료하지 않는다. 넘어짐·600-step 시간 제한은 유지한다.
- **새 pickup 배치와 checkpoint:** `pickup_cross_probability: 0.5`가 room metadata에 저장된다. 이 필드가 없는 이전 혼합/partition checkpoint는 현재 config의 resume·평가 검사에서 거부된다. 기존 Stage 1에서 전이해 새로 학습하며, 새 설정에서 생성한 checkpoint끼리 재개·평가한다. 관측 차원·센서 계산·보상은 같다.
- **전이:** sensor metadata version **7**. Stage 1 schema-9에서 새로 전이하고 actor encoder·관측 RMS를 동결한다. 이전 개별 센서 실험 checkpoint는 직접 resume·평가하지 않는다. 같은 혼합 실험 checkpoint만 resume한다.
- **물리 수정과 checkpoint:** 같은 pickup 설정의 기존 혼합 checkpoint는 현재 코드에서 평가·재개할 수 있다. 센서/관측 계약은 그대로이며 새 checkpoint에 `physics_metadata.version: 1`을 저장한다. 기존 checkpoint도 새 물리 환경에서 실행되므로 수정 전 결과와 물리 조건이 다르다. 새 출력은 `_cpa_static`으로 분리한다. 실행 중인 학습·viewer에는 코드 변경이 자동 반영되지 않는다.
- **통계:** TensorBoard의 `cliff/scenario/{cliff,bridge,room}/`에 환경 step·종료·낙상·과제별 샘플/성공 step을 기록한다. 평가 JSON의 `scenarios`에는 실제 평가 시작 장면별 trial 수·종료율·과제별 최종 성공률을 기록한다. 완료 후 재시작된 장면을 해당 trial의 장면으로 잘못 집계하지 않는다. 선택 확률 1/3은 에피소드 기준이며, 생존 시간에 따라 PPO step 비중은 달라진다.

### 학습·평가·VNC 실행

저장소 루트에서 GPU 점유를 확인하고 `GPU`를 지정한다. 새 본학습은 기존 출력과 분리된 `output/approach_stage2_fan_mixed_cpa_static_wall`에 저장된다(벽 CPA 교체 전 결과는 `_cpa_static`). 전이 기본 Stage 1은 `output/stage1/8000.pth`다(2026-10-06부터, 이전 `output/stage1/28-17-01-48/nn/stage1.pth`).

```bash
GPU=0
STAGE1=output/stage1/8000.pth  # train/watch 기본값이라 STAGE1_CHECKPOINT는 생략 가능
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_mixed_train.sh 2 2048 3

# 실제 학습으로 생성된 혼합 실험 checkpoint 경로를 지정한다.
CKPT="output/approach_stage2_fan_mixed_cpa_static_wall/<run>/nn/ApproachStage2FanMixed.pth"

# 로컬 viewer: 매 에피소드마다 세 장면과 네 과제 랜덤 선택
TOKENHSI_GPU="$GPU" HEADLESS=0 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_mixed_test.sh "$CKPT" 2 1 3 10

# 화면 없는 랜덤 평가
TOKENHSI_GPU="$GPU" HEADLESS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_mixed_test.sh "$CKPT" 2 96 3 3

# 서버 noVNC: 동일한 랜덤 평가 경로
TOKENHSI_GPU="$GPU" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_mixed_vnc.sh "$CKPT" 2 1 3 10

# 같은 혼합 실험 재개
TOKENHSI_GPU="$GPU" RESUME_CHECKPOINT="$CKPT" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_mixed_train.sh 2 2048 3
```

학습 시각화는 `amp_ma_cliff_sensor_watch.yaml`을 사용하며 기본 **4환경**, 환경 수는 1 이상이다. PPO/AMP minibatch 64·AMP buffer 16384로 소수 환경 rollout에 맞춘다. 네트워크·환경·센서·보상은 동일하며, 본학습은 기존 2048환경 설정을 사용한다. 시각화 출력은 `output/approach_stage2_fan_mixed_cpa_static_wall_watch`다. `SENSOR_VIS=1`로 첫 환경의 비전센서를 표시한다. 서버 VNC는 같은 GPU 설정을 내부 학습에 전달한다.

```bash
# 로컬 학습 viewer
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_mixed_watch.sh 2 4 3

# 서버 VNC 학습 viewer
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_mixed_watch_vnc.sh 2 4 3
```

`TASK_GRAPH` 기본값은 `random_scenario`다. 이전 실험에서 `TASK_GRAPH=holding_at` 등을 export해 두었다면 해제해야 네 과제가 랜덤하게 나온다. `SENSOR_MASK=1`은 센서 마스킹 평가이고, `EPISODE_LENGTH`는 평가 시간 제한을 바꾸는 선택 옵션이다. `SCENARIO=cliff|bridge|room`은 평가(test/VNC)에서만 한 장면으로 고정한다. sensor metadata의 혼합 비율은 그대로라 기존 checkpoint를 쓸 수 있고, 학습·watch에서 지정하면 에러로 종료한다.

짧은 학습 확인은 본학습 명령에 `MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_stage2_fan_mixed_static_check/train`을 추가한다. CPU 검사는 `tokenhsi/tests/test_mixed_sensor.py`, 실제 물리 검사는 다음 명령이다.

```bash
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_sm120 bash -c '
  source tokenhsi/scripts/multi_agent/runtime_env.sh
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  PYTHONPATH=tokenhsi python tokenhsi/tests/mixed_sensor_physics_smoke.py \
    --task HumanoidMAMixedSensor --headless --no_video --test --eval \
    --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_fan_mixed.yaml \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_cliff_sensor.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --num_agents 2 --num_objects 3 --num_envs 96 \
    --output_path output/approach_stage2_fan_mixed_static_check/physics'
```

초기 혼합 구현 검증(2026-09-30, 픽업 반대편 배치 추가 전): CPU **49개 통과**, PhysX **96환경×24 reset**에서 9종 장면 전환 모두 발생, 활성 AT 목표 **1,163건**의 위치·높이·binding 확인. room→단일 무대→다리→room의 물리 전환에서 지지 상자 각 **96/96**, 두 room 접촉 검사 각각 **96/96**, 지형 최대 이동 **0.057mm**를 확인했다. 2048환경·2 epoch 학습/저장, actor encoder 75개·관측 RMS 6개 불변, scalar 256종 유한값, sensor head 갱신, 이전 partition checkpoint 거부를 확인했다.

저장 checkpoint의 **96환경×3회·최대 64-step** 랜덤 평가를 완료했다. 실제 시작 장면은 단일 무대 **81**, 연결다리 **108**, room **99**였고, 각 장면에 네 과제가 모두 나왔다. 표본에서 정확한 1/3 또는 25% 개수를 강제하지 않는다. 이번 짧은 평가의 AT 성공은 0이며, 장기 낙하 회피·대기·우회 운반 성능은 본학습으로 확인해야 한다.

검증 산출물: [CPU 로그](../output/approach_stage2_fan_mixed_cpa_check/cpu.log), [물리 검사](../output/approach_stage2_fan_mixed_cpa_check/physics/physics.json), [checkpoint 검사](../output/approach_stage2_fan_mixed_cpa_check/checkpoint_audit.json), [학습 로그](../output/approach_stage2_fan_mixed_cpa_check/train.log), [랜덤 평가](../output/approach_stage2_fan_mixed_cpa_check/eval/metrics/edge_context_20260930_162253.json).

픽업 반대편 배치 추가 후 검증(2026-09-30): CPU **51개 통과**. PhysX 96환경×24 reset에서 room의 반대편 픽업은 HOLDING **189/359**, HOLDING_AT **221/425**건이며 모두 직선 경로가 칸막이에 막혔다. SIT/CLIMB은 반대편 0건이다. 활성 AT 목표 **1,204건**, 9종 장면 전환, 부분 reset·물리 접촉 검사도 통과했다. 2048환경·2 epoch 학습/저장 후 96환경·최대 64-step 평가에서 단일 무대/다리/room **31/33/32**건과 장면별 네 과제를 확인했다. actor encoder 75개·RMS 6개 불변, 모델·scalar 256종 유한값, 이전 배치 checkpoint 거부를 확인했다. 과제 성공은 0이며 우회 행동의 학습 성능은 검증하지 않았다.

픽업 변경 당시 증거: [CPU 로그](../output/approach_stage2_fan_mixed_pickup_check/cpu.log), [물리 검사](../output/approach_stage2_fan_mixed_pickup_check/mixed_physics/physics.json), [checkpoint 검사](../output/approach_stage2_fan_mixed_pickup_check/checkpoint_audit.json), [학습 로그](../output/approach_stage2_fan_mixed_pickup_check/train.log), [랜덤 평가](../output/approach_stage2_fan_mixed_pickup_check/eval/metrics/edge_context_20260930_164750.json).

### 고정 지형·RSI 튕김 수정 검증 (2026-09-30)

CPU **60개 통과**, PhysX **96환경×24 reset**에서 9종 장면 전환·부분 reset·지지·칸막이 접촉을 확인했다. 환경별 고정 actor 9개는 reset·접촉 전후 이동 **0m**였다. 별도 다리/작업 상자 위 사람 지지 검사는 각각 **32/32**, 다리의 하중 아래 최대 이동은 **1.1mm**였다.

reset 직후 **0.2초**를 검사했다. 튕김 기준은 root 상승 속도 >3m/s 또는 초기 대비 높이 증가 >0.4m다. RSI 자세 유지와 기존 혼합 500-epoch checkpoint를 각각 32환경×64회 검사했다.

| 검사 | 튕김 / reset 수 | 최대 상승 속도 | 최대 높이 증가 |
| --- | ---: | ---: | ---: |
| RSI 자세 유지 | 0 / 2048 | 0.545m/s | 4.85cm |
| 기존 checkpoint 정책 | 0 / 2048 | 0.386m/s | 3.19cm |

최소 초기 발 여유는 두 검사 모두 5mm였다. **2048환경·2 epoch** 학습과 저장 모델의 **96환경·최대 64-step** 표준 평가도 완료했다. 이는 reset 물리 및 실행 검증이며 장기 학습 성능이나 모든 접촉 상태의 안정성을 보장하지 않는다.

증거: [CPU](../output/approach_stage2_fan_mixed_static_check/cpu.log), [물리](../output/approach_stage2_fan_mixed_static_check/physics/physics.json), [RSI reset](../output/approach_stage2_fan_mixed_static_check/reset_hold/reset_physics.json), [정책 reset](../output/approach_stage2_fan_mixed_static_check/reset_policy/reset_physics.json), [다리/상자 지지](../output/approach_stage2_fan_mixed_static_check/support.json), [학습](../output/approach_stage2_fan_mixed_static_check/train.log), [checkpoint](../output/approach_stage2_fan_mixed_static_check/checkpoint.json), [평가](../output/approach_stage2_fan_mixed_static_check/eval/metrics/edge_context_20260930_190124.json). 실제 viewer: [단일 무대](../output/approach_stage2_fan_mixed_static_check/viewer/cliff.png), [연결다리](../output/approach_stage2_fan_mixed_static_check/viewer/bridge.png), [room](../output/approach_stage2_fan_mixed_static_check/viewer/room.png).

정책 reset 검사 재실행(`CKPT`는 검사할 혼합 checkpoint; `--checkpoint`를 빼면 RSI 자세 유지 검사):

```bash
CHECKPOINT="$CKPT" TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_sm120 bash -c '
  source tokenhsi/scripts/multi_agent/runtime_env.sh
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  PYTHONPATH=tokenhsi python tokenhsi/tests/mixed_reset_physics_smoke.py \
    --task HumanoidMAMixedSensor --headless --no_video --test --eval --seed 47 \
    --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_fan_mixed.yaml \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_cliff_sensor.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --num_agents 2 --num_objects 3 --num_envs 32 --checkpoint "$CHECKPOINT" \
    --output_path output/approach_stage2_fan_mixed_static_check/reset_policy'
```

수정 후 실행 확인용 checkpoint로 바로 viewer를 열려면(2 epoch 모델이므로 학습 성능 확인용은 아님):

```bash
CKPT=output/approach_stage2_fan_mixed_static_check/train/ApproachStage2FanMixed_30-18-57-57/nn/ApproachStage2FanMixed.pth
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/approach_stage2_fan_mixed_vnc.sh "$CKPT" 2 1 3 10
```

## Stage 2 3종 혼합 · 정사각형 센서

`approach_stage2_square_mixed.yaml`은 위 3종 혼합 config에서 **센서만 바꾼 별도 실험**이다. task(`HumanoidMAMixedSensor`), 장면 혼합, 과제 비율, RSI, 보상, 종료, 학습 config는 fan mixed와 같다. 기존 fan mixed 실험과 스크립트는 바꾸지 않았다.

- **센서 범위:** `sensor_pattern: square`. 에이전트 root(pelvis) XY를 중심으로 하고 root heading을 따라 회전하는 한 변 14m 정사각형이다. `sensor_range: 7.0`은 중심에서 앞·뒤·좌·우로 7m라는 뜻이다(2026-10-08에 10m에서 축소). 머리 시점·시야각은 쓰지 않으며, 뒤쪽도 앞쪽과 같은 범위를 덮는다.
- **샘플링:** `sensor_resolution: 48`로 48×48, 에이전트당 **2304점**이다. 점 간격은 14/47 ≈ 0.298m이고 ±7m 경계에도 샘플이 있다. 바깥 인덱스는 로컬 x(뒤→앞), 안쪽 인덱스는 로컬 y(오른쪽→왼쪽)다.
- **높이값:** 부채꼴과 같은 가림 없는 수직 조회다(무대·지면·다리·과제 상자·활성 room 벽). `(표면 높이 − pelvis 높이)`를 ±3m로 clip하고 3으로 나눈다. 미관측 값(2.0)은 생기지 않는다.
- **관측·모델:** 관측은 **5215차원**이다(fan mixed 2655). 센서 encoder 구조는 같고 첫 층 입력만 2304로 바뀐다.
- **checkpoint:** sensor metadata **version 8**. fan mixed(version 7) checkpoint와 이전 10m 정사각형 checkpoint는 resume·평가 검사에서 거부된다. Stage 1에서 새로 전이한다(기본 `output/stage1/8000.pth`).
- **비용:** 2048환경에서 센서 함수만 측정하면 step당 5.0ms(부채꼴 1024점)에서 10.9ms로 늘었다. 학습 epoch 시간은 약 +7%로 추정한다(2026-10-08, 다른 학습과 GPU를 공유한 상태에서 측정했고 실제 학습으로 확인하지 않았다).
- **viewer:** `SENSOR_VIS=1`에서 두 에이전트의 샘플을 하늘색 십자로, 정사각형 경계를 파란 선으로 표시한다. 경계는 바깥 샘플을 잇는 선이라 지형 높이를 따라 올라가고 내려간다.
- **output:** 학습 `output/approach_stage2_square_mixed`, watch `output/approach_stage2_square_mixed_watch`, 센서 마스킹 평가 `output/approach_stage2_square_mixed_masked`.

```bash
GPU=0
STAGE1=output/stage1/8000.pth  # train/watch 기본값이라 STAGE1_CHECKPOINT는 생략 가능
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_square_mixed_train.sh 2 2048 3

# 짧은 학습 확인: 환경 수는 유지하고 반복 횟수만 제한
TOKENHSI_GPU="$GPU" MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_stage2_square_mixed_check/train \
  bash tokenhsi/scripts/multi_agent/approach_stage2_square_mixed_train.sh 2 2048 3

# 실제 학습으로 생성된 정사각형 센서 checkpoint 경로를 지정한다.
CKPT="output/approach_stage2_square_mixed/<run>/nn/ApproachStage2SquareMixed.pth"

# 로컬 viewer
TOKENHSI_GPU="$GPU" HEADLESS=0 SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_square_mixed_test.sh "$CKPT" 2 1 3 10

# 화면 없는 랜덤 평가
TOKENHSI_GPU="$GPU" HEADLESS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_square_mixed_test.sh "$CKPT" 2 96 3 3

# 서버 noVNC
TOKENHSI_GPU="$GPU" SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_square_mixed_vnc.sh "$CKPT" 2 1 3 10

# 같은 정사각형 센서 실험 재개
TOKENHSI_GPU="$GPU" RESUME_CHECKPOINT="$CKPT" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_square_mixed_train.sh 2 2048 3

# 학습 시각화: 기본 4환경, amp_ma_cliff_sensor_watch.yaml
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_square_mixed_watch.sh 2 4 3
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT="$STAGE1" SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_square_mixed_watch_vnc.sh 2 4 3
```

`TASK_GRAPH`·`SENSOR_MASK`·`EPISODE_LENGTH`의 의미는 fan mixed와 같다. 테스트 시각화에서 특정 장면만 보려면 `SCENARIO=cliff|bridge|room`을 붙인다. 평가 전용이며 학습·watch에서는 에러로 종료한다.

`BRIDGE_FRICTION`은 연결다리 actor의 접촉 마찰계수이며 기본값은 `1.0`이다. 유한한 0 이상 값을 지정한다. 무대·방·사람·작업 상자의 마찰은 바꾸지 않는다. square mixed, fan mixed, fan bridge의 train/test/watch/VNC에서 공통으로 적용된다. 실행 시 적용하는 값으로 checkpoint metadata에는 추가하지 않으며 기존 checkpoint도 그대로 평가할 수 있다. 재개·평가할 때 원하는 값을 다시 지정한다.

```bash
# room 장면만 noVNC로 보기
SCENARIO=room TOKENHSI_GPU="$GPU" SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_square_mixed_vnc.sh "$CKPT" 2 1 3 10

# 연결다리 마찰계수를 0.3으로 지정해 bridge 장면만 보기
SCENARIO=bridge BRIDGE_FRICTION=0.3 TOKENHSI_GPU="$GPU" SENSOR_VIS=1 \
  bash tokenhsi/scripts/multi_agent/approach_stage2_square_mixed_vnc.sh "$CKPT" 2 1 3 10
```

7m 변경 후 검증(2026-10-08): CPU `test_vision_sensor.py`·`test_mixed_sensor.py` **24개 통과**(정사각형 테스트는 config의 범위를 읽는다). PhysX 96환경×6 reset에서 앞·뒤 변 중심이 heading 방향 ±7m, 중심 오차는 최대 0.15mm였다. 관측과 재계산 값이 일치했고 관측은 5215차원이다. 10m checkpoint 거부도 확인했다(`env_7m/square_sensor.json`). 관측 차원·학습 경로가 같아 학습·평가는 다시 실행하지 않았다.

최초 10m 구현 검증(2026-10-08):
- CPU 테스트 **70개 통과**(센서·무대·다리·room·혼합 관련 8개 파일). 새로 추가한 2개는 root 중심·heading 회전·뒤쪽 상자 관측 검사와 viewer 외곽선 검사다.
- PhysX **96환경×6 reset**에서 관측 5215차원, metadata 8을 확인했다. 센서 중심과 root XY의 오차는 최대 0.12mm였다. 앞·뒤 변 중심은 heading 방향으로 ±10m였다. 관측에 들어간 센서 값은 재계산 값과 일치했다.
- **2048환경·`MAX_ITERATIONS=1`** 전이 학습과 저장을 완료했다. tensor 복사 169개·신규 12개, 센서 첫 층 256×2304, 모든 값이 유한했다. fan mixed checkpoint는 거부됐다.
- 저장 checkpoint로 96환경·최대 64-step 평가를 완료했다(시작 장면 29/37/30).
- 실제 viewer 1환경에서 세 장면을 캡처했다.
- 이것은 실행 검증이며, 장기 학습 성능은 검증하지 않았다.

증거: `output/approach_stage2_square_mixed_check/`의 `env/square_sensor.json`, `train.log`, `eval/metrics/`, `screenshots/`. 확인 스크립트(`square_env_check.py`, `square_capture.py`)도 같은 폴더에 있다.

## Stage 2 room partition · HOLDING_AT·내부 칸막이

[새 실제 Isaac Gym 화면·검증 보고서](room_partition_report_20260930.md). 환경은 `HumanoidMARoom`을 공유한다.

- 4면 벽 내부 중앙에 두께 0.4m·바닥 위 높이 2.5m의 칸막이를 둔다. 양 끝 통로는 각각 1.8m다. 방 크기 6–8m와 1024개·가림 없는 비전센서는 기본 room과 같다.
- HOLDING/SIT/CLIMB/HOLDING_AT을 agent별 각각 25%로 샘플한다. 두 사람과 상자 3개는 실내에 둔다. HOLDING/HOLDING_AT은 agent별로 같은 쪽/반대편 시작을 각각 50%로 선택하며(`room.partition.pickup_cross_probability: 0.5`), 반대편 시작은 상자를 집으러 가는 직선 경로가 칸막이에 막히게 배치한다. SIT/CLIMB은 같은 쪽 시작을 유지한다. 활성 HOLDING_AT 목적지는 항상 상자의 반대편이다. 상자와 목적지의 직선 연결이 칸막이를 지나므로 끝 통로를 통한 이동이 필요하다.
- 목적지 높이는 바닥+대상 상자 높이/2다. graph의 무작위 객체 binding·goal slot을 따라 배치하며, 실제 viewer의 활성 목표 원/marker는 담당 에이전트 색을 쓴다. 미래 목표에는 현재 다른 사람이 서 있을 수 있다. 초기 상자와 목표의 겹침은 배제한다.
- 기존 단일 collision 항에 칸막이 4개 면의 CPA 위험도를 포함한다. 과제 보상·계수·0.7m·벽 최근접점 CPA·벽 시간 할인 0.95·벽 접촉 후 계속 진행은 같다. 네트워크 관측 크기는 2655차원이다.
- 새 `pickup_cross_probability: 0.5` 설정 이전의 partition checkpoint도 room metadata가 달라 직접 resume·평가하지 않는다. 새 배치에서는 Stage 1 전이로 시작한다.
- sensor metadata version 6으로 기존 plain room/bridge/cliff checkpoint를 구분한다. Stage 1에서 새로 전이하며 같은 partition checkpoint만 resume·평가한다. 기본 output은 `output/approach_stage2_fan_room_partition_cpa_wall`, 마스킹 평가는 `_cpa_wall_masked`다.

학습·로컬 추론·화면 없는 평가·서버 VNC:

```bash
GPU=0
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT=output/stage1/28-17-01-48/nn/stage1.pth \
  bash tokenhsi/scripts/multi_agent/approach_stage2_fan_room_partition_train.sh 2 2048 3

CKPT="output/approach_stage2_fan_room_partition_cpa_wall/<new_run>/nn/ApproachStage2FanRoomPartition.pth"
TOKENHSI_GPU="$GPU" HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_stage2_fan_room_partition_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" HEADLESS=1 bash tokenhsi/scripts/multi_agent/approach_stage2_fan_room_partition_test.sh "$CKPT" 2 16 3 3
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_stage2_fan_room_partition_vnc.sh "$CKPT" 2 1 3 10
```

짧은 학습에는 `MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_stage2_fan_room_partition_cpa_check/train`을 추가한다. resume은 `STAGE1_CHECKPOINT` 대신 `RESUME_CHECKPOINT`를 지정한다. `TASK_GRAPH=holding_at`은 한 에이전트의 HOLDING_AT과 다른 에이전트의 HOLDING을 고정하는 평가 preset이다.

실제 viewer 캡처 재현:

```bash
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_sm120 \
  bash tokenhsi/scripts/multi_agent/run-gui.sh bash -c '
    source tokenhsi/scripts/multi_agent/runtime_env.sh
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    PYTHONPATH=tokenhsi python tokenhsi/tests/room_capture.py \
      --task HumanoidMARoom --no_video --test --eval \
      --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_fan_room_partition.yaml \
      --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_cliff_sensor.yaml \
      --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
      --num_agents 2 --num_objects 3 --num_envs 1 \
      --output_path output/approach_stage2_fan_room_partition_cpa_check/screenshots'
```

CPU 검사는 `test_room_partition.py`와 기존 room/bridge/cliff 검사, PhysX 검사는 해당 config로 `room_physics_smoke.py --headless --num_envs 64`를 사용한다. 캡처의 두-agent HOLDING_AT 장면은 관계를 보기 쉽게 촬영 시 두 과제를 고정한 예시이며, 본학습 비율을 바꾸지 않는다.

픽업 반대편 배치 검증(2026-09-30): 실제 PhysX 64환경×8 reset에서 HOLDING **120/265**, HOLDING_AT **123/223**건이 반대편 시작이며 모두 직선 경로가 칸막이에 막혔다. SIT/CLIMB 반대편 0건, 활성 AT 목표 223건과 실내 배치·벽/칸막이 접촉 검사도 통과했다. [물리 검사 결과](../output/approach_stage2_fan_mixed_pickup_check/room_physics/physics.json).

## Stage 2 협력 실험 · 29번

29번은 지정한 Stage 1 `.pth`에서 actor/critic/AMP weight와 관찰 통계를 옮겨 새 실행을 시작한다. `RESUME_CHECKPOINT`는 이미 시작한 **같은 Stage 2 변형**을 이어 학습할 때만 쓴다. 기존 29번의 SIT 성공과 보상 정규화는 **28번 정의**다.

```bash
STAGE1_CHECKPOINT='/absolute/path/to/Stage1.pth' TOKENHSI_GPU="$GPU" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_train.sh 2 2048 3
RESUME_CHECKPOINT='/absolute/path/to/ApproachStage2Coordination.pth' TOKENHSI_GPU="$GPU" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachStage2Coordination.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=place_stack \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=place_climb \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_vnc.sh "$CKPT"
```

Stage 2 plane은 34번의 SIT plane 성공·자기 edge 합계·동료 edge 평균을 적용한다. 협력 과제 비율은 `place_climb/place_sit/place_stack = 20/20/40%`, 독립 과제는 20%다. 독립 과제 내부의 HOLDING/SIT/CLIMB/HOLDING+AT/HOLDING+ON_TOP은 34번과 같이 10/10/10/35/35이며 RSI·가까운 시작 일정도 34번과 같다. 협력 과제는 공유 물체에 맞춘 Stage 2 RSI를 유지한다. 별도 config·스크립트·output을 사용한다.

```bash
STAGE1_CHECKPOINT='/absolute/path/to/ApproachScenarioStage1SelfSum.pth' TOKENHSI_GPU="$GPU" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_sit_plane_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachStage2CoordinationSitPlane.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=place_stack \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_sit_plane_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=place_stack \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_sit_plane_vnc.sh "$CKPT"
```

3·4인 평가 graph는 `tokenhsi/data/cfg/multi_agent/graphs/`에 있다. 입력 크기 검증용이며 성공은 별도 학습으로 확인한다.

## 원본 1번

원본 비교가 필요할 때 `ma_carry_train.sh`, `ma_carry_test.sh`, `ma_carry_watch.sh`를 사용한다. 실험별 GPU 지정과 데이터 경로는 위 공통 규칙을 따른다.

지표 정의는 [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 참고한다. TensorBoard는 `tensorboard --logdir output --host 127.0.0.1 --port 6006`으로 실행한다.
