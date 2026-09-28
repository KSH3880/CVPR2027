# Multi-Agent Carry 실행 가이드

현재 실행 가능한 실험은 원본 1번과 Stage 1 **27·28·30·31·32·33·34·35번**, Stage 2 **29번 및 SIT plane 변형**이다. 과거 9~26번 config와 전용 실행 스크립트는 정리했다. 모든 명령은 저장소 루트에서 실행한다.

## 공통 규칙

- 학습 인자는 `[num_agents] [num_envs] [num_objects]`; Stage 1 기본값은 `2 2048 3`이다. 짧은 확인도 환경 수 2048을 유지하고 `MAX_ITERATIONS`만 줄인다.
- 실행 전 `nvidia-smi`로 점유를 확인하고 명령 앞에 `TOKENHSI_GPU`를 지정한다. 실행 중인 프로세스는 임의 종료하지 않는다.
- 본학습 전 셸의 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 확인한다.
- 학습은 기본적으로 scratch다. 기존 실험의 reward/checkpoint 계약을 섞지 않는다. Stage 2 전이는 아래의 `STAGE1_CHECKPOINT`를 사용한다.
- 평가·VNC 인자는 `<checkpoint.pth> [agents] [envs] [objects] [repeats]`다. `HEADLESS=0`은 로컬 viewer, `HEADLESS=1`은 화면 없는 평가다.
- 로컬 viewer와 서버 VNC의 상자 색은 매 reset의 과제 배정을 따른다. 단독 대상은 해당 에이전트 색, 공동 대상은 노란색, 나머지는 회색이다. 실행 중인 뷰어는 다시 시작해야 반영된다.
- 데이터 원본 `/home/hwanhee/CVPR2027/TokenHSI`는 읽기 전용이며, 이 저장소는 심링크로 사용한다.

## 현재 실험

| 번호 | Config | 역할 |
| ---: | --- | --- |
| 1 | [amp_humanoid_ma_carry.yaml](../tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml) | 원본 multi-agent carry 기준 |
| 27 | [approach_scenario_independent_with_climb.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_independent_with_climb.yaml) | 독립 물체·goal binding, 점 기준 성공 |
| 28 | [approach_scenario_stage1_plane.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_plane.yaml) | 27번의 ON_TOP·CLIMB을 상판 영역 성공으로 변경; Stage 2 기본 출발점 |
| 29 | [approach_stage2_coordination.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_coordination.yaml) | Stage 1 checkpoint에서 협력 graph 학습 |
| Stage 2 plane | [approach_stage2_coordination_sit_plane.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_coordination_sit_plane.yaml) | 34번 기반 SIT plane·보상·독립 과제 분포 |
| 30 | [approach_scenario_stage1_sit_plane_normalized.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_sit_plane_normalized.yaml) | 28번 + SIT 상판 성공·유효 edge 평균 보상 |
| 31 | [approach_scenario_stage1_skill_curriculum.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_skill_curriculum.yaml) | 30번 + 가까운 시작·후반 CLIMB RSI·어려운 과제 증량·보상 shaping |
| 32 | [approach_scenario_stage1_skill_curriculum_reward_preserved.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved.yaml) | 31번의 시작 상태·RSI·샘플링을 유지하고 state/progress는 30번 식 |
| 33 | [approach_scenario_stage1_at_goal_fix.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_at_goal_fix.yaml) | 32번 + AT 목표 slot reset 오류 수정 |
| 34 | [approach_scenario_stage1_self_sum.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_self_sum.yaml) | 33번 + 자기 edge 보상 합계·동료 edge 평균, 과제 비율 10/10/10/35/35 |
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
