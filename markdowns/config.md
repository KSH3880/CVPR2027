# Multi-Agent Carry 실행 가이드

실행 가능한 20개 config의 용도와 명령을 한곳에 정리한다. 모든 명령은 저장소 루트에서 실행한다.

```bash
cd /home/cvlab/Desktop/CVPR2027/approach_clean_scenario_ontopclimb_plane
```

## 공통 규칙

- 학습 인자: `[num_agents] [num_envs] [num_objects]`; 기본값은 `2 2048 3`이다.
- 로컬/VNC 인자: `<checkpoint.pth> [num_agents] [num_envs] [num_objects] [eval_repeats]`; 기본 예시는 `2 1 3 10`이다.
- 학습 환경 수는 RTX PRO 6000 기준 2048로 유지한다. 짧은 확인도 환경 수가 아니라 `MAX_ITERATIONS`만 줄인다.
- 실행 전에 `nvidia-smi`로 GPU 점유를 확인하고 `TOKENHSI_GPU`를 한 번만 지정한다.
- 본학습 전에 셸에 남은 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT`를 확인한다.
- 다른 reward/schema의 checkpoint는 섞지 않는다. 21·22번도 박스 분포가 다르므로 전용 checkpoint를 사용한다.
- 현재 Stage 1 상세는 [edge_context_stage1.md](edge_context_stage1.md), 과거 설계·실험 문서는 [legacy](legacy/README.md)에 있다.

## Config 목록

2~8번은 삭제되어 번호가 비어 있다.

| 번호 | Config | 용도 |
| ---: | --- | --- |
| 1 | [amp_humanoid_ma_carry.yaml](../tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml) | 원본 TokenHSI 방식 |
| 9 | [approach_rsi_all_edges.yaml](../tokenhsi/data/cfg/multi_agent/approach_rsi_all_edges.yaml) | Holding·At approach blending + RSI 웜업 |
| 10 | [approach_rsi_all_edges_success_sat.yaml](../tokenhsi/data/cfg/multi_agent/approach_rsi_all_edges_success_sat.yaml) | 9번 + current success 포화 |
| 11 | [approach_distance.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance.yaml) | 통합 distance progress, 기존 +10/latch |
| 12 | [approach_distance_success.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_success.yaml) | current success reward + edge 포화 |
| 13 | [approach_distance_success_no_sat.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_success_no_sat.yaml) | 12번에서 edge 포화 제거 |
| 14 | [approach_distance_success_holding_k10.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_success_holding_k10.yaml) | 12번의 Holding k=10 비교 |
| 15 | [approach_distance_success_ontop_mixed.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_success_ontop_mixed.yaml) | carry/독립 OnTop/의존 OnTop 혼합, pretrained 전이 |
| 16 | [approach_distance_edge_context_success.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_edge_context_success.yaml) | Holding·At PRE/TERM context, schema 2 scratch |
| 17 | [approach_distance_edge_context_ontop.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_edge_context_ontop.yaml) | sampled OnTop, schema 3 scratch |
| 18 | [approach_distance_edge_context_interaction.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_edge_context_interaction.yaml) | SIT/CLIMB interaction, schema 4 scratch |
| 19 | [approach_distance_edge_context_stage1.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_edge_context_stage1.yaml) | 독립 relation skill, schema 5 scratch |
| 20 | [approach_distance_edge_context_stage1_ground_sit_climb.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_edge_context_stage1_ground_sit_climb.yaml) | 19번 + 단독 SIT/CLIMB 대상 바닥 배치 |
| 21 | [approach_distance_edge_context_stage1_common_boxes.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_edge_context_stage1_common_boxes.yaml) | 20번 + 가변 박스와 box-top SIT 목표 |
| 22 | [approach_distance_edge_context_stage1_fixed_boxes_sit_fix.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_edge_context_stage1_fixed_boxes_sit_fix.yaml) | 고정 박스와 수정된 SIT 목표 |
| 23 | [approach_distance_edge_context_stage1_primitives_rsi.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_edge_context_stage1_primitives_rsi.yaml) | H/S/C 단일 edge + relation RSI, schema 6 scratch |
| 24 | [approach_distance_stage1_climb_rsi.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_stage1_climb_rsi.yaml) | CLIMB 전용 semantic graph + RSI, schema 7 scratch |
| 25 | [approach_scenario_no_climb.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_no_climb.yaml) | 4-template 무작위 object/goal binding, CLIMB 없는 schema 8 scratch |
| 26 | [approach_scenario_with_climb.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_with_climb.yaml) | 25번 + standalone CLIMB, schema 9 scratch |
| 27 | [approach_independent_climb_placement_region.yaml](../tokenhsi/data/cfg/multi_agent/approach_independent_climb_placement_region.yaml) | CLIMB·AT·ONTOP 독립 과제, 안전영역 성공, schema 10 scratch |

25번의 `ApproachScenarioNoClimb_23-08-13-26` 및 그 이전 checkpoint는 미사용 goal의 GTA 좌표가 오염된 상태로 학습됐다. resume하지 말고 현재 코드로 scratch 재학습한다.

### 27번 독립 CLIMB·배치 실험

- 26번의 중앙점 기준 state와 progress를 유지한다. ONTOP 성공은 source 상자 중심이 support 상자 윗면의 안쪽 영역(각 변 전체 길이의 10% 제외)에 있고, source bottom과 support top의 높이 오차가 1mm 이하다. CLIMB 성공은 root XY가 대상 상자 안쪽 영역(각 변 5% 제외)에 있고, 목표 root Z 오차가 20cm 이하이며, 평균 발 높이와 윗면 높이 오차가 7cm 이하다. 성공 시 해당 edge의 세 보상 항을 각각 0.2로 포화한다.
- 각 agent는 CLIMB·HOLDING_AT·HOLDING_ON_TOP을 각각 1/3 빈도로 받는다. ONTOP+ONTOP은 제외한다. CLIMB/AT끼리의 4개 순서쌍은 각각 1/12, ONTOP이 포함된 4개 순서쌍은 각각 1/6이다. 에이전트마다 별도의 task 물체를 쓰고, ONTOP의 받침은 남은 중립 물체만 쓴다. 물체의 논리 슬롯과 물리 배정은 매 리셋 섞이며, AT+AT의 두 goal도 분리된다.
- 과제별 RSI 분포는 26번의 CLIMB/HOLDING_AT/HOLDING_ON_TOP 행을 그대로 쓴다. task reward는 자기 몫 100%, 상대 몫 0%다. 기존 checkpoint와 보상·샘플러 계약이 달라 scratch로 시작한다.
- 학습: `TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/approach_independent_climb_placement_region_train.sh 2 2048 3`
- 로컬 추론: `TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=climb_ontop bash tokenhsi/scripts/multi_agent/approach_independent_climb_placement_region_test.sh "$CKPT" 2 1 3 10`
- 서버 VNC: `TOKENHSI_GPU=0 TASK_GRAPH=climb_ontop bash tokenhsi/scripts/multi_agent/approach_independent_climb_placement_region_vnc.sh "$CKPT"`
- 화면 없는 평가: 로컬 추론 명령의 `HEADLESS=1`로 변경한다. `TASK_GRAPH=random_scenario`는 학습 분포, `climb|holding_at|holding_ontop|climb_ontop|at_ontop`은 고정 조합이다. `TASK_ROLE_SWAP=1`로 고정 조합의 agent 역할을 바꿀 수 있다.

## 학습

아래에서 원하는 실험 하나만 실행한다. `GPU`는 실제 빈 장치로 바꾼다. 16~27번은 scratch이며, 15번만 스크립트에 지정된 기존 carry checkpoint를 전이한다.

```bash
GPU=5

# 1
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/ma_carry_train.sh 2 2048 3
# 9
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_rsi_all_edges_train.sh 2 2048 3
# 10
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_rsi_all_edges_success_sat_train.sh 2 2048 3
# 11
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_train.sh 2 2048 3
# 12
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_success_train.sh 2 2048 3
# 13
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_success_no_sat_train.sh 2 2048 3
# 14
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_success_holding_k10_train.sh 2 2048 3
# 15
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_success_ontop_mixed_train.sh 2 2048 3
# 16
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_success_train.sh 2 2048 3
# 17
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_train.sh 2 2048 3
# 18
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_interaction_train.sh 2 2048 3
# 19
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_train.sh 2 2048 3
# 20
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_ground_sit_climb_train.sh 2 2048 3
# 21
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_common_boxes_train.sh 2 2048 3
# 22
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_fixed_boxes_sit_fix_train.sh 2 2048 3
# 23
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_primitives_rsi_train.sh 2 2048 3
# 24
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_stage1_climb_rsi_train.sh 2 2048 3
# 25
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_no_climb_train.sh 2 2048 3
# 26
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_with_climb_train.sh 2 2048 3
# 27
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_independent_climb_placement_region_train.sh 2 2048 3
```

짧은 확인과 resume은 본학습 output과 분리한다.

```bash
GPU=5
MAX_ITERATIONS=3 OUTPUT_PATH=output/approach_distance_edge_context_stage1_primitives_rsi_check \
  TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_primitives_rsi_train.sh 2 2048 3

RESUME_CHECKPOINT='/absolute/path/to/checkpoint.pth' SEED=42 \
  TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_primitives_rsi_train.sh 2 2048 3
```

1번 `ma_carry_train.sh`는 `RESUME_CHECKPOINT` 환경변수를 처리하지 않는다.

## 로컬 시각화

`CKPT`를 반드시 같은 번호의 학습 결과로 바꾼다. 로컬 창을 띄우려면 `HEADLESS=0`을 유지한다.

```bash
GPU=5

# 1
CKPT='/path/to/ma_carry.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 bash tokenhsi/scripts/multi_agent/ma_carry_test.sh "$CKPT" 2 1 3 10
# 9
CKPT='/path/to/ApproachRSIAllEdges.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_rsi_all_edges_test.sh "$CKPT" 2 1 3 10
# 10
CKPT='/path/to/ApproachRSIAllEdgesSuccessSat.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_rsi_all_edges_success_sat_test.sh "$CKPT" 2 1 3 10
# 11
CKPT='/path/to/ApproachDistance.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_distance_test.sh "$CKPT" 2 1 3 10
# 12
CKPT='/path/to/ApproachDistanceSuccess.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_distance_success_test.sh "$CKPT" 2 1 3 10
# 13
CKPT='/path/to/ApproachDistanceSuccessNoSat.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_distance_success_no_sat_test.sh "$CKPT" 2 1 3 10
# 14
CKPT='/path/to/ApproachDistanceSuccessHoldingK10.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_distance_success_holding_k10_test.sh "$CKPT" 2 1 3 10
# 15
CKPT='/path/to/ApproachDistanceSuccessOntopMixed.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 ONTOP_SCENARIO=carry_ontop_dependent bash tokenhsi/scripts/multi_agent/approach_distance_success_ontop_mixed_test.sh "$CKPT" 2 1 3 10
# 16
CKPT='/path/to/ApproachDistanceEdgeContextSuccess.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_success_test.sh "$CKPT" 2 1 3 10
# 17
CKPT='/path/to/ApproachDistanceEdgeContextOntop.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=at_ontop bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_test.sh "$CKPT" 2 1 3 10
# 18
CKPT='/path/to/ApproachDistanceEdgeContextInteraction.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=hold_sit bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_interaction_test.sh "$CKPT" 2 1 3 10
# 19
CKPT='/path/to/ApproachDistanceEdgeContextStage1.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_sit bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_test.sh "$CKPT" 2 1 3 10
# 20
CKPT='/path/to/ApproachDistanceEdgeContextStage1GroundSitClimb.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=sit bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_ground_sit_climb_test.sh "$CKPT" 2 1 3 10
# 21
CKPT='/path/to/ApproachDistanceEdgeContextStage1CommonBoxes.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=sit bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_common_boxes_test.sh "$CKPT" 2 1 3 10
# 22
CKPT='/path/to/ApproachDistanceEdgeContextStage1FixedBoxesSitFix.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=sit bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_fixed_boxes_sit_fix_test.sh "$CKPT" 2 1 3 10
# 23
CKPT='/path/to/ApproachDistanceEdgeContextStage1PrimitivesRsi.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=sit bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_primitives_rsi_test.sh "$CKPT" 2 1 3 10
# 24
CKPT='/path/to/ApproachDistanceStage1ClimbRsi.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_distance_stage1_climb_rsi_test.sh "$CKPT" 2 1 3 10
# 25
CKPT='/path/to/ApproachScenarioNoClimb.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_no_climb_test.sh "$CKPT" 2 1 3 10
# 26
CKPT='/path/to/ApproachScenarioWithClimb.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_scenario_with_climb_test.sh "$CKPT" 2 1 3 10
# 27
CKPT='/path/to/ApproachIndependentClimbPlacementRegion.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=climb_ontop bash tokenhsi/scripts/multi_agent/approach_independent_climb_placement_region_test.sh "$CKPT" 2 1 3 10
```

화면 없는 평가는 같은 명령에서 `HEADLESS=1`로 바꾸고 환경 수를 `1`에서 `64`로 늘린다. 16~18번은 `TASK_GRAPH=random`, 19~23번은 `TASK_GRAPH=random_stage1`, 25~27번은 `TASK_GRAPH=random_scenario`로 학습 분포를 평가한다. 결과 JSON은 해당 output의 `metrics/`에 저장된다.

## 서버 VNC 시각화

12번과 14~27번은 전용 wrapper가 있다. `CKPT`와 `GPU`만 바꾸고 원하는 명령 하나를 실행한다.

```bash
GPU=5

# 12
CKPT='/path/to/ApproachDistanceSuccess.pth'
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_success_vnc.sh "$CKPT"
# 14
CKPT='/path/to/ApproachDistanceSuccessHoldingK10.pth'
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_success_holding_k10_vnc.sh "$CKPT"
# 15
CKPT='/path/to/ApproachDistanceSuccessOntopMixed.pth'
TOKENHSI_GPU="$GPU" ONTOP_SCENARIO=carry_ontop_dependent bash tokenhsi/scripts/multi_agent/approach_distance_success_ontop_mixed_vnc.sh "$CKPT"
# 16
CKPT='/path/to/ApproachDistanceEdgeContextSuccess.pth'
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_success_vnc.sh "$CKPT"
# 17
CKPT='/path/to/ApproachDistanceEdgeContextOntop.pth'
TOKENHSI_GPU="$GPU" TASK_GRAPH=at_ontop bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_vnc.sh "$CKPT"
# 18
CKPT='/path/to/ApproachDistanceEdgeContextInteraction.pth'
TOKENHSI_GPU="$GPU" TASK_GRAPH=hold_sit bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_interaction_vnc.sh "$CKPT"
# 19
CKPT='/path/to/ApproachDistanceEdgeContextStage1.pth'
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_sit bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_vnc.sh "$CKPT"
# 20
CKPT='/path/to/ApproachDistanceEdgeContextStage1GroundSitClimb.pth'
TOKENHSI_GPU="$GPU" TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_ground_sit_climb_vnc.sh "$CKPT"
# 21
CKPT='/path/to/ApproachDistanceEdgeContextStage1CommonBoxes.pth'
TOKENHSI_GPU="$GPU" TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_common_boxes_vnc.sh "$CKPT"
# 22
CKPT='/path/to/ApproachDistanceEdgeContextStage1FixedBoxesSitFix.pth'
TOKENHSI_GPU="$GPU" TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_fixed_boxes_sit_fix_vnc.sh "$CKPT"
# 23
CKPT='/path/to/ApproachDistanceEdgeContextStage1PrimitivesRsi.pth'
TOKENHSI_GPU="$GPU" TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_primitives_rsi_vnc.sh "$CKPT"
# 24
CKPT='/path/to/ApproachDistanceStage1ClimbRsi.pth'
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_distance_stage1_climb_rsi_vnc.sh "$CKPT"
# 25
CKPT='/path/to/ApproachScenarioNoClimb.pth'
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_no_climb_vnc.sh "$CKPT"
# 26
CKPT='/path/to/ApproachScenarioWithClimb.pth'
TOKENHSI_GPU="$GPU" TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_scenario_with_climb_vnc.sh "$CKPT"
# 27
CKPT='/path/to/ApproachIndependentClimbPlacementRegion.pth'
TOKENHSI_GPU="$GPU" TASK_GRAPH=climb_ontop bash tokenhsi/scripts/multi_agent/approach_independent_climb_placement_region_vnc.sh "$CKPT"
```

전용 wrapper가 없는 1·9·10·11·13번은 해당 test script를 공통 GUI wrapper로 감싼다.

```bash
GPU=5
CKPT='/path/to/checkpoint.pth'
TEST_SCRIPT=tokenhsi/scripts/multi_agent/approach_distance_success_no_sat_test.sh
TOKENHSI_GPU="$GPU" VNC_DIR="$HOME/opt/vnc" \
  bash tokenhsi/scripts/multi_agent/run-gui.sh \
  bash "$TEST_SCRIPT" "$CKPT" 2 1 3 10
```

기본 포트는 `6080`이다. VS Code Remote-SSH의 Ports 탭에서 6080을 포워딩한 뒤 접속한다.

```text
http://localhost:6080/vnc.html?autoconnect=1&resize=remote
```

일반 SSH는 로컬 PC에서 다음 터널을 유지한다.

```bash
ssh -N -L 6080:127.0.0.1:6080 hwanhee@SERVER_HOST
```

- 포트 충돌 시 명령 앞에 `PORT=6081`을 붙이고 포워딩·URL도 같은 번호로 바꾼다.
- `x11vnc not found`면 `VNC_DIR="$HOME/opt/vnc"`를 지정한다.
- `websockify not found`면 `WEBSOCKIFY="$HOME/anaconda3/envs/tokenhsi/bin/websockify"`를 지정한다.
- 평가 종료 전 중단하려면 실행한 서버 터미널에서 `Ctrl+C`를 누른다.

## Output과 공통 옵션

- 기본 output: `output/<실험명>/`
- 짧은 검증: `OUTPUT_PATH=output/<실험명>_check`
- VNC 평가: `output/<실험명>_vnc`
- seed 고정: `SEED=42`
- 같은 실험 resume: `RESUME_CHECKPOINT='/absolute/path/to/checkpoint.pth'`
- 역할 반전: 지원하는 viewer에서 `TASK_ROLE_SWAP=1`
- checkpoint 검색: `find output/<실험명> -type f -name '*.pth'`

19~22번 preset은 `holding|sit|climb|holding_at|holding_ontop|holding_sit|holding_climb|random_stage1`, 23번은 `holding|sit|climb|random_stage1`, 25번은 `holding|sit|holding_at|holding_ontop|random_scenario`, 26번은 여기에 `climb`이 추가된다. 27번은 `climb|holding_at|holding_ontop|climb_ontop|at_ontop|random_scenario`를 사용한다. 17번은 `at_ontop|ontop_chain|independent_ontop|random`, 18번은 `sit_only|climb_only|hold_sit|hold_climb|at_then_sit|at_then_climb|ontop_then_climb|random`을 사용한다.

TensorBoard는 다음처럼 실행한다. 지표 정의는 [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 참고한다.

```bash
source tokenhsi/scripts/multi_agent/runtime_env.sh
tensorboard --logdir output --host 127.0.0.1 --port 6006
```
