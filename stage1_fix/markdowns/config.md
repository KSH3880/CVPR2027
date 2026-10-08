# Multi-Agent Carry 실행 가이드

현재 실행 가능한 실험은 PUSH/DOOR Stage 1, 원본 1번과 Stage 1 **27·28·30·31·32·33·34·35·36·37번 및 paired/unified 변형**, Stage 2 **29번 및 SIT plane 변형**이다. 과거 9~26번 config와 전용 실행 스크립트는 정리했다. 모든 명령은 저장소 루트에서 실행한다.

## PUSH 상자 크기 (공통 학습·평가)

`push_door_stage1.yaml`와 `push_door_stage1_random_start.yaml`의 상자는 **1.1×1.1×1.1m 정육면체**다. 밀도는 약11.833kg/m³로 조정해 기존 질량15.75kg을 유지한다. `push_box_start_x()`가 최대 목표 거리·상자 XY 반경·전방 jitter를 고려해 미사용 문과 최소20cm 여유를 두고 배치한다. 랜덤 시작의 사람-상자 X 간격은 1.1~1.35m다. 큰 형상은 관측 bbox와 실제 물리 모두에 적용한다.

이 설정은 추적 파일에 반영되므로 commit/push/pull 대상이다. 실행 중인 작업은 기존 형상을 유지하고 새 프로세스부터 적용된다. 상자 형상/밀도가 checkpoint 계약에 포함돼 이전 크기의 checkpoint와 혼용하지 않는다.

## 공통 규칙

- 학습 인자는 `[num_agents] [num_envs] [num_objects]`; Stage 1 기본값은 `2 2048 3`이다. 짧은 확인도 환경 수 2048을 유지하고 `MAX_ITERATIONS`만 줄인다.
- 실행 전 `nvidia-smi`로 점유를 확인하고 명령 앞에 `TOKENHSI_GPU`를 지정한다. 실행 중인 프로세스는 임의 종료하지 않는다.
- 본학습 전 셸의 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 확인한다.
- 학습은 기본적으로 scratch다. 기존 실험의 reward/checkpoint 계약을 섞지 않는다. Stage 2 전이는 아래의 `STAGE1_CHECKPOINT`를 사용한다.
- 평가·VNC 인자는 `<checkpoint.pth> [agents] [envs] [objects] [repeats]`다. `HEADLESS=0`은 로컬 viewer, `HEADLESS=1`은 화면 없는 평가다.
- 로컬 viewer와 서버 VNC의 상자 색은 매 reset의 과제 배정을 따른다. 단독 대상은 해당 에이전트 색, 공동 대상은 노란색, 나머지는 회색이다. 상자 정리 데모는 모든 상자를 같은 회색으로 표시한다. AT goal 점도 해당 edge 담당 에이전트 색이며, goal 슬롯이 바뀌어도 배정을 따른다. 실행 중인 뷰어는 다시 시작해야 반영된다.
- 데이터 원본 `/home/hwanhee/CVPR2027/TokenHSI`는 읽기 전용이며, 이 저장소는 심링크로 사용한다.
- 4명·16상자 정리 데모는 아래 **상자 정리 VNC 데모** 절을 따른다. 지정된 rescue checkpoint를 사용하는 평가 전용 실행이다.

## 현재 실험

| 번호 | Config | 역할 |
| ---: | --- | --- |
| Push/Door | [push_door_stage1.yaml](../tokenhsi/data/cfg/multi_agent/push_door_stage1.yaml) | 독립 PUSH/문 개방·유지, task/phase AMP, grounded 진행도·유지 보상 |
| Push/Door random start | [push_door_stage1_random_start.yaml](../tokenhsi/data/cfg/multi_agent/push_door_stage1_random_start.yaml) | ZIP 기반 독립 랜덤 시작·유지 AMP door_tail·별도 checkpoint |
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
| 36 | [approach_scenario_stage1_placement_focus.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_placement_focus.yaml) | 34번 + HOLDING+AT/HOLDING+ON_TOP만 각 50% 샘플링 |
| 37 | [approach_scenario_stage1_ontop_putdown.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_ontop_putdown.yaml) | 36번 + ON_TOP에도 충돌 없는 중간 구간의 `putDown` RSI 10% 적용 |
| Paired AT | [approach_scenario_stage1_paired_at_no_near.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_paired_at_no_near.yaml) | 2-agent/3-object, 매 scene AT+AT, 가까운 시작 없음 |
| Paired placement | [approach_scenario_stage1_paired_placement_no_near.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_paired_placement_no_near.yaml) | 2-agent/4-object, scene별 AT+AT 또는 ON_TOP+ON_TOP 각 50%, 가까운 시작 없음 |
| Paired owner HOLDING | [approach_scenario_stage1_paired_placement_owner_holding.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_paired_placement_owner_holding.yaml) | Paired placement + AT/ON_TOP edge에 담당자의 선행 HOLDING `φ` 입력 |
| Unified semantic | [approach_scenario_stage1_unified.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified.yaml) | 2H/4O 독립 5과제, canonical binding·토큰 순열·과제 조건 AMP·unified RSI |
| Unified shared edge | [approach_scenario_stage1_unified_shared_edge_encoder.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder.yaml) | Unified semantic의 actor·critic semantic EdgeEncoder 파라미터 공유 |
| Unified owner HOLDING | [approach_scenario_stage1_unified_owner_holding.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_owner_holding.yaml) | Unified semantic + AT/ON_TOP 담당 HOLDING φ 입력 |
| Unified size RSI | [approach_scenario_stage1_unified_size_rsi.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi.yaml) | 행동별 실제 asset 크기·bbox progress·물리 검사한 후반 RSI, 과제 비율 5/10/25/30/30 |
| Unified size RSI typed bias | [approach_scenario_stage1_unified_size_rsi_typed_bias.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias.yaml) | Size RSI 유지, 타입·관계·타입별 독립 layer/head bias 표 |
| Unified size RSI typed message | [approach_scenario_stage1_unified_size_rsi_typed_bias_message.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message.yaml) | Typed bias 유지, 64-D 관계 메시지 추가 |
| Unified size RSI task message | [approach_scenario_stage1_unified_size_rsi_task_message.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message.yaml) | 복합 carry 과제·역할별 bias/message, 단방향 3연결·자기 보상만 |

27~37번 Stage 1은 2-agent/3-object 독립 graph와 schema 9 semantic 관찰을 쓴다. ON_TOP 두 개를 동시에 샘플하지 않는다. 28~37번의 팀 보상은 자기 0.9, 동료 0.1이다. 30~33번은 유효 edge를 agent별로 평균 내어 단일 edge와 두 edge의 최대 task 보상을 0.6으로 맞춘다. 34~37번은 자기 edge를 합산하고 동료 edge만 평균 내며 state·progress·성공 식은 33번과 같다.

Paired AT/paired placement도 schema 9 semantic 관찰과 34번의 reward·RSI·AMP를 유지한다. 두 agent의 **과제 relation만 같은 타입**으로 샘플하고 실제 motor action은 각자 선택한다. Paired placement는 두 ON_TOP에 서로 다른 받침을 주기 위해 4개 물체를 쓴다. 두 기준 실험은 가까운 시작 확률이 처음부터 끝까지 0이며, 각자 별도 checkpoint/output에서 scratch로 시작한다.

Paired owner HOLDING은 paired placement의 샘플링·reward·RSI·AMP·가까운 시작 없음 설정을 유지한다. 관측만 edge당 5개에서 6개 필드로 늘리고 AT/ON_TOP에는 같은 owner의 `H→source O` HOLDING `φ`, 나머지 유효 edge에는 1을 기록한다. actor·critic의 기존 semantic 64차원 뒤에서 AT/ON_TOP에만 상태 잔차를 더한다. 기존 paired checkpoint와 관측·네트워크 형식이 달라 별도 output에서 scratch로 학습한다.

2026-09-30 공통 코드 수정 이후 scenario의 상자 속도 패널티는 현재 graph에 연결된 담당 물체를 따른다. 이전에는 legacy agent assignment를 사용해 랜덤 binding에서 다른 물체의 패널티가 들어갈 수 있었다. 모델/관측 shape는 같아서 기존 checkpoint를 로드할 수 있지만, 수정 전후 학습의 penalty 의미는 다르다. 이미 실행 중인 프로세스에는 자동 적용되지 않으며 이후 새 프로세스부터 적용된다.

33번은 32번의 보상·샘플링·RSI를 유지한다. AT graph가 두 번째 목표 slot을 가리킬 때 reset도 해당 목표를 배치하도록 공통 코드의 slot 선택을 수정했다. 수정 후 새로 시작하는 27~32번 실행에도 적용되지만, 이미 실행 중인 프로세스와 checkpoint는 바뀌지 않는다. 비교 실험은 33번 전용 output에 scratch로 시작한다.

## 학습

아래 `GPU`를 사용할 장치 번호로 바꾼다. 각 Stage 1 실험은 별도 output에 scratch로 시작한다.

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
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_train.sh 2 2048 3  # 36
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_train.sh 2 2048 3  # 37
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_train.sh 2 2048 3  # Paired AT
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_train.sh 2 2048 4  # Paired placement
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_train.sh 2 2048 4  # Paired owner HOLDING
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_train.sh 2 2048 4  # Unified semantic
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_train.sh 2 2048 4  # Unified shared edge
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_train.sh 2 2048 4  # Unified owner HOLDING
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_train.sh 2 2048 4  # Unified size RSI
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_train.sh 2 2048 4  # Unified size RSI typed bias
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_train.sh 2 2048 4  # Unified size RSI typed message
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
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_test.sh "$CKPT" 2 1 3 10  # 36
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_test.sh "$CKPT" 2 1 3 10  # 37
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_test.sh "$CKPT" 2 1 3 10  # Paired AT
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_test.sh "$CKPT" 2 1 4 10  # Paired placement
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_test.sh "$CKPT" 2 1 4 10  # Paired owner HOLDING
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_test.sh "$CKPT" 2 1 4 10  # Unified semantic
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_test.sh "$CKPT" 2 1 4 10  # Unified shared edge
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_test.sh "$CKPT" 2 1 4 10  # Unified owner HOLDING
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_test.sh "$CKPT" 2 1 4 10  # Unified size RSI
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_test.sh "$CKPT" 2 1 4 10  # Unified size RSI typed bias
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_test.sh "$CKPT" 2 1 4 10  # Unified size RSI typed message
TOKENHSI_GPU=3 HEADLESS=0 bash tokenhsi/scripts/multi_agent/box_cleanup_demo_test.sh  # 4명·16상자 정리 데모
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
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_vnc.sh "$CKPT"  # 36
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_vnc.sh "$CKPT"  # 37
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_vnc.sh "$CKPT"  # Paired AT
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_vnc.sh "$CKPT"  # Paired placement
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_vnc.sh "$CKPT"  # Paired owner HOLDING
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_vnc.sh "$CKPT"  # Unified semantic
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_vnc.sh "$CKPT"  # Unified shared edge
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_vnc.sh "$CKPT"  # Unified owner HOLDING
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_vnc.sh "$CKPT"  # Unified size RSI
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_vnc.sh "$CKPT"  # Unified size RSI typed bias
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_vnc.sh "$CKPT"  # Unified size RSI typed message
TOKENHSI_GPU=3 PORT=6080 bash tokenhsi/scripts/multi_agent/box_cleanup_demo_vnc.sh  # 4명·16상자 정리 데모
TOKENHSI_GPU=0 PORT=6080 bash tokenhsi/scripts/multi_agent/at_goal_demo_vnc.sh  # AT 중심 높이 0~2m·세 XY 위치
```

## AT 높이·위치 VNC 데모

`distill_pth/ApproachScenarioStage1RescueAtKLClimb50_00008000.pth`를 사용하는 평가 전용 데모다. 별도 학습이나 새 YAML 없이 34번 config와 rescue checkpoint의 평가 호환 처리를 사용한다. 2명·3물체·1환경에서 두 사람 모두 HOLDING+AT를 수행한다.

- 목표 **상자 중심** z를 바닥 기준 `0, 0.2, …, 2.0m`로 순회한다. 각 높이마다 `above_box`(초기 상자 바로 위), `midpoint`(초기 사람·상자의 XY 중간), `side`(사람→상자 방향의 왼쪽으로 1m)를 순서대로 보여준다. 목표는 각 장면 동안 고정이다.
- 기본 33장면이며 장면마다 초기화한다. 같은 반복에서는 seed를 다시 적용해 초기 장면을 맞춘다. `REPEATS=2`면 다른 seed로 33장면을 한 번 더 실행한다. 기본 한 장면 600 control step(최대 시뮬레이션 시간 20초)이며 넘어지면 다음 장면으로 진행한다.
- z=0처럼 상자 중심이 바닥 아래로 들어가야 하는 목표도 값을 보정하지 않는다. 실제로 불가능한 목표에서 정책이 어떻게 반응하는지 보는 용도다. 바닥 접촉이나 목표 도달로 조기 완료하지 않는다.
- 목표는 담당자 색의 마커로 표시하며 카메라는 2m 높이까지 포함한다. 콘솔에 현재 장면 번호·XY 모드·목표 높이를 표시한다. `output/at_goal_demo/at_goal_results.json`에 목표·초기/최종 위치·최종 XYZ 오차를, viewer 실행 시 장면별 마지막 PNG를 저장한다. 이는 관찰용 기록이며 성공률 평가는 아니다.

```bash
# 실행 전 점유 확인. GPU 번호와 사용하지 않는 VNC 포트를 선택한다.
nvidia-smi
TOKENHSI_GPU=0 PORT=6080 bash tokenhsi/scripts/multi_agent/at_goal_demo_vnc.sh
# 바로 위로 들어 올리기만: 11개 높이
TOKENHSI_GPU=0 PORT=6080 GOAL_XY_MODES=above_box bash tokenhsi/scripts/multi_agent/at_goal_demo_vnc.sh
# 특정 높이만 보기 (쉼표로 구분, 공백 없음)
TOKENHSI_GPU=0 PORT=6080 GOAL_HEIGHTS=0.6,1.0,1.4,1.8,2.0 bash tokenhsi/scripts/multi_agent/at_goal_demo_vnc.sh
# 로컬 viewer / 짧은 headless 실행 확인
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/at_goal_demo_test.sh
TOKENHSI_GPU=0 HEADLESS=1 EPISODE_LENGTH=32 OUTPUT_PATH=output/at_goal_demo_check bash tokenhsi/scripts/multi_agent/at_goal_demo_test.sh
```

6080 포트를 포워딩한 뒤 noVNC에 접속한다. checkpoint는 첫 인자로 변경할 수 있지만 rescue 계약이 맞아야 한다. `EPISODE_LENGTH`, `SEED`, `REPEATS`, `OUTPUT_PATH`를 지정할 수 있으며 `KEEP_OPEN=0`이면 마지막 장면 뒤 viewer를 닫는다. 학습용 데모가 아니므로 train 스크립트는 없다.

## 상자 정리 VNC 데모

별도 학습/config 없이 34번 YAML을 바탕으로 `box_cleanup_demo.py`에서 평가 전용 task/player를 등록한다. 기본 checkpoint는 `distill_pth/ApproachScenarioStage1RescueAtKLClimb50_00008000.pth`다. 사람 4명·물리 상자 16개·8개 semantic edge로 정책을 그대로 로드하며, H223/O30 정규화 통계도 재사용한다. 데모 로더만 옛 rescue variant와 추가 AT/CLIMB shaping 설정을 현재 self-sum runtime에 대응시킨다. schema·packet·나머지 reward 계약은 검사하고, 원본 checkpoint와 일반 학습/평가 로더는 변경하지 않는다. 데모의 결과는 운반 진행 기록이며 rescue 보상 비교 평가는 아니다.

- 중앙에 상자 16개를 불규칙한 2단 더미로 모아 쌓는다. 가로·세로 0.40/0.45/0.50m, 높이 0.30/0.35/0.40m를 독립적으로 섞으며 밀도는 100kg/m³다. 기본 중심 간격 70cm에서 두 줄의 가로 위치·각 상자의 위치를 흐트러뜨리고 수평 회전을 ±15° 섞는다. 받침 위 대상도 조금 어긋나게 놓으며 시작 시 물체끼리 겹치지 않는다. 크기는 실행 시작 시 고정되고 위치·회전은 매 반복 다시 샘플한다. 배정 여부와 관계없이 같은 회색이다.
- 동서남북 사람 4명이 자기 방향의 윗단 상자를 하나씩 옮기고, 모두 끝나면 남은 윗단 상자 네 개를 배정한다. 운반 대상 8개 모두 처음에는 받침 상자 위에 있으며 실제 높이에 맞춰 초기 중심·바닥 목적지 높이를 계산한다. 해당 checkpoint의 저장된 학습 설정은 가로·세로 0.40~0.65m, 높이 0.25~0.55m다. 데모 크기는 이 범위 안이며 크기별 성공률을 보장하지 않는다.
- 목적지 XY 오차 12cm 미만·회전 bbox 바닥 오차 2.5cm 미만·선속도 0.15m/s 미만·각속도 0.5rad/s 미만을 15 control step(0.5초) 유지하면 내려놓기로 기록한다. 손 접촉 해제는 요구하지 않는다. 먼저 끝난 사람은 기존 배정을 유지하며 정책이 계속 자세를 제어한다.
- 네 상자가 동시에 안정적으로 놓이면 graph와 목적지만 바꾼다. 사람·상자를 teleport/reset하지 않고 두 번째 운반을 시작한다. 두 라운드 후 종료하며 viewer는 마지막 화면을 유지한다. 중앙 상자도 실제 물리 물체여서 접촉하면 움직이거나 더미가 무너질 수 있다.
- 이전 동일 크기·정렬된 중앙 더미 버전의 seed 42 검사에서는 7개 운반 후 마지막 윗단 집기가 정체되어 시간 초과했다. 현재 불규칙·크기 혼합 더미의 8개 완주는 아직 확인하지 못했다. 사용자 선택에 따라 4명·16물체 전체 관측을 그대로 입력하며 주변 관측 추론은 적용하지 않는다.

```bash
# 서버 VNC: 기본 10회·한 회 최대 1500 step(50초), 6080 포워딩 후 noVNC 접속
TOKENHSI_GPU=3 PORT=6080 bash tokenhsi/scripts/multi_agent/box_cleanup_demo_vnc.sh
# 로컬 viewer / 화면 없는 검사
TOKENHSI_GPU=3 HEADLESS=0 bash tokenhsi/scripts/multi_agent/box_cleanup_demo_test.sh
TOKENHSI_GPU=3 HEADLESS=1 REPEATS=1 OUTPUT_PATH=output/box_cleanup_demo_check bash tokenhsi/scripts/multi_agent/box_cleanup_demo_test.sh
```

checkpoint를 바꾸려면 첫 인자로 경로를 넘긴다. `REPEATS` 기본값은 10이며 각 반복마다 두 라운드 장면을 처음부터 재생한다. `KEEP_OPEN=0`은 viewer가 마지막 반복 종료 시 자동 닫히게 한다. `EPISODE_LENGTH` 기본값은 1500 control step(시뮬레이션 시간 50초), `SEED`는 42다. 실패·시간 초과도 `OUTPUT_PATH/cleanup_results.json`에 실제 상자 크기와 함께 기록하고 다음 라운드를 강제 성공시키지 않는다. Viewer는 `frame_0/120/300/600.png`와 `final.png`를 저장한다.

## Unified 독립 5과제 · semantic / shared edge / owner-HOLDING 비교

Semantic과 owner-HOLDING은 **edge 입력 상태 유무만 다르다**. Shared edge는 semantic과 과제·관측·보상이 같고 actor·critic의 semantic EdgeEncoder 임베딩·MLP·bias projection만 공유한다. 토크나이저·GTA/Transformer·head는 분리한다. 기존 paired/focus 실험은 보존하며 각자 새 output에서 scratch로 시작한다. 새 AMP 입력은 프레임당 기존 129차원에 과제 one-hot 3차원을 더한 132차원, 10프레임 총 1320차원이다. 기존 paired checkpoint와는 AMP 및 variant 계약이 달라 이어 학습하지 않는다.

- 학습: 2명·4물체. 각 agent가 독립적으로 `HOLDING/SIT/CLIMB/HOLDING_AT/HOLDING_ON_TOP = 5/5/20/35/35%`를 샘플한다. 두 ON_TOP도 가능하며 명목 빈도는 12.25%다.
- 논리 할당(0-based): H₀→O₀→G₀, H₁→O₁→G₁. SIT/CLIMB도 각자의 Oᵢ를 사용한다. ON_TOP은 O₀→O₂, O₁→O₃. 물리 asset slot만 reset마다 섞이며 모든 관측·RSI·보상은 동일 logical-to-physical mapping을 따른다.
- H/O/G별 인코딩 뒤 transformer 입력을 모든 타입에 걸쳐 매 forward 무작위 순열로 섞는다. 한 forward의 batch는 같은 순열을 사용하며, GTA와 bias의 두 축도 함께 이동한다. 출력은 canonical 순서로 복원한 다음 human별 actor/value head에 전달한다. 순열 등가성에 따른 수치 동등성은 검증하되 추가 학습 이득을 보장하지 않는다.
- Goal GTA quaternion은 `(0,0,0,1)`로 고정한다. 상자 크기는 X=0.40m, Y=0.30m, Z=0.30~0.50m(0.05m 간격)다.
- loco RSI의 담당 상자는 **owner root XY에서 반경 1~2m**에 배치한다. AT 목표/ON_TOP 받침에 가까운 시작은 0%이며 별개 설정이다. pickUp/carryWith/putDown/sit/climb RSI의 참조 물체 위치는 유지한다. 물리 배치 부적합 시 graph를 유지한 채 reset을 다시 뽑는다.
- state/progress/success 0.2/0.2/0.2, 자기 edge 합계·동료 edge 평균의 0.9/0.1 공유, placement 성공 시 matching HOLDING 포화는 유지한다. TERM 입력은 추가하지 않는다.

| 과제 | RSI |
| --- | --- |
| HOLDING | loco 50%, pickUp 50% |
| SIT | loco 50%, sit 50% |
| CLIMB | loco 50%, climb 50%; 원래 허용 시점 사용, 후반 강제 샘플링 없음 |
| HOLDING+AT | loco 50%, pickUp 10%, carryWith 30%, putDown 10% |
| HOLDING+ON_TOP | loco 50%, pickUp 10%, carryWith 40%; putDown 없음 |

AMP는 단일 판별기에 **각 agent의 할당 과제 family**를 매 프레임 붙인다. HOLDING/AT/ON_TOP은 carry, SIT는 sit, CLIMB은 climb이다. RSI로 걷기를 뽑았어도 family는 할당 과제를 따른다. 전문가의 과제 내부 분포는 원본 unified에 맞춘다: carry=loco/OMOMO/pickUp/putDown 1/3·1/3·1/6·1/6, sit=loco/sit 1/3·2/3, climb=loco/climb/climbNoRSI 각 1/3. `carryWith`는 RSI에는 있지만 이 AMP 전문가 혼합에는 없다.

`skillDiscProb`는 리셋 시 family 비율 75/5/20으로 환산한 명목 전체 분포다. 실제 조건부 expert 생성은 `utils/unified_training.py`의 분포를 사용한다. 학습에서는 전문가·리플레이를 현재 rollout의 family와 **행별로 맞춰** 뽑아 PPO minibatch마다 family 빈도가 일치한다. 리플레이에 해당 family가 아직 없을 때만 같은 family의 현재 rollout을 사용한다. 과제 라벨은 AMP 정규화에서 통과시키며 참조 이력·부분 reset에도 유지한다. task/AMP reward 비중은 기존 0.5/0.5다.

Semantic 실험은 edge당 `[valid,src,dst,relation,owner]` 5필드이고 semantic type/relation만 bias에 인코딩한다. Owner 실험은 6번째 값에 동일 owner·source의 HOLDING φ를 넣고 AT/ON_TOP의 semantic64 뒤에서 상태 잔차를 더한다. 다른 유효 edge의 입력 값은 1이며, 상태 잔차는 placement edge에만 적용한다.

아래 학습 명령에서 필요한 실험을 선택한다. `TOKENHSI_GPU=0`은 로컬 단일 GPU 기준이다.

```bash
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_train.sh 2 2048 4
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_train.sh 2 2048 4
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_train.sh 2 2048 4
```

각 checkpoint에는 같은 실험의 test/VNC를 사용한다. 로컬 viewer는 다음과 같다.

```bash
CKPT='/absolute/path/to/checkpoint.pth'
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_test.sh "$CKPT" 2 1 4 10
```

`TASK_GRAPH=holding|sit|climb|holding_at|holding_ontop`은 두 사람 모두 해당 과제로 평가한다. 기본 평가는 loco 시작이다. placement의 들고 있는 시작을 보려면 `TASK_GRAPH=holding_at` 또는 `holding_ontop`에 `EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0`을 추가한다. `HEADLESS=1`은 화면 없는 평가다.

```bash
TOKENHSI_GPU=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_vnc.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_vnc.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_vnc.sh "$CKPT" 2 1 4 10
```

각 기본 output은 `output/approach_scenario_stage1_unified`, `output/approach_scenario_stage1_unified_shared_edge_encoder`, `output/approach_scenario_stage1_unified_owner_holding`이다. Shared edge는 별도 variant/checkpoint로 scratch 학습하며 기존 unified checkpoint를 resume하지 않는다. 짧은 검증은 별도 `_check` output과 `MAX_ITERATIONS=2`를 사용한다. 2명·4물체 sampler이며 가변 인원 평가 sampler 확장은 포함하지 않는다.

## Stage 1 AT/ON_TOP 집중 실험 · 36번

34번과 동일한 reward·RSI·가까운 시작 설정에서 `HOLDING/SIT/CLIMB/HOLDING_AT/HOLDING_ON_TOP = 0/0/0/0.5/0.5`로 샘플한다. 한 scene에 AT와 ON_TOP이 각각 하나씩 배정되고, 두 과제 모두 자기 HOLDING edge를 포함한다. 단독 HOLDING 및 SIT/CLIMB 과제는 학습 중 나오지 않는다. `templateRsi`의 AT/ON_TOP 행은 그대로 적용되고, CLIMB 후반 RSI는 CLIMB 과제가 없어 사용되지 않는다. `skillDiscProb`의 SIT/CLIMB 모션 비율은 34번과 동일하다. 새 variant·output에서 scratch로 시작한다.

```bash
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachScenarioStage1PlacementFocus.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_vnc.sh "$CKPT"
```

## Stage 1 같은 과제 쌍 · 가까운 시작 없음

**Paired AT**은 3개 물체에서 두 agent에게 각각 다른 물체·목표를 배정한다. 각 agent의 edge는 `H→O HOLDING`, `O→G AT`이며 한 scene에 AT edge가 두 개다. **Paired placement**는 4개 물체에서 scene마다 `AT+AT` 또는 `ON_TOP+ON_TOP`을 50%씩 뽑는다. ON_TOP scene의 두 source와 두 support는 모두 다른 물체다. 두 실험 모두 34번의 RSI·AMP·보상식을 유지하고 `near_start` 확률만 0으로 고정한다. `TASK_GRAPH=holding_at`/`holding_ontop`으로 paired placement 평가 장면을 각각 고를 수 있다. 다른 물체 수와 variant는 서로 checkpoint 호환되지 않는다.

```bash
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachScenarioStage1PairedAtNoNear.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_vnc.sh "$CKPT"

TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_train.sh 2 2048 4
CKPT='/absolute/path/to/ApproachScenarioStage1PairedPlacementNoNear.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_vnc.sh "$CKPT"
```

## Stage 1 Paired placement · 담당자 HOLDING 상태 입력

위 Paired placement와 동일한 2-agent/4-object·AT+AT/ON_TOP+ON_TOP·가까운 시작 없음·RSI·AMP·reward를 사용한다. AT/ON_TOP edge의 여섯 번째 관측 필드는 동일 owner와 source 물체의 직전 HOLDING edge `φ`다. HOLDING 등 다른 유효 edge에는 1을 넣고, 무효 edge는 0으로 마스킹한다. `φ`는 손–상자 중심 거리 기반의 0~1 상태이며 실제 물리적 파지 플래그는 아니다. 기존 paired checkpoint는 새 관측 폭과 맞지 않는다.

```bash
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_train.sh 2 2048 4
CKPT='/absolute/path/to/ApproachScenarioStage1PairedPlacementOwnerHolding.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_vnc.sh "$CKPT"
```

## Stage 1 ON_TOP putDown RSI 실험 · 37번

36번과 동일한 과제·보상 분포에서 `HOLDING_ON_TOP` RSI 중 `carryWith` 10%를 `putDown`으로 옮긴다. ON_TOP `putDown`은 원본 모션의 45~70% 구간에서만 시작하고, 받침 상자를 모션의 최종 놓기 XY에 배치한다. 이 구간은 원본 모션의 후기 상자·받침 겹침을 피한다. 받침의 물리 크기는 기존 분포 그대로이며, reset 시 상자 또는 사람과 겹치면 재추첨한다. AT `putDown` RSI 10%와 나머지 설정은 36번과 같다. 별도 variant·output에서 scratch로 시작한다.

```bash
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachScenarioStage1OntopPutdown.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_vnc.sh "$CKPT"
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

## Unified size RSI

`approach_scenario_stage1_unified_size_rsi`는 unified semantic에서 분리한 scratch 실험이다. 기존 unified checkpoint는 reward 계약이 달라 재개할 수 없다. 2-agent/4-object canonical 배정, semantic edge 관찰, 토큰·edge 순열, state·성공 조건·보상 가중치, 가까운 시작 없음은 유지한다.

| 과제/물체 | X·Y 각각 (m) | Z (m) |
| --- | --- | --- |
| HOLDING, HOLDING_AT, HOLDING_ON_TOP source | 0.20–0.60 | 0.20–0.60 |
| SIT source | 0.25–0.80 | 0.35–0.50 |
| CLIMB source | 0.40–0.80 | 0.25–0.50 |
| ON_TOP support | 0.50–0.80 | 0.25–0.45 |

- X/Y/Z는 각각 독립적인 5cm 격자 균등분포이고 밀도는 100kg/m³다. 큰 carry 상자는 최대 21.6kg이며, 범위 안의 모든 크기에서 행동 성공이 보장되는 것은 아니다.
- 물리 asset 크기는 scene 생성 때 확정한다. 먼저 과제 비율 **HOLDING/SIT/CLIMB/AT/ON_TOP = 5/10/25/30/30**의 혼합 크기 분포를 만들고, reset에서는 해당 크기가 허용하는 과제를 `p(task)/허용 크기 조합 수`에 비례해 다시 선택한다. 이는 전체 혼합 분포에서 목표 과제 비율과 과제별 균등 크기를 보존한다. 유한한 환경 pool의 실측 비율에는 표본 오차가 있다. 물리 상자 randomAssignment를 끄고 source 0/1·support 2/3을 고정해 실제 크기와 RSI 캐시의 owner 연결을 보존한다. 토큰·edge 입력 순열은 계속 사용한다. 고정 평가 preset은 처음부터 해당 과제 크기로 asset을 생성한다.
- Progress는 `1/(1+max(dxy-r_valid,0)/1m)`다. `r_valid=목표 상자 XY 반대각선+buffer`; buffer는 HOLDING 0.20m, SIT/CLIMB 0.30m, ON_TOP 0.10m다. AT는 점 목표여서 반대각선 없이 0.10m다. ON_TOP은 받침 크기를 사용한다.
- AT/ON_TOP RSI는 모두 **loco/pickUp/carryWith/putDown=40/10/40/10**. HOLDING·SIT·CLIMB의 기존 50/50 skill 비율은 유지한다. AMP expert 내부 skill 분포는 원본 unified와 같고 family 사전분포만 carry/sit/climb=65/10/25로 맞춘다.
- 학습 첫 로딩에서 실제 asset의 크기별로 SIT/CLIMB/pickUp/carryWith/putDown **전체 프레임**을 0.1초 물리 검사한다. ON_TOP putDown은 source·받침 크기 쌍과 reference 최종 XY에 둔 접지 받침을 함께 검사한다. 이 RSI 받침 회전은 identity다. 학습 reset 중에는 시뮬레이션 검사를 하지 않는다.
- 통과 조건은 모든 검사 step에서 finite, root 속도 변화 ≤3m/s, 상자 속도 ≤5m/s, root·상자 이동 ≤0.30m다. 미세 접촉만으로 탈락시키지 않으며 own_success를 초기화 필수 조건으로 요구하지 않는다. Reset에서는 다른 owner/남는 상자가 reference 몸을 깊게 침범하면 배치를 다시 뽑는다. 이 검사는 초기 충격 검사이고 장기 성공 보장은 아니다.
- SIT/CLIMB은 reference의 상자 접근·높이로 상호작용 구간을 추정한다. 마지막 상호작용 뒤의 퇴장 프레임은 배제하고, **70%는 유효 상호작용 프레임의 마지막 30%, 30%는 전체 유효 프레임**에서 선택한다. 후반 후보가 없으면 앞의 유효 프레임을 쓰며, 불연속 유효 구간을 유지한다. 가능한 clip끼리는 원래 clip 가중치를 사용한다.
- 요청 skill에 유효 프레임이 없으면 같은 과제의 다른 허용 skill, 최종적으로 loco를 선택한다. 실제 선택한 skill·motion ID·시간으로 humanoid pose/velocity, 담당 source, AMP history를 함께 초기화한다. humanoid pose의 크기 보정·IK는 하지 않는다. SIT/CLIMB source는 reference XY/회전과 실제 높이를, carry source는 reference 궤적과 바닥 높이 clamp를 사용한다.
- 캐시는 `output/rsi_cache/*.pkl.gz`에 저장한다. motion tensor·상자 reference·asset 파일·물리 설정·검사 기준·RSI 코드가 바뀌면 별도 캐시를 만든다. 기본 학습 SEED는 42이며 `SEED=...`로 변경할 수 있다. 같은 설정에 새 크기 조합이 생기면 누락 조합만 추가 검사한다. 첫 실행은 이 과정으로 오래 걸릴 수 있다.
- Size RSI 실행 경로는 고정 asset의 과제 배정 확률을 한 번 계산하고 reset graph를 GPU에서 배치 생성한다. 캐시 RSI 전에 기존 motion/time을 뽑아 버리던 중복 작업도 제거했다. 설정·분포·graph 의미·AMP 연결·기존 물리 캐시는 유지한다. 기존 unified 실행 경로는 그대로이며 새 프로세스부터 적용된다. 중복 난수 추출 제거로 같은 seed의 학습 궤적은 달라질 수 있다.
- TensorBoard `sampling/rsi_<task>_actual_<skill>`은 최종 수락된 reset의 실제 skill 비율, `sampling/rsi_<task>_fallback_attempt_fraction`은 전체 초기화 시도의 대체 비율이다. `sampling/rsi_ontop_putdown_fraction`도 확인한다. `sampling/rsi_<relation>_reference_success`와 `first_step_success`로 reference 초기 성공과 첫 물리 step 뒤의 성공을 비교한다. 기존 `own_success`는 실제 물리 rollout의 성공 지표다.

```bash
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=6 TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_vnc.sh "$CKPT" 2 1 4 10
```


## Unified size RSI typed bias

`approach_scenario_stage1_unified_size_rsi_typed_bias`는 위 Size RSI의 과제 분포·물리 크기·후반 RSI·캐시·AMP·보상·성공 조건·`r_valid`를 유지하고 semantic bias 생성부만 교체한 별도 scratch 실험이다.

- 전용 train config는 `amp_ma_carry_relation_unified_size_rsi_typed_bias.yaml`이며 `relation_bias_mode: typed_lookup`을 사용한다. 기존 레거시 `lookup`과 다르다. Env variant와 train mode가 맞지 않거나 actor/critic 표 공유·bias 비활성화를 지정하면 실행 전에 거부한다.
- `(source 타입, relation, target 타입)`으로 `[3,11,3,4,2]` 표에서 layer/head bias 8개를 직접 조회한다. 표는 0으로 초기화하고 PPO/critic 역전파로 학습한다. Actor·critic은 각각 792개 파라미터의 독립 표를 갖는다. 16/32/16 edge embedding·공유 edge MLP·projection은 이 실험에서 사용하지 않는다.
- 유효 edge의 방향·endpoint에 bias를 넣고, 나머지는 타입별 NONE·대각선 SELF를 사용한다. Edge 순열·토큰/GTA/bias 동시 순열·human readout 복원은 기존과 같다. Node tokenizer·GTA·attention·action/value head는 유지한다.
- 같은 타입/관계 조합은 agent·물체 슬롯과 관계없이 같은 표 값을 사용한다. 연속 관측에 따른 attention 변화는 기존 QK/GTA가 담당한다.
- Output은 `output/approach_scenario_stage1_unified_size_rsi_typed_bias`다. 기존 Size RSI checkpoint로 직접 resume/eval할 수 없다. 이 실험에서 저장한 checkpoint는 `RESUME_CHECKPOINT`로 재개한다. 기존 output과 실행 중인 학습을 변경하지 않는다.

```bash
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_train.sh 2 2048 4
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=5 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_vnc.sh "$CKPT" 2 1 4 10
```


## Unified size RSI typed bias + relation message

`approach_scenario_stage1_unified_size_rsi_typed_bias_message`는 typed-bias 실험에 관계 메시지만 추가한 별도 scratch 실험이다. Size RSI의 과제 분포·크기·후반 RSI 캐시·AMP·보상·성공 조건·`r_valid`를 유지한다.

- 기존 scalar bias 표 `[3,11,3,4,2]`와 QK→bias→softmax는 유지한다. 새 관계 표 `[3,11,3,64]`를 `(source 타입, relation, target 타입)`으로 조회하고 두 head에 32차원씩 나눈다. 표는 layer 간 공유하고 actor/critic은 분리한다. 추가 학습 파라미터는 각 6,336개, 합계 12,672개다.
- `relation_message: {enable: true, alpha: 1.0, init_std: 0.02}`가 전용 train config에 있다. 과제 relation은 작은 정규분포로 초기화하고 NONE/SELF·비과제 메시지는 0이다. 유효한 방향성 과제 edge에만 같은 attention 가중치를 곱해 source 노드로 합산한다. 기존 V 메시지의 GTA 복귀 변환 뒤, head 결합·output projection 전에 관계 메시지를 더한다.
- 저장된 edge만 gather/scatter로 처리하며 dense `[batch,head,node,node,32]`를 만들지 않는다. 토큰 셔플 때 src/dst도 함께 바꾸며 edge 순열·goal slot 교환에 대해 action/value/gradient가 유지되는지 검사한다.
- TensorBoard `relation_attention/{actor,critic}/layer{0..3}/node_message_rms`, `relation_message_rms`, `relation_to_node_rms`에 두 메시지의 크기를 기록한다. 기존 attention 진단과 동일하게 샘플된 관측에서 측정하며 relation 값에는 alpha가 적용된다.
- Env variant와 메시지 설정을 함께 검사한다. 기존 Size RSI/typebias checkpoint를 이 실험에 직접 resume/eval할 수 없다. 새 실험에서 생성한 checkpoint로만 `RESUME_CHECKPOINT` 재개한다. Output은 `output/approach_scenario_stage1_unified_size_rsi_typed_bias_message`다.

```bash
# Scratch 학습: GPU 점유를 확인한 뒤 지정
RESUME_CHECKPOINT='' TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_train.sh 2 2048 4
# 로컬 viewer / headless 평가 / 서버 VNC
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=5 TASK_GRAPH=holding_at EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_vnc.sh "$CKPT" 2 1 4 10
# 같은 실험 checkpoint 재개
RESUME_CHECKPOINT="$CKPT" TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_train.sh 2 2048 4
```


## Unified size RSI task message

`approach_scenario_stage1_unified_size_rsi_task_message`는 typed message 기반의 별도 scratch 실험이다. 과제를 `sit / climb / carry_at / carry_ontop`으로 입력하며, 각 agent의 사전 샘플링 비율은 **10 / 25 / 32.5 / 32.5%**다. 실제 고정 asset에는 기존 크기 조건부 샘플링을 적용한다. 단독 HOLDING 과제는 제외한다.

- 정책 suffix는 agent당 `[valid, task, actor, payload, target]` 한 레코드다. Task ID는 위 순서의 0~3, endpoint는 canonical token ID다. 2H/4O/2G에서 `568 scene + 56 pose + 10 task = 634`차원이며, suffix는 관측 정규화를 우회해 PPO sample에 그대로 저장한다.
- `carry_at(H,O,G)`와 `carry_ontop(H,O,S)`는 각각 actor→payload, actor→target, payload→target의 **단방향 3연결**을 만든다. 셋 모두 같은 task ID지만 역할 쌍으로 별도 조회한다. `carry_ontop(H,O,S)`의 O는 위에 놓을 물체, S는 받침이다. Sit/climb은 actor→target 한 연결이다. 역방향은 일반 QK attention과 NONE bias를 사용한다.
- Actor/critic 각각 bias `[4 task,3 src role,3 dst role,4 layer,2 head]`와 message `[4,3,3,64]` 표를 학습한다. 역할은 actor=0/payload=1/target=2다. NONE/SELF 배경 bias는 기존 물리 타입 기준 `[3,2,3,4,2]` 표다. 합계 각 **2,736개** 파라미터이며 각자 독립이다. Bias 0 초기화, message std=0.02·alpha=1, layer 공유·head별 32차원·GTA 복귀 뒤 가중합은 이전 message 실험과 같다.
- 정책 task packet은 현재 보상 graph의 owner와 endpoint에서 매번 구성한다. 보상 graph의 HOLDING+AT/ON_TOP은 유지하며 새 H→target 연결에는 별도 보상을 만들지 않는다. `self=1, teammate=0, edge_aggregation=self_sum`: 자기 primitive 보상을 합산한다. 최대 task 보상은 sit/climb 0.6, carry 1.2이며 carry를 2로 나누지 않는다. State/progress/success·r_valid·성공 saturation·패널티 식은 유지한다.
- 내부 template 슬롯은 기존 순서 `[HOLDING,SIT,CLIMB,HOLDING_AT,HOLDING_ON_TOP]`, 비율 `[0,.10,.25,.325,.325]`이다. RSI 프레임·물리 캐시와 AMP family/expert 연결은 기존 primitive graph를 사용한다. Carry family 총량 65%와 크기 분포가 같으므로 기존 asset 생성·RSI 캐시를 재사용한다. YAML의 HOLDING 크기/RSI 항목은 내부 공통 형식으로 남지만 단독 과제는 뽑지 않는다.
- Token 순열 시 bias 두 축·메시지 endpoint·GTA pose를 함께 바꾸고 human readout을 복원한다. Task/edge 레코드 순서에는 의존하지 않는다. `TASK_GRAPH=carry_at/carry_ontop`을 지원하고 기존 `holding_at/holding_ontop`은 별칭이다. `holding` 단독 preset은 거부한다.
- 전용 train YAML은 `amp_ma_carry_relation_unified_size_rsi_task_message.yaml`, mode는 `task_role_lookup`, output은 `output/approach_scenario_stage1_unified_size_rsi_task_message`다. Packet version 5와 별도 variant로 기존 checkpoint의 직접 resume/eval을 거부한다. 같은 새 실험 checkpoint로만 재개한다. 성공률 로그의 HOLDING/AT/ON_TOP은 계속 **보상 primitive** 기준이며 정책 task 분류와 구분한다.

```bash
# Scratch 학습
RESUME_CHECKPOINT='' TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_train.sh 2 2048 4
# 로컬 viewer / headless 평가 / carryWith 상태의 서버 VNC
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=5 TASK_GRAPH=carry_at EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_vnc.sh "$CKPT" 2 1 4 10
# 같은 실험 checkpoint 재개
RESUME_CHECKPOINT="$CKPT" TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_train.sh 2 2048 4
```

## BONES BVH → AMP 모션 변환 (데이터 준비 전용)

원본은 `tokenhsi/data/dataset_bones_dooropen/*.bvh`와 `dataset_bones_push/*.bvh`다. CPU에서 `phys_humanoid_v3`로 retarget하고 30FPS Poselib `ref_motion.npy`를 생성한다. 새 학습 실험 config가 아니며, 기존 학습 YAML·정책·보상은 변경하지 않는다.

```bash
TOKENHSI_CONDA_ENV=tokenhsi bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh; python tokenhsi/scripts/multi_agent/convert_bones_amp.py --skip-existing'
TOKENHSI_CONDA_ENV=tokenhsi bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh; python tokenhsi/scripts/multi_agent/validate_bones_amp.py'
TOKENHSI_CONDA_ENV=tokenhsi bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh; python tokenhsi/scripts/multi_agent/preview_bones_amp.py'
TOKENHSI_CONDA_ENV=tokenhsi bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh; python -m pytest -q tokenhsi/tests/test_bones_bvh.py'
```

`--dataset door|push`로 종류를 선택하고 `--clip <정확한 BVH stem>`을 반복해 일부만 변환할 수 있다. 일부 실행은 `_selected.yaml`만 기록해 전체 manifest를 덮어쓰지 않는다. `--skip-existing`은 원본·파서·변환기 SHA256와 FPS가 일치하는 결과만 재사용한다. 기본은 전체 clip이며 끝의 30FPS 격자 미만 구간만 잘린다.

Manifest는 [dataset_bones_door_amp.yaml](../tokenhsi/data/dataset_bones_door_amp.yaml)의 `doorOpen`, [dataset_bones_push_amp.yaml](../tokenhsi/data/dataset_bones_push_amp.yaml)의 `push`다. 모션별 결과는 `<dataset>/motions/<clip>/phys_humanoid_v3/`의 `ref_motion.npy`, `conversion_report.json`, `preview_data.json`이며 데이터셋별 `conversion_summary.json`에 모인다. 최대 손발 오차 >5cm, DOF 속도 >15rad/s, 중앙값 발 바닥 여유 >10cm인 clip은 보관하되 기본 manifest의 weight=0으로 샘플링에서 제외한다. 이 기준은 변환 검토 기준이며 과제 성공 판정은 아니다.

검증 결과·재생 페이지·시점별 그림은 `output/bones_amp_conversion_check/validation.json`, `preview.html`, `contact_sheet.png`다. HTML은 인터넷 없이 재생·모션 선택·scrub·시점·원본 관절 비교가 가능하다. 실제 multi-agent AMP builder의 인간 모션 부분은 129-D/step, 10-step은 1290-D다. 기존 Unified의 task-family 라벨 및 새 PUSH/OPEN expert binding은 별도 통합이 필요하다. 원본에는 문/상자 pose·접촉 정보가 없으므로 `object_state.npy`는 생성하지 않는다. 현재 humanoid는 손가락·손목 DOF가 없어 BVH 손가락 동작을 보존하지 않는다. 참조 데이터의 로드와 과제 학습·물리 접촉 검증은 구분한다.

## 왼쪽 힌지·오른쪽 손잡이 자동 닫힘 문 (물리 환경 준비)

`tokenhsi/data/assets/door/left_hinge_right_handle.urdf`와 재사용 `DoorFixture`를 추가했다. 정면의 사람은 local -X에 서서 +X를 본다. +Y가 왼쪽이고, 힌지는 왼쪽(+Y), 손잡이는 오른쪽(-Y)이다. 양의 Z 회전으로 문이 안쪽(+X)·왼쪽으로 열린다. 문짝은 0.9×2.1×0.045m, 손잡이는 1.05m 높이, 최대 개방은 110°다. 문틀은 고정이고 문짝·앞/뒤 손잡이는 다른 actor와 충돌한다.

문 힌지는 `target=0`, 기본 stiffness=6.0N·m/rad·damping=3.0N·m·s/rad의 compliant position drive로 복원 스프링을 만든다. 90°·정지에서 초기 닫힘 토크는 약 9.42N·m(손잡이 환산 약 11.59N)다. 손으로 잡고 있을 때도 닫힘 힘이 걸리며, 접촉을 제거하면 돌아온다. 잠금장치·손잡이 회전은 없다.

```bash
# 실행 전 GPU 점유 확인. 아래 환경 수는 학습이 아닌 물리 검증용이다.
nvidia-smi
TOKENHSI_GPU=5 HEADLESS=1 NUM_ENVS=4 OUTPUT_PATH=output/door_environment_check bash tokenhsi/scripts/multi_agent/door_environment_test.sh
# 로컬 viewer
TOKENHSI_GPU=5 HEADLESS=0 NUM_ENVS=1 REPEATS=3 bash tokenhsi/scripts/multi_agent/door_environment_test.sh
# 서버 VNC viewer (미사용 포트 선택)
TOKENHSI_GPU=5 PORT=6107 NUM_ENVS=1 REPEATS=3 bash tokenhsi/scripts/multi_agent/door_environment_vnc.sh
# 더 약한 복원력 예: stiffness만 변경하면 닫히는 시간도 달라진다.
TOKENHSI_GPU=5 HEADLESS=1 DOOR_STIFFNESS=3.0 DOOR_DAMPING=3.0 bash tokenhsi/scripts/multi_agent/door_environment_test.sh
TOKENHSI_CONDA_ENV=tokenhsi bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh; python -m pytest -q tokenhsi/tests/test_door_asset.py'
```

`door_environment.py`는 충돌 probe가 손잡이를 밀어 개방하고, probe를 치워 스프링 복귀를 검증한다. closed/45°/90° 손잡이 좌표, 회전 배치, 부분 reset, 초기 복귀를 함께 확인한다. `OUTPUT_PATH/physics_report.json`에 결과를, viewer에서는 `closed.png`, `pushed_open.png`, `released_closed.png`를 저장한다. 새 학습 YAML·reward·AMP·checkpoint 계약을 추가한 실험은 아니므로 train 명령은 아직 없다. 사람 정책이 문을 여는 학습은 별도 통합 대상이다.

`DoorFixture.add()`로 기존 Gym env에 문을 배치하고 `prepare_sim` 이후 `bind_tensors()`를 호출한다. `observe()`에서 angle/angular_velocity·문짝 pose·앞/뒤 손잡이 위치·base pose를 읽고 `reset(door_ids, angle=0)`로 선택한 문만 초기화한다. reset 뒤 새 body pose를 읽기 전에 물리 step/refresh가 필요하다. 사람의 PD target tensor를 함께 설정할 때 문 힌지 target은 0으로 유지해야 자동 닫힘이 유지된다. 기본 asset과 다른 치수는 `DoorSpec`와 `write_door_asset()`로 별도 생성해야 한다.

## Push + Door open/hold Stage 1

[환경 config](../tokenhsi/data/cfg/multi_agent/push_door_stage1.yaml)와 [학습 config](../tokenhsi/data/cfg/train/rlg/amp_ma_push_door_stage1.yaml)는 별도 scratch 실험이다. `HumanoidMAPushDoor`가 2명·상자 2개·문 2개를 배치하며, agent마다 PUSH/DOOR를 독립 50/50으로 배정한다. 문은 왼쪽 힌지·오른쪽 앞/뒤 손잡이, 6N·m/rad 스프링·3N·m·s/rad 감쇠로 자동 닫힌다. 학습은 2048환경이며 새 wrapper의 기본 GPU는 **6**이다.

- 모델: clean_scene unified Transformer(64-D, 4-layer, 2-head), actor/critic 별도 인코더·공유 agent action head·GTA. 한 scene에 H223×2/O34×4/T9×2/pose7×8 = **656-D**, agent당 32 actions. Object에 box/door 종류·힌지 각도/속도, Target에 task·개방/유지 단계·유지 시간·성공 상태·최고 진행도를 입력한다. Semantic edge MLP bias는 task에 따라 담당 box 또는 door와 goal에 연결된다. 기존 5과제 graph packet/message 실험의 checkpoint와 호환되지 않는다.
- 보상: 새 최대 진행도에 `0.4`, 현재 유지 조건에 매 step `0.6`, 연속 유지 성공 시 최초 한 번 `0.2`다. Push는 목표 XY 거리 감소(0.5m/s 기준), door는 80°까지 개방 진행(0.6rad/s 기준)이다. 최고 진행도를 기록해 왕복으로 보상을 재획득하지 못한다. 접근/손 거리/자세/power/동료 보상은 추가하지 않는다.
- PUSH 유지: 목표 XY 오차 <15cm, 상자 속도 <0.15m/s, upright(z축 >0.9), 회전 bbox 바닥 높이 절댓값 <4cm. 0.5초 연속이면 성공이다. 진행도도 grounded/upright일 때만 지급한다.
- DOOR 유지: 각도 ≥80°, 각속도 <0.3rad/s, 담당 손과 앞/뒤 손잡이 사이 <16cm이며 해당 손·해당 손잡이 양쪽 net contact force >0.5N. 2초 연속이면 성공이며, 이후에도 유지 보상을 계속 준다. 접촉 또는 개방 조건을 잃으면 연속 시간이 초기화된다. 성공 기록은 episode 안에서 유지한다. Isaac Gym net-force tensor는 collision pair를 제공하지 않으므로 거리와 양쪽 힘을 함께 쓰는 접촉 판정이다. 실제 grasp joint/magnet은 만들지 않는다.
- AMP: [motion manifest](../tokenhsi/data/dataset_push_door_stage1.yaml)의 push/doorOpen/loco 참조를 로드한다. PUSH=family0, door 개방=family1, door 유지=family2; **129 motion features+3 labels ×10 frames=1320-D**다. 기본 door 개방 reference는 clip phase 0~0.7, 유지 reference는 0.7~1.0이다. 이는 원본 문 각도 annotation이 없는 상태의 설정 가능한 시간 분할이다. 전체 AMP window를 선택 구간 내부에서 샘플링한다.
- 실제 문 각도 80°에서 유지 family로 전환하고 65° 아래에서 개방 family로 복귀한다. 물리 motion history는 보존하면서 전체 window의 label을 현재 phase로 통일하며, demo/replay는 같은 family와 매칭한다. `interaction.amp.hold_source: loco`를 새 config에서 선택하면 유지 단계 reference를 loco로 변경할 수 있다. 기본은 `door_tail`이다.
- 초기화: 물체 상태가 없는 BONES를 object RSI로 사용하지 않는다. loco 참조 자세를 +X heading으로 정렬하고 속도 0으로 사람을 가까이 배치한다. 상자는 바닥, 문은 닫힌 상태, push 목표는 앞쪽 0.7~1.1m다. Gym actor-state tensor의 env-local 좌표를 사용한다. 문을 열어 놓은 RSI/curriculum은 포함하지 않는다.
- checkpoint에 physics/reward/task 분포/AMP phase/좌표계 계약과 resolved env/train 설정을 저장하며 변경된 계약으로 resume/eval하는 것을 거부한다. `output/<run>/relation_config.yaml`, `nn/*.pth`, `summaries/`에서 설정·가중치·진단을 확인한다. 본학습 수렴은 짧은 실행 검증과 별개다.

```bash
# 본학습: 2명·2048환경·상자 2개+문 2개
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/push_door_stage1_train.sh

# 짧은 확인용 실행 (본학습 output과 분리)
TOKENHSI_GPU=6 MAX_ITERATIONS=1 OUTPUT_PATH=output/push_door_stage1_check bash tokenhsi/scripts/multi_agent/push_door_stage1_train.sh

CKPT=output/push_door_stage1/<run>/nn/PushDoorStage1.pth
# 화면 없는 평가 / 로컬 viewer: 인자는 checkpoint, envs, repeats
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=push_door bash tokenhsi/scripts/multi_agent/push_door_stage1_test.sh "$CKPT" 16 3
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=push_door bash tokenhsi/scripts/multi_agent/push_door_stage1_test.sh "$CKPT" 1 3
# 서버 viewer
TOKENHSI_GPU=6 PORT=6108 TASK_GRAPH=push_door bash tokenhsi/scripts/multi_agent/push_door_stage1_vnc.sh "$CKPT" 1 3
```

`TASK_GRAPH=random|push|door|push_door|door_push`로 평가 task를 고른다. `EPISODE_LENGTH`는 평가 길이, `RESUME_CHECKPOINT`는 본학습 resume, `OUTPUT_PATH`는 출력 위치를 바꾼다. 유지 성능은 평가 JSON의 `ever_success`와 `current_success`를 함께 확인한다.


## Push/Door Stage 1 random start (ZIP 서버 반영)

`push_door_stage1_random_start`는 제공된 ZIP과 같은 시작 분포를 사용한다. env의 `startRandomization`이 있을 때만 매 reset·agent별 scene/box XY·상자 yaw·사람 접근 거리/좌우/yaw·PUSH 목표 거리/방향을 샘플링한다. 문은 닫힘·수직, 초기 속도는 0이며 loco 참조 rigid body와 kinematic을 새 사람 pose에 맞춘다. 부분 reset과 env-local 좌표를 유지한다.

기존 `push_door_stage1`은 고정 시작 그대로다. 사용자 비교 실험 요청에 따라 새 YAML의 `interaction.amp.hold_source`도 기존 YAML과 동일한 `door_tail`로 되돌렸다. 문 개방은 door 모션 0~0.7, 유지는 후반 0.7~1.0 참조를 사용한다. 보상·성공·motion manifest·656-D 관측·32 actions·모델 구조는 동일하다. checkpoint metadata에 시작 분포가 선택적으로 포함되므로 고정/랜덤 checkpoint를 혼용할 수 없다. 별도 output에서 scratch 학습한다.

학습 2048환경, 기본 GPU 6, 기존 `runtime_env.sh`/MPS 연결을 사용한다. 아래 명령은 **나중에 사용할 명령이며 이번 작업에서는 실행하지 않았다**.

```bash
# 학습
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/push_door_stage1_random_start_train.sh
# 화면 없는 평가
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_door_stage1_random_start_test.sh <random_start_checkpoint.pth> 16 3
# 로컬 데스크톱 viewer
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/push_door_stage1_random_start_view.sh <random_start_checkpoint.pth> 1 10
# 서버 VNC
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/push_door_stage1_random_start_vnc.sh <random_start_checkpoint.pth> 1 3
```

전용 train 설정은 `tokenhsi/data/cfg/train/rlg/amp_ma_push_door_stage1_random_start.yaml`, output 기본값은 학습 `output/push_door_stage1_random_start`, 평가 `_eval`, 로컬 viewer `_local_viewer`, VNC `_viewer`다. 사용자 요청으로 CPU 테스트·문법·diff 검사 및 모든 GPU 실행을 생략했다. 서버 실행 검증은 아직 하지 않았다.


랜덤 시작 Push/Door는 기존 Unified와 동일하게 `agentCollisionPenalty: true`, `agentCollisionCoeff: 0.5`, `agentCollisionDist: 0.7`을 적용한다. 사람 root XY 거리 d에 대해 각 agent에 `-0.5 * max(0, (0.7-d)/0.7)`를 부과한다. 초기 lane 간격은 3.2m를 유지한다. 기본 task 보상에 패널티가 추가되며 `agent_collision_penalty` extras로 기록한다. 이 설정은 checkpoint 계약에 포함되므로 추가 전 checkpoint와 혼용하지 않는다. 기존 고정 시작 실험에는 적용하지 않는다. 테스트/학습은 실행하지 않았다.


## DOOR 참조 방향 제한

공통 Push/Door config의 `env.interaction.door.motion_direction: left_open`은 BONES `left_side` 원본과 `right_side` 미러(`_M`)만 로드한다. `left_side` 미러와 `right_side` 원본은 제외한다. 현재8개/36.233초이며 기존 weight=0 검토 제외는 유지한다. MotionLib 로드 단계에서 필터링하므로 AMP 개방/후반 참조와 같은 라이브러리를 쓰는 RSI에 함께 적용된다. 좌측 개방(+Y) 손 궤적을 대표 A512 원본/미러에서 확인했다. 방향 설정은 checkpoint 계약에 포함되며 이전 방향 혼합 checkpoint와 구분한다. 실행 중인 프로세스에는 자동 반영되지 않는다.

## 과제 RSI와 작은 각도 DOOR 시작 (서버 공통)

`push_door_stage1_task_rsi.yaml`과 `amp_ma_push_door_stage1_task_rsi.yaml`을 사용하는 별도 실험이다. 기존 fixed/random_start 설정은 loco 초기화를 유지한다. PUSH는80%, DOOR는50%가 과제 참조 RSI로 시작한다. DOOR 나머지50%는 문에서1.2~1.8m 떨어진 loco 자세·닫힌 문으로 시작한다. 선택된 참조 구간의 후반 절반에는70%를 배정한다.

DOOR RSI의75%는5~25° 초기 각도를 사용한다. 해당 각도에서 문틀 여유를 확보하는 자세 pool을 먼저 샘플한 뒤 안전 각도를 고르므로 문틀 검사 때문에 큰 각도로 밀려나지 않는다. 나머지25%는5~79° 전체 안전 각도에서 뽑는다. 따라서 실제25° 이하 비율은75%보다 높을 수 있다. 초기 각도는 BONES의 문 annotation에서 추출한 값이 아니라 합성 값이다. 손잡이 정렬과 몸/팔다리의 보수적 문틀 여유3cm를 검사하며 불가능한 참조 자세는 제외한다. 이후 정책 동작 중 끼임까지 보장하지 않는다. PUSH는 공통1.1m 정육면체와 손 간격3cm, 남은 거리0.2~0.6m를 사용한다.

서버 실행(기존 GPU6/MPS runtime 유지):
```bash
bash tokenhsi/scripts/multi_agent/push_door_stage1_task_rsi_train.sh
# 짧은 확인 학습은 명시적으로 선택
MAX_ITERATIONS=1 OUTPUT_PATH=output/task_rsi_check bash tokenhsi/scripts/multi_agent/push_door_stage1_task_rsi_train.sh
bash tokenhsi/scripts/multi_agent/push_door_stage1_task_rsi_test.sh <task_rsi_checkpoint.pth>
bash tokenhsi/scripts/multi_agent/push_door_stage1_task_rsi_view.sh
```

view는 학습된 정책이 아닌 초기 자세 preview다. R 재샘플/Esc 종료, 기본은 현재 학습 혼합비(PUSH RSI80%, DOOR 접촉RSI50%/원거리 접근50%)다. `RSI_PREVIEW_PROBABILITY=1`로 모두 접촉RSI, `RSI_PREVIEW_PROBABILITY=0`으로 모두 접근 시작을 본다. GUI가 필요한 preview는 서버 DISPLAY/VNC 환경에서 실행한다. 학습·평가는 기존 manifest와 BONES 데이터가 필요하다. 새 초기화 설정은 checkpoint 계약에 포함되므로 이전 로컬v2 또는 다른 실험 checkpoint를 새v3 config로 resume하지 않는다. 새 학습을 시작하거나 기존 checkpoint에 맞는 설정을 사용한다.

로컬 GPU0/NVRTC/MPS 우회는 여전히 Git 제외 `.local/run.sh`에서만 처리한다. 로컬 확인도 이제 공통 task와 공통 config를 사용한다. 서버 runtime_env.sh/MPS 설정은 수정하지 않았다. 출력·데이터·로컬 runtime은 전송 대상에서 제외한다.

PUSH RSI v4: 양손을 연결한 XY 방향에 수직인 상자 면 법선으로 몸 방향을 정렬해 양손 깊이를 같게 한다. 실제 asset 골격 길이와 관절값으로 참조 몸 위치를 계산한다. `push_hand_radius: 0.04`, `push_surface_gap: 0.001`로 손 중심 대신 손 충돌 표면 기준1mm 간격을 둔다. `hand_gap: 0.03`은 DOOR 정렬에만 사용한다. RSI 리셋 시 해당 사람의 PD 목표를 초기 관절값으로 설정해 화면 반영용 물리 step에서 이전/0 목표 때문에 손이 떨어지지 않도록 한다. 정책의 첫 action은 정상적으로 새 목표를 설정한다. 설정 계약은 `task_both_push_hands_contact_v4`이며 기존v3 checkpoint와 혼용하지 않는다. 실행 중인 학습에는 코드 변경이 소급 적용되지 않는다.

2026-10-08: 공통 fixed/random_start/task_rsi PUSH 상자 마찰을0.6→0.1로 낮췄다. 바닥 마찰1.0과 각 config의 기존 질량은 유지한다. CPU PhysX에서0.6/0.3/0.1/0.05/0.0 후보를 비교했으며 작은 감소의 효과는 제한적이었다. 높은 접촉점과 강한 수평 힘에서 전도는 여전히 발생하므로 마찰 감소를 전도 해결로 해석하지 않는다. 결과 output/push_box_friction_probe/report.json. 실행 중 프로세스에는 자동 반영되지 않으며 interaction metadata가 바뀌므로 이전 마찰의 checkpoint와 새 config를 혼용하지 않는다.

2026-10-08 최신 PUSH 질량: 공통 fixed/random_start/task_rsi 상자를30kg으로 통일했다. 크기1.1m 정육면체, density=30/(1.1³)≈22.539444㎏/㎥, friction=0.1이다. 이전15.75/34.61kg 설정을 대체한다. 질량 증가는 전도와 미끄러짐에 필요한 힘을 모두 높이며 높은 손 위치의 전도까지 해결하지 않는다. 새 설정은 다음 실행부터 적용되며 이전 질량 checkpoint와 새 config는 혼용하지 않는다.

## DOOR 보상 handle_gated_v2 (서버 공통, 2026-10-08)

공통3개 env config의 door.reward_version=handle_gated_v2. 개방 최고각도 증가 보상(최대0.4/step)은 손잡이 접촉 조건을 만족할 때만 받는다. 접촉 없이 열린 최고 기록도 갱신해 나중에 잡거나 닫았다 다시 열어 소급 보상을 받을 수 없다. 접근 보상은 손-앞/뒤 손잡이 최소 거리가 이전 최저 기록보다 감소한 양을dt×0.5m/s로 정규화해 최대0.1/step이다. 문 각도 변화가0.01rad를 넘으면 접근 보상을 주지 않으며 최저 거리 기록은 갱신한다(문 자체가 손 쪽으로 움직여 얻는 보상 제한). 손잡이 접촉 중에는 접근 보상을 주지 않는다.

닫힘 패널티는 직전 각도보다 닫힌 양에서0.005rad의 작은 진동 여유를 빼고dt×0.6rad/s로 정규화해 최대-0.2/step이다. 스프링 때문에 자연스럽게 닫히는 것도 대상이다. 기존80°/각속도0.3rad/s 미만/손잡이 접촉 유지 보상0.6과 연속2초 성공 보너스0.2는 유지한다. 접촉은 손잡이16cm 이내의 해당 손과 손잡이에 모두0.5N 초과 힘이 있는 근사 검사이며 정확한 충돌 쌍이나 grasp를 증명하지는 않는다.

리셋에서 RSI 초기 문 각도/손잡이 거리로 직전 각도·최저 거리 기록을 초기화한다. 기존PUSH 보상과 task/AMP 혼합비0.5/0.5는 유지한다. TensorBoard reward_terms에 door_approach, door_closing을 추가했다. 새 보상 설정은 checkpoint 계약에 포함되므로 이전 보상 checkpoint와 새 config는 혼용하지 않는다. 기존 서버 실행법은 그대로 사용한다. 실행 중인 프로세스에는 수정이 자동 적용되지 않는다.

### PUSH 동선을 문 반대쪽으로 변경 (2026-10-08)

fixed/random_start/task_rsi의 `interaction.push.direction: away_from_door`는 PUSH 목표와 사람 시작 위치를 상자 중심으로 반 바퀴 회전한다. 상자는 기존 위치를 유지하고 사람은 문 쪽에서 상자를 -X 방향으로 민다. 상자 방향도 회전하여 RSI의 목표 재샘플링이 문 반대 방향을 유지한다. DOOR 배치와 보상·모델·AMP는 유지한다. 방향 설정은 checkpoint 계약에 포함되어 이전 방향 checkpoint와 혼용하지 않으며, 새 설정으로 scratch 학습한다. 실행 중 프로세스에는 적용되지 않는다.

### 방향 변경 전 checkpoint 잠시 보기

random_start test/view/VNC에 `CFG_ENV`로 별도 env YAML을 전달할 수 있다. 기본값은 현재 학습 YAML 그대로다. epoch 3500 (`PushDoorStage1RandomStart_08-01-39-16`)의 원래 interaction·시작 분포·충돌 설정을 checkpoint metadata에서 복원한 viewer YAML은 `/home/hwanhee/juan/CVPR2027/runs/checkpoint_views/push_door_random_start_08-01-39-16_3500.yaml`이다.

```bash
CFG_ENV=/home/hwanhee/juan/CVPR2027/runs/checkpoint_views/push_door_random_start_08-01-39-16_3500.yaml \
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/push_door_stage1_random_start_vnc.sh \
  output/push_door_stage1_random_start/PushDoorStage1RandomStart_08-01-39-16/nn/PushDoorStage1RandomStart_00003500.pth 1 3
```

이 명령에만 `CFG_ENV`를 지정하며 셸 전체에 export하지 않는다. 다른 checkpoint는 해당 metadata와 일치하는 YAML이 필요하다. 현재 학습의 문 반대 방향은 유지한다.

### 보상 원복·손잡이 높이 조정 (2026-10-08)

사용자 요청으로 직전 PUSH 양손 보상/진행 감쇠 및 DOOR 개방 강화/각도 보상을 제거했다. 원래 PUSH 진행0.4·정착0.6·성공0.2와 DOOR handle_gated_v2 보상으로 돌아갔다. 문 반대 PUSH 동선은 유지한다.

공통 fixed/random_start/task_rsi의 `interaction.door.handle_height: 0.95`로 손잡이를 기존1.05m에서10cm 낮췄다. 명치 부근으로 낮추려는 요청의 첫 조정값이며, 실제 agent 자세에 따른 명치 높이는 viewer로 확인해 조정한다. 이 설정은 물리 URDF·앞/뒤 손잡이 관측·RSI 높이 선택/정렬에 연결된다. 높이가 다른 asset은 `runs/door_assets/`에 spec별로 생성하여 원본 URDF와 구 viewer를 보존한다. 높이 키가 없는 구 checkpoint viewer는 기존1.05m를 사용한다. 변경된 geometry 계약으로 구 checkpoint와 현재 config 혼용은 거부한다. 진행 중 실행에는 적용되지 않는다.

CPU27테스트 및 CPU PhysX1-step에서 앞/뒤 손잡이 실제 높이0.95m 검증 통과. 전체 agent RSI·학습 동작은 미검증이며 GPU/MPS를 사용하지 않았다.

### 로컬 TaskRSI 5500 checkpoint 보기

`PushDoorStage1TaskRSI_00005500.pth`는 원래 task RSI 초기화와 변경 전 PUSH/DOOR 보상을 사용한다. `CFG_ENV=/home/hwanhee/juan/CVPR2027/runs/checkpoint_views/push_door_task_rsi_from_local_5500.yaml`을 명령 앞에 지정하고 random_start VNC wrapper에 해당 checkpoint를 전달한다. 두 train YAML의 네트워크 설정은 동일함을 확인했다. 이전3500전용 YAML과 혼용하지 않는다. 현재 학습 설정은 유지한다.

### DOOR RSI 정면·참조 손 일치 필터 (2026-10-08)

현재 task_rsi 초기화 계약은 `task_door_loco_approach_v8`다. `door_facing_degrees: 60`으로 골반과 torso의 정면 XY 방향이 손잡이 방향에서 모두60° 이내인 후보만 허용한다. 기존 문틀 여유·손 높이·팔 reach 조건도 함께 적용한다. PUSH 초기화와 보상은 유지한다.

`door_hand_binding: user_paths_v1`은 사용자가 확인한 손 라벨을 사용한다. `left_side` 원본은 오른손(0), `right_side` 미러는 왼손(1)이다. 높이에 가까운 반대 손으로 바꾸지 않으며 해당 손의 높이/reach/정면/문틀 조건을 만족하는 frame만 남긴다. 이전 reach 추정 라벨을 대체했다.


CPU 실제 DOOR8모션 검사에서 사용자 손 라벨 적용 후124프레임이 남았고 오른손·왼손 모두 존재한다. 후보/clip별 손은 `/home/hwanhee/juan/CVPR2027/runs/rsi_checks/door_facing_hands.json`, 대표 top-view 자세는 같은 디렉터리의 `door_facing_hands.png`다. preview 로그의 `[RSI hand]`는0=오른손,1=왼손,-1=non-RSI다. 기존 view 명령과 R 재샘플을 사용한다. AMP도 아래 사용자 손 조건화5-family를 사용한다. 기존 viewer YAML은 새 필터 키가 없어 기존 RSI 계약을 유지한다. 새 설정과 구checkpoint 계약은 불일치한다.

### DOOR 접근 시작 혼합 (2026-10-08)

`push_door_stage1_task_rsi`에만 DOOR의 접촉 참조RSI 확률 `interaction.task_rsi.door_rsi_probability: 0.5`를 적용한다. 나머지50%는 기존 loco 참조와 닫힌 문으로 시작하며 `startRandomization.door_distance: [1.2, 1.8]`에서 문 plane까지 X 방향 거리를 샘플한다(손잡이까지의 유클리드 거리와는 다름). 사람은 손잡이 쪽을 보도록 기존 heading/lateral 분포를 사용한다. PUSH RSI80%와 기타 실험의 초기화는 유지한다. 기존 DOOR 접근 보상을 사용하며 보상 자체는 수정하지 않았다. AMP는 원거리 접근에 별도 loco family를 사용하고 손잡이 근처에서 손별 doorOpen으로 전환한다.

RSI preview는 기본으로 학습 분포를 따른다. 명령 앞의 `RSI_PREVIEW_PROBABILITY=0`은 떨어진 시작만, `=1`은 손잡이 접촉RSI만 보여준다. R로 재샘플해 비교한다. 기본50:50은 확률적이며 소수 reset에서는 같은 종류가 연속될 수 있다. 새 계약v6으로 구checkpoint 혼용은 거부한다. 변경은 새 실행에 적용된다. CPU30테스트 통과, 실제 걷기/문 개방은 아직 학습으로 검증하지 않았다.

### 사용자 확인 손 라벨과 AMP family (2026-10-08)

`interaction.amp.hand_conditioning: user_hands_v1`을 공통 fixed/random_start/task_rsi에 적용했다. 라벨0=PUSH,1=오른손 개방,2=왼손 개방,3=오른손 유지,4=왼손 유지,5=DOOR 접근(loco)이다. `left_side` 원본=오른손, `right_side` 미러=왼손으로 전문가 clip을 고르고 demo/replay도 같은 phase/손 family로 매칭한다. 현재 접근 loco 포함6-family AMP 입력은1350-D이며(이전3종1320-D,5종1340-D) 정책/critic 관측과 action 차원은 유지한다. 정규화에서 현재6개 one-hot을 보존하고 물리 history 전체에 현재 family를 적용한다. 다른 task의 기본3-family는 유지한다.

접촉RSI는 선택한 clip의 사용자 손 라벨로 시작한다. 접근 시작은 좌우를 랜덤 지정하고, 실제 손잡이에 손이 닿으면 거리+손/손잡이 양쪽 힘 조건으로 접촉 손을 갱신한다. 손 전환 시 AMP 참조/replay도 전환하며 두 손 접촉은 더 가까운 손을 따른다. 손별 정책을 따로 만드는 변경은 아니다. `hold_source:loco`를 사용하는 구/다른 설정은 유지 참조를 공유하지만 현재 설정은 door_tail이므로 유지도 손별 clip을 쓴다.

검증: CPU29테스트 통과(기존 Unified3-family 회귀 포함), 실제8모션 사용자 라벨 일치·RSI124frame·오른손/왼손 각2048 AMP clip 샘플의 손 일치·phase window 확인. 보고서 `/home/hwanhee/juan/CVPR2027/runs/rsi_checks/door_user_hands_amp.json`. GPU 학습/물리 및 손 전환 동작은 미실행. 구checkpoint는 현재6-family와 호환되지 않으며 기존 viewer YAML로만 본다. 새 설정은 scratch 학습한다.

### 거리 기반 DOOR 접근 loco → 개방 AMP (2026-10-08)

공통 fixed/random_start/task_rsi의 `interaction.amp.approach_loco`를 사용한다. agent root와 앞/뒤 손잡이 중 가까운 손잡이의 XY 거리가 `enter_door_distance: 0.8`m 이하면 손별 개방 family로 들어가고, 다시 `return_loco_distance: 1.0`m 밖이면 접근 loco family로 돌아간다. 중간 구간은 이전 상태를 유지한다. 실제 손잡이 접촉이나 개방 유지 phase는 거리보다 우선해 DOOR AMP를 유지한다. reset에서 거리 기준을 초기화하고 물리 step마다 갱신한다.

접근 family5의 전문가 source는 `loco`와 phase0~1이며 `hold_source`와 무관하다. 개방은 사용자 확인 손별 `doorOpen`, 유지는 `hold_source: door_tail`을 유지한다. demo/replay도6-family로 매칭하고 현재 phase 라벨은 전체 물리 history에 반영한다. 보상·actor/critic 정책 관측·action은 유지하고 AMP 입력은1350-D다. 설정은 interaction checkpoint 계약에 포함되어 기존3/5종 checkpoint와 혼용하지 않는다. 학습 명령은 같은 task_rsi_train.sh이며 새로 scratch 실행한다. 구 viewer YAML은 기존3/5-family를 유지한다.

CPU31테스트 통과: 거리 진입/복귀 히스테리시스·접촉/유지 우선·6종/loco demo/replay·one-hot 보존·기존3종 회귀. GPU/전체물리 학습은 실행하지 않았으며 실제 접근 성능은 새 학습에서 확인해야 한다.

### 새 TaskRSI 정책 VNC viewer

RSI 초기 자세만 보는 `push_door_stage1_task_rsi_view.sh`와 달리, `push_door_stage1_task_rsi_vnc.sh`는 현재 TaskRSI checkpoint의 실제 정책을 실행한다. GPU/MPS는 기존 공통 wrapper를 그대로 사용한다.

```bash
TOKENHSI_GPU=6 PORT=6081 bash tokenhsi/scripts/multi_agent/push_door_stage1_task_rsi_vnc.sh \
  output/push_door_stage1_task_rsi/PushDoorStage1TaskRSI_08-12-17-28/nn/PushDoorStage1TaskRSI_00000500.pth 1 10
```

500 checkpoint와 현재 config 계약 일치를 CPU에서 확인했다. 브라우저는6081포워딩 후 `http://localhost:6081/vnc.html?autoconnect=1&resize=remote`에 접속한다. 출력은 `output/push_door_stage1_task_rsi_viewer`다. 구checkpoint 복원용CFG_ENV는 필요 없다.

### TaskRSI 접근 보상 소폭 증가 (2026-10-08)

현재 `push_door_stage1_task_rsi.yaml`의 `interaction.door.approach_weight`만0.1→0.15로 올렸다. 기존 손-손잡이 최소거리 기록 개선 조건·AMP 거리전환·기타보상은 유지한다. fixed/random_start 계수는0.1이다. 변경은 새 실행부터 적용되며 진행 중 학습에는 반영되지 않는다. reward계수가 checkpoint 계약에 포함되어 기존 checkpoint는 원래viewer설정으로 확인해야 한다.

### PUSH 손 위치 보상 복구 (2026-10-08)

사용자의 원복 요청은 보상 크기에 관한 것이었으므로 현재 task_rsi의 PUSH 손 위치 항을 복구했다. 양손 중 낮은 표면 정렬품질 q를 사용해 `0.15*q*(기존 PUSH 진행 + 목표 정착 조건)`을 더한다. 상자 로컬좌표에서 미는 측면·손 반경4cm·높이60~95%·거리 scale15cm를 사용해 손이 올라가거나 한손이 벗어나면 추가보상이 줄어든다. 실제 접촉힘 gate는 아니다. 이전에 함께 넣었던 진행보상 감쇠는 추가하지 않아 기존 progress는 유지한다. DOOR 접근0.15·원래 개방0.4·높이0.95m·AMP거리전환을 유지한다. fixed/random_start에는 손 항을 추가하지 않았다. 새 실행부터 적용하며 기존 checkpoint와 현재 보상계약은 달라진다.

### 학습 iteration 로그

Multi-agent 학습은 각 iteration 시작에 `[train] iter=<epoch_num> frames=<누적 프레임> (starting)`을 출력한다. checkpoint의 epoch와 같은 번호이며 resume 시 이어진 번호를 사용한다. print_stats가 켜진 rank0에서 flush하여 출력한다. 새 프로세스부터 적용된다.


### PUSH 박스 크기 지정 (2026-10-08)

현재 TaskRSI 기본 박스는 1.3×1.3×1.3m, 질량30kg이다. 다른 PushDoor 설정 기본값은 유지한다.
`BOX_SIZE`는 train/test/RSI view에 전달되며 VNC에서도 동일하게 지정한다. 직접 실행은 `--push_box_size 1.3`을 쓴다.
크기를 변경하면 밀도와 랜덤 시작의 최소 접근 간격을 자동 조정한다. 손 목표는 박스 크기에 따라 계산된다.

```bash
cd /home/hwanhee/juan/CVPR2027/stage1_fix
BOX_SIZE=1.3 TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/push_door_stage1_task_rsi_train.sh
```

평가/view에도 학습 때와 같은 `BOX_SIZE`를 지정한다. 크기가 다른 checkpoint는 계약 검사에서 거부되므로 새 크기는 새 학습으로 시작한다. 실행 중 학습에는 반영되지 않는다.
