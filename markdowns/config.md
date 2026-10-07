# Multi-Agent Carry 실행 가이드

현재 실행 가능한 실험은 원본 1번과 Stage 1 **27·28·30·31·32·33·34·35·36·37번 및 paired/unified 변형**, Stage 2 **29번 및 SIT plane 변형**이다. 과거 9~26번 config와 전용 실행 스크립트는 정리했다. 모든 명령은 저장소 루트에서 실행한다.

## 빠른 확인

- Distill 실행: [carry 두 과제](#carry-task-embedding-distillation), [네 과제 전체](#four-task-embedding-distillation). 각각 별도 config/output/checkpoint를 사용한다.
- 최근 비교 대상은 **Task message / 공유 Task MLP / Task MLP split / Task embedding**이다. [네 실험 공통 설정과 차이](#task-4종-공통-설정과-비교), [최근 확인 기록](#task-3종-최근-확인-기록)을 먼저 확인한다.
- 실행 명령: [Task message](#unified-size-rsi-task-message), [공유 Task MLP](#unified-size-rsi-task-mlp), [Task MLP split](#unified-size-rsi-task-mlp-split), [Task embedding](#unified-size-rsi-task-embedding). 네 실험 모두 **2명·2048환경·4물체**, 전용 train/test/VNC와 output을 사용한다. 서로의 checkpoint를 resume/eval에 섞지 않는다.
- `split`은 **task별 concat 이후 MLP와 layer/head projection 분리**다. 역할 embedding은 공유하며, actor/critic은 독립이다. 두 MLP 실험 모두 관계 message는 없다.
- `task_embedding`은 역할 입력·중간 MLP·message 없이 task/NONE/SELF의 6개 embedding과 H/O/G 타입 쌍별 projection을 사용한다. Actor/critic은 독립이며 각 branch 안에서 모든 카테고리가 projection을 공유한다.
- 학습 중 HOLDING/AT/ON_TOP 성공률은 보상 primitive 지표다. 정책의 네 task 및 별도 평가의 운반 완수율과 구분한다. Bias는 원래 값과 같은 물리 타입의 NONE 대비 차이를 구분한다.
- 현재 GPU/PID·epoch는 문서의 과거 기록으로 판단하지 않는다. 실행 전 `nvidia-smi`, 실행 중인 `tokenhsi/run.py` 인자와 해당 run의 `relation_config.yaml`·`summaries/`·`nn/`을 확인한다.

## 공통 규칙

- 학습 인자는 `[num_agents] [num_envs] [num_objects]`; Stage 1 기본값은 `2 2048 3`이다. 짧은 확인도 환경 수 2048을 유지하고 `MAX_ITERATIONS`만 줄인다.
- 실행 전 `nvidia-smi`로 점유를 확인하고 명령 앞에 `TOKENHSI_GPU`를 지정한다. 실행 중인 프로세스는 임의 종료하지 않는다.
- 본학습 전 셸의 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 확인한다.
- 학습은 기본적으로 scratch다. 기존 실험의 reward/checkpoint 계약을 섞지 않는다. Stage 2 전이는 아래의 `STAGE1_CHECKPOINT`를 사용한다.
- 평가·VNC 인자는 `<checkpoint.pth> [agents] [envs] [objects] [repeats]`다. `HEADLESS=0`은 로컬 viewer, `HEADLESS=1`은 화면 없는 평가다.
- 로컬 viewer와 서버 VNC의 상자 색은 매 reset의 과제 배정을 따른다. 단독 대상은 해당 에이전트 색, 공동 대상은 노란색, 나머지는 회색이다. 상자 정리 데모는 모든 상자를 같은 회색으로 표시한다. AT goal 점도 해당 edge 담당 에이전트 색이며, goal 슬롯이 바뀌어도 배정을 따른다. 실행 중인 뷰어는 다시 시작해야 반영된다.
- 공유 데이터 원본은 읽기 전용으로 취급하고 이 저장소에서는 심링크로 사용한다. 현재 로컬 링크는 `/home/cvlab/Desktop/CVPR2027/TokenHSI`로 연결된다(2026-10-07, 12개 링크 모두 유효). 서버 문서의 `/home/hwanhee/CVPR2027/TokenHSI`와 경로가 다르므로 다른 머신에서는 실제 링크 대상을 확인한다. RSI 캐시는 이 저장소의 `output/rsi_cache`다.
- 4명·16상자 정리 데모는 아래 **상자 정리 VNC 데모** 절을 따른다. 지정된 rescue checkpoint를 사용하는 평가 전용 실행이다.

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
| Unified size RSI task MLP | [approach_scenario_stage1_unified_size_rsi_task_mlp.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp.yaml) | Task message와 동일한 과제·RSI·AMP·자기 보상, task·역할 MLP bias만 사용 |
| Unified size RSI task MLP split | [approach_scenario_stage1_unified_size_rsi_task_mlp_split.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split.yaml) | Task MLP의 역할 embedding 공유, task별 MLP·projection 독립 |
| Unified size RSI task embedding | [approach_scenario_stage1_unified_size_rsi_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding.yaml) | Task/NONE/SELF embedding과 H/O/G 타입 쌍별 projection, 중간 MLP·메시지 없음 |
| Carry task embedding distill | [approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill.yaml) | Carry AT/ON_TOP 각 50%, 타입 쌍 projection 유지, 동결 unified teacher KL |
| Four-task embedding distill | [approach_scenario_stage1_unified_size_rsi_task_embedding_distill.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill.yaml) | Sit/climb/carry_at/carry_ontop 원래 분포 유지, 동결 unified teacher KL |

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
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_train.sh 2 2048 4  # Unified size RSI task message
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_train.sh 2 2048 4  # Unified size RSI task MLP
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_train.sh 2 2048 4  # Unified size RSI task MLP split
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_train.sh 2 2048 4  # Unified size RSI task embedding
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_train.sh 2 2048 4  # Carry task embedding distill
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_train.sh 2 2048 4  # Four-task embedding distill
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
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_test.sh "$CKPT" 2 1 4 10  # Unified size RSI task message
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_test.sh "$CKPT" 2 1 4 10  # Unified size RSI task MLP
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_test.sh "$CKPT" 2 1 4 10  # Unified size RSI task MLP split
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_test.sh "$CKPT" 2 1 4 10  # Unified size RSI task embedding
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_test.sh "$CKPT" 2 1 4 10  # Carry task embedding distill
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_test.sh "$CKPT" 2 1 4 10  # Four-task embedding distill
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
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_vnc.sh "$CKPT"  # Unified size RSI task message
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_vnc.sh "$CKPT"  # Unified size RSI task MLP
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_vnc.sh "$CKPT"  # Unified size RSI task MLP split
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_vnc.sh "$CKPT"  # Unified size RSI task embedding
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_vnc.sh "$CKPT"  # Carry task embedding distill
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_vnc.sh "$CKPT"  # Four-task embedding distill
TOKENHSI_GPU=3 PORT=6080 bash tokenhsi/scripts/multi_agent/box_cleanup_demo_vnc.sh  # 4명·16상자 정리 데모
```

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


## Task 4종 공통 설정과 비교

아래 네 실험의 환경 설정은 checkpoint 격리용 `stage1_variant`만 다르다. Task bias/message 표현을 비교하며 과제·보상·크기·RSI·AMP와 634-D 관측을 유지한다.

| 항목 | Task message | 공유 Task MLP | Task MLP split | Task embedding |
| --- | --- | --- | --- | --- |
| 실험명 접미사 | `task_message` | `task_mlp` | `task_mlp_split` | `task_embedding` |
| Train mode | `task_role_lookup` | `task_role_mlp` | `task_role_mlp_split` | `task_type_embedding` |
| Task bias | task·출발/도착 역할·layer/head별 표 | 역할/task embedding → 공유 MLP → 공유 projection | 역할/task embedding → task별 MLP → task별 projection | task embedding → 물리 타입 쌍별 projection |
| 관계 message | task·역할별 학습 벡터 표 | 없음 | 없음 | 없음 |
| NONE/SELF 배경 | 물리 타입·관계·물리 타입별 표 | 별도 EdgeEncoder MLP | 별도 EdgeEncoder MLP, task별 분리하지 않음 | NONE/SELF embedding → task와 공유하는 타입 쌍별 projection |
| Actor/critic | 독립 | 독립 | 독립 | 독립 |

네 이름 앞에는 `approach_scenario_stage1_unified_size_rsi_`가 붙는다. Env YAML은 `tokenhsi/data/cfg/multi_agent/`, train YAML은 `tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_<접미사>.yaml`, wrapper는 `tokenhsi/scripts/multi_agent/<실험명>_{train,test,vnc}.sh`다. Output은 `output/<실험명>/<run>/`이며 저장 설정은 `relation_config.yaml`, 모델은 `nn/`, TensorBoard는 `summaries/`에 있다.

### Task와 edge

H=actor(사람), O=payload(운반 상자), G=바닥 target, S=받침 target이다. Sit/climb의 O는 payload가 아닌 target 역할이다. 각 agent가 독립 배정받으며 비율은 크기 조건부 샘플링 전의 사전 확률이다.

| Task ID | 정책 task | 사전 비율 | 정책의 단방향 연결 | 보상·RSI·AMP에 사용하는 primitive |
| ---: | --- | ---: | --- | --- |
| 0 | `sit` | 10% | H→O(target) | SIT |
| 1 | `climb` | 25% | H→O(target) | CLIMB |
| 2 | `carry_at` | 32.5% | H→O, H→G, O→G | HOLDING(H→O) + AT(O→G) |
| 3 | `carry_ontop` | 32.5% | H→O, H→S, O→S | HOLDING(H→O) + ON_TOP(O→S) |

Carry의 세 연결에는 **모두 같은 `carry_at` 또는 `carry_ontop` task ID**가 들어간다. 역할 쌍 `(actor,payload)`, `(actor,target)`, `(payload,target)`이 다르므로 bias도 달라질 수 있다. Split에서도 한 task 내부 세 연결은 같은 전용 MLP·projection을 쓴다. 단독 HOLDING task는 없고 추가 H→target 연결에는 보상을 중복 부여하지 않는다.

### 보상·RSI·AMP 및 평가

- 보상은 자기 primitive 합계 ×1, 동료 보상 ×0이며 carry를 2로 나누지 않는다. PPO의 `task_reward_w=0.5`, `disc_reward_w=0.5`는 그대로다. 팀 보상 제거와 task/AMP 혼합 가중치는 다른 설정이다.
- RSI는 SIT의 loco/sit 50/50, CLIMB의 loco/climb 50/50, 두 carry의 loco/pickUp/carryWith/putDown 40/10/40/10이다. 물리 검사한 유효 프레임에서 후반 70%/전체 30% 방식으로 뽑고, 유효 skill이 없는 경우 fallback으로 실제 비율이 달라질 수 있다.
- AMP family는 carry/sit/climb 65/10/25이며 두 carry task는 같은 carry expert family를 사용한다. RSI skill 분포와 AMP expert 분포를 동일한 것으로 해석하지 않는다.
- 공유 MLP는 `16-D src role + 32-D task + 16-D dst role → 64→64→64 → layer/head bias` 경로 전체를 task 간 공유한다. Split은 embedding 표를 유지하고 뒤의 MLP·projection만 네 task로 분리한다. NONE/SELF 배경 encoder는 두 경우 모두 task encoder와 별도다.
- Task message의 메시지는 task·역할별 **상태와 무관한 학습 벡터**다. 같은 attention으로 집계하며, 현재 동작을 입력받아 행동 지시를 생성하는 별도 네트워크는 아니다.
- Test wrapper는 기본 `HEADLESS=1`, `EVAL_SKILLS=loco`, `EVAL_SKILL_PROBS=1.0`이다. 이미 든 상태를 확인할 때는 `TASK_GRAPH=carry_at` 또는 `carry_ontop`과 `EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0`을 함께 지정한다. `holding_at/holding_ontop`은 별칭이다. 로컬 viewer는 `HEADLESS=0`, 서버 viewer는 전용 VNC wrapper를 쓴다.

### 지표 읽는 법

- 성공률 비교는 `info/epochs`에 맞춰 최근 200 epoch 평균을 사용한다. 시작 시간이 다른 split은 같은 epoch 구간도 비교한다.
- `relation/edge/{holding,at,ontop,sit,climb}/own_success`는 개별 primitive 조건이다. Carry 전체 완료율과 같지 않으며 추가 정책 edge H→G/H→S의 별도 성공률은 없다. `relation/goal/final_agent_success`는 에피소드 종료 시 담당 과제 전체 성공이다. RSI를 포함한 학습 지표이므로 처음부터 수행하는 평가 결과로 해석하지 않는다.
- Bias는 **task bias 원래 값**과 **Δ = task bias − 같은 물리 타입 쌍의 NONE bias**를 함께 본다. 4 layer × 2 head의 부호 있는 평균은 head별 편차를 가린다. 큰 양의 Δ가 NONE 억제에서 나올 수 있으며 실제 attention은 QK 점수와 함께 결정된다. Bias 증가만으로 성공률 개선 원인을 확정하지 않는다.

## Task 3종 최근 확인 기록

**2026-10-06 23:10 KST 로그 확인분**을 2026-10-07 문서에 반영했다. 실시간 상태가 아니며 아래 성공률은 각 run의 최근 200 epoch 평균이다. 세 실험 모두 2명·2048환경·4물체, seed 42로 본학습을 시작했다.

| 항목 | Task message | 공유 Task MLP | Task MLP split |
| --- | ---: | ---: | ---: |
| 확인 epoch | 10,409 | 10,767 | 3,006 |
| HOLDING | 61.6% | 61.7% | 18.6% |
| AT | 5.7% | 6.0% | 6.3% |
| ON_TOP | 4.4% | 15.7% | 4.2% |
| SIT | 9.2% | 20.4% | 0.9% |
| CLIMB | 19.9% | 20.4% | 16.2% |
| 종료 시 담당 과제 전체 성공 | 8.7% | 18.2% | 6.5% |

동일한 2,801~3,000 epoch의 전체 성공은 message 5.41%, 공유 MLP 5.70%, split 6.48%였다. 해당 확인에서 scalar NaN/Inf는 없고 최근 구간의 물리 reset 실패·동료 보상 기여는 0이었다. Split의 actor/critic 각각 네 task MLP가 본학습 checkpoint 500→1,000 사이에 모두 업데이트된 것도 확인했다. 한 seed의 학습 기록이며 분리 효과나 별도 평가 성공률을 확정하지 않는다.

Bias 확인 checkpoint는 message 10,000 / 공유 MLP 10,500 / split 3,000 epoch다. 공유 MLP의 carry_at H→O는 원래 bias +2.398, NONE −16.112, Δ +18.511로 NONE 억제의 영향이 컸다. Split의 동일 연결은 원래 +0.717, Δ +4.006이었다. 전체 edge·actor/critic 값은 로컬 분석 산출물 [CSV](../output/trend_review_20261006/three_edge_vs_none_latest.csv)·[그래프](../output/trend_review_20261006/three_edge_vs_none_latest.png)를 참조한다. 이 `latest` 산출물은 재분석 때 갱신되며 Git에 포함되지 않는다.

검증 이력은 [changelog](../changelog.md)에 있다. Split 관련 CPU 84개 및 2048환경 짧은 학습, carry_at/carry_ontop 각각 16환경·32-step headless 평가가 통과했다. 장기 성능 비교와 VNC 화면 검증을 대신하는 결과는 아니다.

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

## Unified size RSI task MLP

`approach_scenario_stage1_unified_size_rsi_task_mlp`는 위 task message의 env 설정을 그대로 사용하는 별도 scratch 실험이다. Env YAML은 checkpoint 분리용 `stage1_variant`만 다르다. 과제 비율 10/25/32.5/32.5%, 단독 HOLDING 제외, carry의 단방향 3연결, 자기 primitive 보상 합계(`self=1, teammate=0`), 크기 분포·물리 검사 RSI·AMP family/전문가 연결·634-D task packet을 모두 공유한다. Carry의 loco/pickUp/carryWith/putDown RSI는 AT와 ON_TOP 모두 40/10/40/10이다.

- 정책은 `task_role_mlp`다. 출발 역할 16-D·task 32-D·도착 역할 16-D 임베딩을 연결한 뒤 `64→64→64` ReLU MLP와 layer/head projection으로 attention bias를 만든다. 36가지 task/역할 조합을 먼저 인코딩하고 각 연결에 조회한다. Actor/critic은 별도 파라미터를 갖는다.
- NONE/SELF 배경은 기본 Size RSI처럼 물리 타입·관계·물리 타입의 `EdgeEncoder` MLP를 사용한다. 두 bias projection은 0으로 초기화한다. GTA·토큰 순열은 유지한다.
- `relation_message.enable=false`이며 관계 메시지 표와 가산 경로는 없다. 일반 attention의 value 집계는 유지한다. 기존 task message와 네트워크 및 checkpoint 계약이 달라 직접 resume/eval을 거부한다.
- 전용 env/train YAML·train/test/VNC·output을 사용한다. RSI 캐시 경로는 `output/rsi_cache`로 같으며 내용 hash가 맞으면 재사용한다. 아래 GPU 0은 현재 로컬 서버 예시이며 실행 전 점유를 확인한다.

```bash
# Scratch 학습: 2명, 2048환경, 4물체
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_mlp bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_train.sh 2 2048 4
# 짧은 학습 확인: 본학습과 output 분리
TOKENHSI_GPU=0 MAX_ITERATIONS=1 RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_mlp_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_train.sh 2 2048 4
# CKPT에 같은 task MLP 실험의 checkpoint 절대 경로를 지정
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=carry_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=0 TASK_GRAPH=carry_ontop EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_vnc.sh "$CKPT" 2 1 4 10
# 같은 task MLP checkpoint에서 재개
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT="$CKPT" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_train.sh 2 2048 4
```

## Unified size RSI task MLP split

`approach_scenario_stage1_unified_size_rsi_task_mlp_split`는 task MLP의 공유 영향을 비교하는 별도 scratch 실험이다. Env는 `stage1_variant`만 다르며 과제·3연결·자기 보상·크기·RSI·AMP·634-D packet은 task MLP 및 task message와 같다. Carry의 RSI는 AT/ON_TOP 모두 loco/pickUp/carryWith/putDown 40/10/40/10이고 기존 물리 캐시를 재사용한다.

- 출발 역할 `Embedding(3,16)`·task `Embedding(4,32)`·도착 역할 `Embedding(3,16)`을 유지한다. 출발·도착 표는 별도이며 네 task가 공유한다. Task 표는 task별 다른 행을 쓴다.
- Concat 이후 `64→64→64` MLP를 sit/climb/carry_at/carry_ontop별로 네 개 만든다. Bias projection도 `[task=4,layer=4,head=2,64]`로 분리한다. 한 task 내부의 세 역할 연결은 같은 전용 MLP·projection을 사용한다. Task 간 역할 embedding 공유의 영향은 계속 남는다.
- Task별 9개 역할 조합만 처리한 뒤 환경별로 조회한다(총 36개). Actor/critic은 전체 파라미터가 독립이다. NONE/SELF 배경은 기존 별도 MLP를 유지하며 task별로 분리하지 않는다. 관계 메시지는 없고 GTA는 유지한다.
- Task encoder는 actor/critic 각각 35,552개 파라미터다(기존 9,056개). 전체 네트워크는 4,209,730개로 52,992개(+1.27%) 증가한다. 추가 FP32 weight 약 0.20 MiB, gradient·Adam 상태까지 약 0.81 MiB이며 이는 전체 실행 메모리 증가의 실측값은 아니다.
- 전용 train mode는 `task_role_mlp_split`이다. Packet v5는 유지하지만 variant/fusion 및 weight shape가 달라 task message·공유 task MLP checkpoint를 직접 resume/eval할 수 없다. 전용 config·train/test/VNC·output에서 scratch로 시작한다.

```bash
# Scratch 학습: 2명, 2048환경, 4물체
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_mlp_split bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_train.sh 2 2048 4
# 짧은 확인용 output
TOKENHSI_GPU=0 MAX_ITERATIONS=1 RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_mlp_split_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_train.sh 2 2048 4
# CKPT에 같은 task MLP split 실험의 checkpoint 절대 경로를 지정
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=carry_at EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=0 TASK_GRAPH=carry_ontop EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_vnc.sh "$CKPT" 2 1 4 10
# 같은 split checkpoint 재개
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT="$CKPT" OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_mlp_split bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_train.sh 2 2048 4
```

## Unified size RSI task embedding

`approach_scenario_stage1_unified_size_rsi_task_embedding`은 공유 task MLP 기반의 별도 scratch 실험이다. Env YAML은 `stage1_variant`만 다르고, train YAML은 `relation_bias_mode=task_type_embedding`만 다르다. 보상·성공 조건·과제 분포·634-D task packet·RSI·AMP·GTA·토큰/edge 셔플을 유지한다.

- Actor/critic 각각 `Embedding(6,64)` 하나를 둔다. 카테고리는 `[sit, climb, carry_at, carry_ontop, NONE, SELF]`다. 역할/물리 타입/agent ID를 embedding 입력으로 넣지 않으며 중간 MLP·관계 message가 없다.
- Projection은 `[src type=3, dst type=3, layer=4, head=2, 64]`다. `b[c,s,t,l,h] = dot(E[c], W[s,t,l,h])`이며 H/O/G 방향 쌍·layer/head마다 독립이다. Task·NONE·SELF는 **같은 W를 공유**하고 카테고리 embedding만 다르다. Actor와 critic은 전체 파라미터가 독립이다.
- 같은 task·타입 쌍·layer/head라면 agent/물체 ID와 무관하게 bias가 같다. `carry_at`의 H→O/H→G/O→G는 embedding을 공유하고 projection은 다르다. `climb H→O`와 `carry_at H→O`는 projection을 공유하고 embedding은 다르다.
- 받침 S는 O 타입이다. 따라서 `carry_ontop`의 H→운반 상자와 H→받침은 같은 bias를 받는다. 운반 상자→받침은 O→O projection을 쓴다. 이전 actor/payload/target **역할 쌍별** 설계와 구분한다.
- NONE은 task 연결이 없는 방향, SELF는 대각선에 적용한다. 배경 위에 task bias를 중복 가산하지 않고 유효 task 연결을 교체한다. 역방향·다른 agent의 비연결 토큰은 NONE으로 남는다.
- Canonical endpoint의 실제 H/O/G 타입으로 bias를 만든 뒤 토큰 순열에 맞춰 bias 두 축과 GTA pose를 함께 이동한다. Human readout은 역순열로 복원한다. Task record/primitive edge 순서로 projection을 고르지 않는다.
- Embedding은 std=.02 랜덤, projection은 0으로 초기화한다. 최초 bias는 0이며 학습 후 성장 속도·성공률 개선은 보장하지 않는다. Encoder당 embedding 384 + projection 4,608 = **4,992개** 파라미터다. 전체 네트워크는 4,130,626개로 공유 task MLP보다 26,112개 적다.
- RSI는 SIT loco/sit 50/50, CLIMB loco/climb 50/50, 두 carry loco/pickUp/carryWith/putDown 40/10/40/10을 유지한다. AMP carry/sit/climb family 65/10/25 및 expert matching, size RSI 캐시 `output/rsi_cache`도 그대로다. 보상은 자기 primitive 합계 ×1·동료 ×0이고 PPO task/AMP 가중치는 0.5/0.5다.
- 전용 variant/fusion과 weight 구조로 기존 세 실험 checkpoint의 직접 resume/eval을 거부한다. 학습·평가 모두 task embedding 전용 wrapper와 checkpoint를 사용한다.

2026-10-07 검증: 관련 CPU 91개, GPU 0의 2048환경 짧은 학습(저장 epoch 2), 저장 모델의 carry_at/carry_ontop + carryWith 각각 16환경·32-step headless 평가를 완료했다. 학습 scalar 233종/466값 모두 finite, 물리 reset 실패·물체 binding/보상 합산 오류 0이며 6개 카테고리와 타입 쌍 projection의 학습 경로를 확인했다. 검증 output은 `output/approach_scenario_stage1_unified_size_rsi_task_embedding_check/`와 `_check_carry_at/`, `_check_carry_ontop/`다. 본학습·장기 성능 및 VNC 화면은 아직 검증하지 않았다.

```bash
# Scratch 학습: 2명·2048환경·4물체. 실행 전 GPU 점유 확인
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_train.sh 2 2048 4
# 짧은 검증은 별도 output
TOKENHSI_GPU=0 MAX_ITERATIONS=1 RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_train.sh 2 2048 4
# CKPT에는 task embedding 실험의 checkpoint 경로 지정
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=carry_at EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=0 TASK_GRAPH=carry_ontop EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_vnc.sh "$CKPT" 2 1 4 10
# 같은 실험 checkpoint 재개
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT="$CKPT" OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_train.sh 2 2048 4
```

## Carry task embedding distillation

`approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill`은 task embedding 모델에서 두 agent가 carry_at/carry_ontop만 각각 50/50 독립 샘플링하는 전용 실험이다. 2명·2048환경·4물체와 기존 task/NONE/SELF `Embedding(6,64)`·H/O/G 타입 쌍 projection을 유지한다. Sit/climb embedding 슬롯과 공통 설정 행은 남지만 해당 과제는 샘플링하지 않는다. 기존 네 과제 실험과 checkpoint/output을 분리한다.

- Source 크기는 carry 범위 XYZ 각각 0.20~0.60m, 0.05m 격자로 생성한다. 받침 범위와 밀도는 기존 설정을 유지한다. 고정 asset의 조건부 과제 확률도 carry 두 개만 허용한다.
- RSI는 두 carry 모두 loco/pickUp/carryWith/putDown 40/10/40/10, 물리 검사·후반 프레임 규칙을 유지한다. 공통 RSI 캐시 코드는 그대로이며 실제 크기 조합의 누락 profile은 최초 검사한다.
- AMP family는 carry 100%, 실제 전문가 skill 분포는 loco/omomo/pickUp/putDown = 1/3, 1/3, 1/6, 1/6이다. AMP one-hot·전문가/리플레이 matching은 유지한다. 보상은 자기 primitive 합계×1, 동료×0, PPO task/AMP 가중치는 0.5/0.5다.
- 동결한 원본 unified Stage 1 teacher에 현재 agent 신체·담당 상자·목표를 354-D로 변환해 전달한다. AT는 graph goal, ON_TOP은 실제 회전 bbox 높이를 반영한 받침 위 source 중심을 목표로 한다. Teacher의 Carry task만 활성화하고 원본 정규화 통계를 사용한다.
- 학생 rollout에서 teacher의 32-D Gaussian mean/sigma를 저장하고, scene/agent minibatch 정렬을 유지해 `0.001 × KL(teacher || student)`를 기존 PPO/AMP loss에 더한다. Critic에는 직접 KL gradient를 주지 않으며 teacher는 업데이트하지 않는다.
- 기본 teacher는 `/home/hwanhee/CVPR2027/TokenHSI/output/tokenhsi/ckpt_stage1.pth`다. `TEACHER_CHECKPOINT`, `TEACHER_KL_COEF`, `TEACHER_GRAD_CHECKS`로 변경한다. Train YAML에도 기본 teacher 설정이 들어 있다. 평가는 학생 checkpoint만 사용하므로 teacher 파일이 필요 없다.
- TensorBoard `distill/`에 KL·mean RMSE·teacher/학생 sigma·PPO/critic loss·template label 수를 기록한다. `TEACHER_GRAD_CHECKS=2`는 초기 두 optimizer update의 KL gradient·critic 비의존·actor 갱신과 teacher 동결을 검사한다.

```bash
# 본학습: GPU 5, 2명·2048환경·4물체. 실행 전 nvidia-smi로 점유 확인
TOKENHSI_GPU=5 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_train.sh 2 2048 4
# 짧은 확인 학습: 현재 종료 조건상 MAX_ITERATIONS=1은 epoch 2까지 진행
TOKENHSI_GPU=5 MAX_ITERATIONS=1 TEACHER_GRAD_CHECKS=2 OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_train.sh 2 2048 4
# CKPT는 이 실험에서 저장한 학생 checkpoint
CKPT='/absolute/path/to/student.pth'
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=carry_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=5 TASK_GRAPH=carry_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_vnc.sh "$CKPT" 2 1 4 10
```

Test wrapper는 기본 `HEADLESS=0`이며 서버의 화면 없는 평가는 `HEADLESS=1`을 명시한다. 기존 task embedding 또는 옛 34번 distill checkpoint를 직접 resume하지 않는다. 짧은 실행 확인은 장기 운반 성공률 검증과 구분한다.

2026-10-07 연결 검증: CPU 63개 통과. GPU 5·2048환경에서 epoch 2/frame 262144까지 학습·저장했고 scalar 212종/424값 모두 finite, 물리 reset 실패 0, teacher 동결·KL actor/embedding gradient·critic 비의존을 확인했다. carry_at/carry_ontop + carryWith 각각 16환경·32-step headless 평가가 정상 종료됐다. 두 짧은 평가의 운반 완수율은 0이며 장기 성능 결과가 아니다. RSI 캐시 6,640개 profile은 `output/rsi_cache/6361a69fcf429a02a66602bb.pkl.gz`, 검증 결과는 `output/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_check/`와 `_check_carry_at/`, `_check_carry_ontop/`에 있다. 본학습은 시작하지 않았다.

## Four-task embedding distillation

`approach_scenario_stage1_unified_size_rsi_task_embedding_distill`은 원래 task embedding 실험의 sit/climb/carry_at/carry_ontop **10/25/32.5/32.5%**와 행동별 크기·RSI·AMP·보상·네트워크를 유지하고 원본 unified teacher KL을 추가한다. Env YAML은 원본과 variant만 다르다. 단독 HOLDING·별도 Traj task는 샘플링하지 않는다. Carry 전용 distill과는 독립된 scratch 실험이며 checkpoint를 직접 혼용하지 않는다.

- Actor/critic 각각 task 4개+NONE+SELF `Embedding(6,64)`와 H/O/G 타입 쌍 projection을 사용한다. 중간 MLP·관계 message는 없다.
- Teacher는 SIT→Sit, CLIMB→Climb, 두 carry→Carry로 라우팅한다. SIT 목표는 상판+pelvis clearance이며 facing은 상자의 로컬 +X다. CLIMB 목표는 상판+character height다. AT graph goal·ON_TOP 회전 bbox 목표·원본 teacher 정규화·teacher 동결·학생 rollout KL은 carry distill과 같은 구현이다.
- RSI는 SIT loco/sit 50/50, CLIMB loco/climb 50/50, 두 carry loco/pickUp/carryWith/putDown 40/10/40/10이다. AMP carry/sit/climb family는 65/10/25이며 전문가/리플레이 matching을 유지한다. 크기별 물리 RSI 캐시는 공유하되 누락 profile은 검사 후 추가한다.
- Teacher 기본 경로는 `/home/hwanhee/CVPR2027/TokenHSI/output/tokenhsi/ckpt_stage1.pth`, KL 계수는 0.001이다. `TEACHER_CHECKPOINT`, `TEACHER_KL_COEF`, `TEACHER_GRAD_CHECKS`를 지원한다. 기존 PPO·AMP 학습에 KL을 더하며 학생 평가는 teacher 파일 없이 가능하다.

```bash
# 본학습: 실행 전 GPU 점유 확인, 기존 carry 학습은 자동 중단하지 않음
TOKENHSI_GPU=5 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_distill bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_train.sh 2 2048 4
# 짧은 연결 검증
TOKENHSI_GPU=5 MAX_ITERATIONS=1 TEACHER_GRAD_CHECKS=2 OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_train.sh 2 2048 4
CKPT='/absolute/path/to/four_task_student.pth'
# 로컬 viewer와 headless 평가; TASK_GRAPH=sit/climb/carry_at/carry_ontop/random_scenario
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=sit bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_test.sh "$CKPT" 2 16 4 3
# 서버 VNC
TOKENHSI_GPU=5 TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_vnc.sh "$CKPT" 2 1 4 10
```

2026-10-07 연결 검증: 관련 CPU 56개 통과. GPU 5·MPS·2048환경에서 epoch 2/frame 262144까지 학습·저장했고 scalar 252종/504값 모두 finite, 물리 reset 실패·binding/보상 합산 오류 0을 확인했다. 네 task 모두 teacher label로 들어가며 teacher 동결·학생 actor/embedding gradient·critic KL 비의존을 확인했다. 저장 학생으로 sit/sit RSI·climb/climb RSI·두 carry/carryWith 각각 16환경·32-step 평가가 정상 종료됐다. Output은 `output/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_check/`와 `_check_<task>/`다. 공통 RSI 캐시는 3,622개 profile을 추가해 총 10,262개다. 이는 연결 검증이며 장기 성능 결과가 아니다. 새 본학습은 시작하지 않았으며 기존 carry 학습을 유지했다.
