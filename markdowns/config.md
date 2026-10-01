# Multi-Agent Carry 실행 가이드

현재 실행 가능한 실험은 원본 1번과 Stage 1 **27·28·30·31·32·33·34·35·36·37번 및 paired/unified 변형**, Stage 2 **29번·SIT plane·unified owner HOLDING 변형**이다. 과거 9~26번 config와 전용 실행 스크립트는 정리했다. 모든 명령은 저장소 루트에서 실행한다.

## 공통 규칙

- 학습 인자는 `[num_agents] [num_envs] [num_objects]`; Stage 1 기본값은 `2 2048 3`이다. 짧은 확인도 환경 수 2048을 유지하고 `MAX_ITERATIONS`만 줄인다.
- 실행 전 `nvidia-smi`로 점유를 확인하고 명령 앞에 `TOKENHSI_GPU`를 지정한다. 실행 중인 프로세스는 임의 종료하지 않는다.
- 본학습 전 셸의 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 확인한다.
- 학습은 기본적으로 scratch다. 기존 실험의 reward/checkpoint 계약을 섞지 않는다. Stage 2 전이는 아래의 `STAGE1_CHECKPOINT`를 사용한다.
- 평가·VNC 인자는 `<checkpoint.pth> [agents] [envs] [objects] [repeats]`다. `HEADLESS=0`은 로컬 viewer, `HEADLESS=1`은 화면 없는 평가다.
- Sampled-edge 로컬 viewer와 서버 VNC의 상자·AT goal marker 색은 매 reset의 과제 배정을 따른다. 단독 대상은 해당 에이전트 색, 공동 물체는 노란색, 미할당 대상은 회색이다. 실행 중인 뷰어는 다시 시작해야 반영된다.
- 데이터 원본 `/home/hwanhee/CVPR2027/TokenHSI`는 읽기 전용이며, 이 저장소는 심링크로 사용한다.

## 현재 실험

| 번호 | Config | 역할 |
| ---: | --- | --- |
| 1 | [amp_humanoid_ma_carry.yaml](../tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml) | 원본 multi-agent carry 기준 |
| 27 | [approach_scenario_independent_with_climb.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_independent_with_climb.yaml) | 독립 물체·goal binding, 점 기준 성공 |
| 28 | [approach_scenario_stage1_plane.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_plane.yaml) | 27번의 ON_TOP·CLIMB을 상판 영역 성공으로 변경; Stage 2 기본 출발점 |
| 29 | [approach_stage2_coordination.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_coordination.yaml) | Stage 1 checkpoint에서 협력 graph 학습 |
| Stage 2 plane | [approach_stage2_coordination_sit_plane.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_coordination_sit_plane.yaml) | 34번 기반 SIT plane·보상·독립 과제 분포 |
| Stage 2 unified owner HOLDING | [approach_stage2_unified_owner_holding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_unified_owner_holding.yaml) | Stage 1 unified owner HOLDING 기반 2H/4O 협력 3과제·독립 5과제 |
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
| Unified owner HOLDING | [approach_scenario_stage1_unified_owner_holding.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_owner_holding.yaml) | Unified semantic + AT/ON_TOP 담당 HOLDING φ 입력 |

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
```

## Unified 독립 5과제 · semantic / owner-HOLDING 비교

두 실험은 **edge 입력 상태 유무만 다르다**. 기존 paired/focus 실험은 보존하며 새 output에서 scratch로 시작한다. 새 AMP 입력은 프레임당 기존 129차원에 과제 one-hot 3차원을 더한 132차원, 10프레임 총 1320차원이다. 기존 paired checkpoint와는 AMP 및 variant 계약이 달라 이어 학습하지 않는다.

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
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_train.sh 2 2048 4
```

각 checkpoint에는 같은 실험의 test/VNC를 사용한다. 로컬 viewer는 다음과 같다.

```bash
CKPT='/absolute/path/to/checkpoint.pth'
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_test.sh "$CKPT" 2 1 4 10
```

`TASK_GRAPH=holding|sit|climb|holding_at|holding_ontop`은 두 사람 모두 해당 과제로 평가한다. 기본 평가는 loco 시작이다. placement의 들고 있는 시작을 보려면 `TASK_GRAPH=holding_at` 또는 `holding_ontop`에 `EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0`을 추가한다. `HEADLESS=1`은 화면 없는 평가다.

```bash
TOKENHSI_GPU=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_vnc.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_vnc.sh "$CKPT" 2 1 4 10
```

각 기본 output은 `output/approach_scenario_stage1_unified`와 `output/approach_scenario_stage1_unified_owner_holding`이다. 짧은 검증은 별도 `_check` output과 `MAX_ITERATIONS=2`를 사용한다. 2명·4물체 sampler이며 가변 인원 평가 sampler 확장은 포함하지 않는다.

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

## Stage 2 unified owner HOLDING

이 실험은 Stage 1 `approach_scenario_stage1_unified_owner_holding`의 2H/4O 관측, owner HOLDING φ edge packet, 토큰 순열, 상자 크기·거리, 과제 조건 AMP, RSI 분포, 보상식과 현재 시점 포화를 이어받는다. Actor encoder를 지정한 Stage 1 checkpoint에서 복사해 고정하고, grounded edge 협력 attention과 확장 action head를 학습한다. 협력 `place_climb/place_sit/place_stack`은 전체 scene의 30/30/30%, 독립 scene은 10%이며 독립 scene의 각 agent는 Stage 1과 같은 5/5/20/35/35%로 샘플한다. 협력 scene에서는 운반 담당 `Hᵢ→Oᵢ HOLDING`, `Oᵢ→Gᵢ AT`와 동료의 `Hⱼ→Oᵢ SIT/CLIMB` 또는 `Hⱼ→Oⱼ HOLDING`, `Oⱼ→Oᵢ ON_TOP`을 동시에 준다. 공유 상자에 SIT/CLIMB하는 동료는 RSI 충돌을 피하려고 loco로 시작하며, Stage 2는 시작부터 하위 과제나 scene 전체가 성공한 reset을 재추첨한다. 자기 edge 합계 0.9 + 동료 edge 평균 0.1, 성공한 AT/ON_TOP과 같은 owner/source의 HOLDING 현재 step 포화는 독립·협력 모두 동일하다. 기존 Stage 2 config와 다른 별도 config·학습 설정·checkpoint·output을 사용한다.

```bash
STAGE1_CHECKPOINT='/absolute/path/to/ApproachScenarioStage1UnifiedOwnerHolding.pth' TOKENHSI_GPU="$GPU" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_unified_owner_holding_train.sh 2 2048 4
RESUME_CHECKPOINT='/absolute/path/to/ApproachStage2UnifiedOwnerHolding.pth' TOKENHSI_GPU="$GPU" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_unified_owner_holding_train.sh 2 2048 4
CKPT='/absolute/path/to/ApproachStage2UnifiedOwnerHolding.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=place_stack \
  bash tokenhsi/scripts/multi_agent/approach_stage2_unified_owner_holding_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" HEADLESS=1 TASK_GRAPH=place_sit \
  bash tokenhsi/scripts/multi_agent/approach_stage2_unified_owner_holding_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU="$GPU" TASK_GRAPH=place_climb \
  bash tokenhsi/scripts/multi_agent/approach_stage2_unified_owner_holding_vnc.sh "$CKPT" 2 1 4 10
```

학습 시작은 해당 Stage 1 변형의 `.pth`가 필요하다. 짧은 확인은 `MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_stage2_unified_owner_holding_check`를 명령 앞에 추가한다. Stage 2 checkpoint를 이어 학습할 때는 `RESUME_CHECKPOINT`를 지정한다.

## 원본 1번

원본 비교가 필요할 때 `ma_carry_train.sh`, `ma_carry_test.sh`, `ma_carry_watch.sh`를 사용한다. 실험별 GPU 지정과 데이터 경로는 위 공통 규칙을 따른다.

지표 정의는 [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 참고한다. TensorBoard는 `tensorboard --logdir output --host 127.0.0.1 --port 6006`으로 실행한다.
