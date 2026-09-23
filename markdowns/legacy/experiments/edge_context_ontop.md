# 17번: Sampled OnTop edge context

`approach_distance_edge_context_success`(16번)의 scalar PRE/TERM·context encoder·fusion·GTA를 유지하고 ON_TOP, 환경별 그래프, task 보상 공유를 추가한 별도 실험이다. 기존 15번의 gate·고정 Ox·시나리오 분할을 사용하지 않는다.

## 실행

저장소 루트에서 실행한다. 이 서버에서는 GPU 5·6 중 점유를 확인해 선택한다. 아래 GPU 5는 예시다. 기본 학습은 **2 agents / 2048 envs / 3 objects**, 기존 checkpoint 전이 없이 scratch이며 전부 학습한다.

```bash
# 학습
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_train.sh 2 2048 3

# 별도 output에서 짧게 확인할 때만 사용
TOKENHSI_GPU=5 MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_distance_edge_context_ontop_check \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_train.sh 2 2048 3

# 이 실험에서 저장한 실제 checkpoint
CKPT='/absolute/path/to/ApproachDistanceEdgeContextOntop.pth'

# 이어 학습
TOKENHSI_GPU=5 RESUME_CHECKPOINT="$CKPT" \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_train.sh 2 2048 3

# 로컬 viewer
TOKENHSI_GPU=5 HEADLESS=0 \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_test.sh "$CKPT" 2 1 3 10

# 화면 없는 평가
TOKENHSI_GPU=5 HEADLESS=1 \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_test.sh "$CKPT" 2 16 3 1

# 서버 VNC: 기본 at_ontop
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_vnc.sh "$CKPT"

# Ox -> Oa -> Ob 연속 쌓기
TOKENHSI_GPU=5 TASK_GRAPH=ontop_chain \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_vnc.sh "$CKPT"

# A는 Ox 위에 놓기, B는 Holding만
TOKENHSI_GPU=5 TASK_GRAPH=independent_ontop \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_vnc.sh "$CKPT"

# 학습과 같은 랜덤 그래프
TOKENHSI_GPU=5 TASK_GRAPH=random \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_vnc.sh "$CKPT"
```

- output: `output/approach_distance_edge_context_ontop/`; VNC output은 `_vnc` suffix.
- Viewer 기본 `at_ontop`: A가 Oa를 Ga에 놓고, B가 Ob를 Oa 위에 놓는다. 고정 preset은 reset 후에도 같은 역할을 유지한다. `TASK_ROLE_SWAP=1`로 역할을 반전한다.
- Viewer는 기본 loco 초기화. 기존 혼합 RSI를 디버깅할 때만 `EVAL_SKILLS=loco,pickUp,carryWith,putDown EVAL_SKILL_PROBS=0.5,0.1,0.3,0.1`을 지정한다.
- `SEED`, `EPISODE_LENGTH`, `OUTPUT_PATH`, VNC `PORT` 사용 가능. 기본 VNC 포트 6080.
- 기본 카메라는 사람·상자 범위를 함께 잡는다. `TASK_CAMERA=agent`로 기존 agent 추적 카메라를 선택한다. F로 추적을 해제하면 수동 조작할 수 있다.
- AT 마커만 표시한다. OnTop의 현재 목표 중심은 owner 색의 비충돌 십자선이다. 성공만으로 reset하지 않는다.
- Train은 항상 `random`이다. Viewer용 `TASK_GRAPH` 환경변수를 상속해도 고정 과제로 학습하지 않는다.

## reset별 그래프

각 agent는 자기 Holding 한 개를 갖고, 추가 edge를 독립적으로 NONE .2 / AT .5 / ON_TOP .3으로 뽑는다. NONE은 edge 부재이며 embedding 타입이 아니다. 그래프는 에피소드 중 고정, reset 때만 바뀐다. A/B 역할과 실제 box assignment는 대칭 랜덤이다.

| 조합 | 확률 |
| --- | ---: |
| Holding / Holding | 4% |
| Holding / AT | 20% |
| Holding / OnTop(Ox) | 12% |
| AT / AT | 25% |
| AT / OnTop(Ox) | 15% |
| AT / OnTop(상대 상자) | 15% |
| OnTop / OnTop chain | 9% |

상대 Holding-only면 Ox만 받침으로 사용한다. 상대 AT면 Ox/상대 상자를 반반 선택한다. 둘 다 OnTop이면 Ox를 바닥으로 하는 두 방향 chain만 허용한다. 자기 받침·상호 순환·같은 받침에 두 child는 금지한다. 총 edge 2/3/4의 확률은 .04/.32/.64다. 그래프 순서도 shuffle하며 참조를 함께 재배열한다.

## 상태·progress·성공·포화

```text
Holding:
  delta = 양손 중점 - 자기 상자 중심
  progress XY = 사람 root - 자기 상자 중심

AT:
  delta = source 상자 중심 - 실제 goal 좌표
  progress XY = delta XY

ON_TOP:
  extent_z = sum(abs(rotation_matrix[2, axis]) * half_size[axis])
  bottom_z = source_center_z - source_extent_z
  top_z = support_center_z + support_extent_z
  gap = bottom_z - top_z
  delta = [source_x-support_x, source_y-support_y, gap]
  progress XY = 두 상자 중심 XY 거리

공통:
  phi = exp(-10 * sum(delta²))
  P = 1 / (1 + max(dxy - 0.5, 0) / 1.0)
  Holding own_success = phi >= 0.9
  AT/ON_TOP own_success = phi >= 0.9 AND abs(delta_z) <= 0.001

  T_i = TERM으로 지정한 edge의 현재 own_success (없으면 false)
  F_i = own_success_i OR T_i
  R_i = 0.2 * (F_i ? 1 : phi_i)
      + 0.2 * (F_i ? 1 : P_i)
      + 0.2 * F_i
```

성공 중 매 step 지급하며, 성공이 깨지면 바로 raw 보상으로 돌아간다. TERM은 다른 edge의 자체 성공만 읽고 재귀 전파하지 않는다. PRE/TERM context에는 raw phi를 넣으며 보상 포화가 실제 phi·성공을 덮어쓰지 않는다. PRE를 보상이나 성공 판정에 곱하지 않는다.

| Edge | PRE | TERM | 필수 목표 |
| --- | --- | --- | --- |
| Holding-only | 없음 → 1 | 없음 → 0 | Holding |
| 후속 placement가 있는 Holding | 없음 → 1 | 자기 placement | 아님 |
| AT | 자기 Holding | 없음 | AT |
| OnTop(Ox) | 자기 Holding | 없음 | OnTop |
| OnTop(상대 상자) | 자기 Holding + 상대 placement | 없음 | OnTop |

PRE 여러 개는 최소 raw phi를 사용한다. 밑의 AT/OnTop에 위의 OnTop을 TERM으로 달지 않는다. 전체 성공은 모든 valid required goal의 **현재 own_success**가 동시에 참일 때다.

## 보상 공유와 물리 배치

```text
L_A = A가 소유한 edge 보상의 합
L_B = B가 소유한 edge 보상의 합
R_task_A = 0.9 * L_A + 0.1 * L_B
R_task_B = 0.9 * L_B + 0.1 * L_A
최종 env reward = mixed task + 각자 power/collision/box-speed penalty
```

모든 그래프에 적용한다. edge 개수로 나누지 않는다. `(0.6, 1.2)`는 `(0.66, 1.14)`가 된다. AMP/PPO 결합은 기존 경로다. 비켜주기·기다리기를 강제하지 않으며 수렴도 보장하지 않는다.

Ox는 다른 상자와 같은 밀도 100의 dynamic rigid body다. 중력·물리 접촉이 적용된다. 초기 source 플랫폼은 기존 RSI에 필요할 때만 유지하고, target 플랫폼은 AT에만 활성화한다. OnTop 목표에 플랫폼을 배치하지 않는다.

3단 적층을 최대 높이 1.2m 안에 두기 위해 **실제 asset 생성 크기 범위를 0.2~0.4m**로 설정했다. 이는 기존 16번의 0.2~0.6m와 다른 물리 설정이다. 관측 bbox만 줄이지 않는다. AT가 받침을 만드는 경우 위 상자 높이까지 포함하여 target 높이를 제한한다. Ox 접근 공간·box 간 겹침·목표 간 간격을 확인하고 최대 16회 물리 배치를 재시도한다. 이때 선택한 그래프는 바꾸지 않는다. 초기 Holding/AT 성공 자체는 재시도 이유가 아니다.

## 정책·rollout·checkpoint

- schema 3, mode `state_relation_edge_ontop_v1`, relation ID Holding=6 / AT=7 / ON_TOP=8.
- 기본 scene 587 + graph packet 28 = **615차원**.
- edge record: `[valid, src, dst, relation, owner, q_pre, q_term]`. 정수 5개는 연결 메타데이터이며 별도 continuous feature나 agent-ID embedding으로 넣지 않는다.
- Context Encoder는 여전히 **2 → 32 → 64**. semantic64와 concat하여 **128 → 64 → 64** fusion 후 기존 entity attention bias에 더한다. 같은 pair는 합산하고 padding은 0이다.
- actor/critic은 저장 관측의 packet만 읽는다. rollout의 obs/next_obs에 실제 당시 graph/context를 저장한다. 현재 simulator graph로 PPO 과거 샘플을 재해석하지 않는다.
- 전체 packet과 GTA pose는 RMS/clipping을 우회한다. 학습 파라미터 크기는 edge capacity와 독립이다.
- 기존 16번·15번 checkpoint의 직접 로드는 거부한다. 새 config로 scratch 학습하거나 이 실험 checkpoint로 resume한다. warm-start 변환은 구현하지 않았다.
- Viewer preset 변경은 같은 checkpoint로 가능하다. Training resume은 reward/mixing/packet/taxonomy와 sampling 설정·M/O 일치를 검사한다.
- Generic explicit graph의 다른 M/O/E에서 actor/critic forward와 strict weight load는 유지한다. 환경의 학습 sampler·보상 공유는 2명/3상자 범위이며, M>2 협동 보상이나 실제 task 성공을 검증한 것은 아니다.

## TensorBoard·trace

우선 `relation/edge/ontop/phi_raw`, `own_success`, `reward_saturated`, `relation/goal/current_scene_success`, `final_scene_success`를 본다. relation 평균은 해당 relation의 valid edge만 분모로 사용한다.

- `relation/sampling/*`: **reset 기준** 조합/edge 수/역할 분포와 물리 재시도·실패.
- `relation/sharing/*`: local/mixed task와 자기/상대 기여분.
- `relation/ontop/*`: signed gap, source 최하단, support 최상단, 회전 반영 extent, 속도·상대 속도, tilt(라디안), 연속 성공 시간.
- source/support net contact는 각 상자 전체 접촉력의 크기이며 **두 상자 사이의 pair contact를 특정하지는 않는다**. reward 조건에 넣지 않는다.
- `diagnostics/edge_context_steps.csv`: 기본 2환경·30step 간격. raw/context/S/T/F, 지급 항, owner·logical/physical binding·PRE/TERM·목표 중심·기하학·접촉력·penalty를 기록한다.

검증 결과와 실제 실행 산출물은 [changelog](../../../changelog.md)와 `output/approach_distance_edge_context_ontop_check/`에 기록한다. 짧은 smoke의 통과는 자연스러운 적층·release·비켜주기 학습 성공을 의미하지 않는다.
