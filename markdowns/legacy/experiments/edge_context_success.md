# Holding·At scalar context / edge 성공 포화

실험 `approach_distance_edge_context_success`는 Holding·At만 사용하는 **새 학습**이다. 기존 k=5/k=10/OnTop 실험을 보존하고 별도 mode `state_relation_v1` / schema 2로 실행한다. [설계 명세](../specs/CODEX_approach_clean_edge_context_success_spec.md), [전체 실행 가이드](../../config.md).

## 보상과 입력

```text
Holding phi = exp(-10 * ||양손 중점 - 상자 중심||²)
At phi      = exp(-10 * ||상자 중심 - 목표 중심||²)
P           = 1 / (1 + max(d_xy - 0.5, 0) / 1.0)
Holding P 입력: 사람 root → 상자 중심
At P 입력:      상자 중심 → 목표 중심

S_Holding = phi >= 0.9
S_At      = phi >= 0.9 AND abs(상자 중심 z - 목표 중심 z) <= 0.001m
T_i       = TERM 참조 edge의 현재 S (TERM 없으면 false)
F_i       = S_i OR T_i

F=false: edge reward = 0.2*phi + 0.2*P
F=true:  edge reward = 0.2 + 0.2 + 0.2 = 0.6
agent task reward = owner가 같은 edge reward 합
```

매 step의 현재 상태를 사용한다. Holding만 성공해도 해당 edge는 0.6, At가 성공하면 자신의 At와 TERM=At인 Holding이 각각 0.6이므로 agent 최대 task reward는 **1.2**다. 기존의 별도 agent 성공 +0.2는 없다. 성공이 깨지면 즉시 포화가 해제되고, reset에서 성공한 상태도 첫 reward step부터 지급한다. 포화는 raw phi나 물리 성공 판정을 바꾸지 않으며, TERM의 F를 재귀적으로 전파하지 않는다. Power/collision/box-speed penalty와 AMP는 그대로 적용한다.

Holding의 0.9는 손 중점 거리 약 10.3cm이며 실제 접촉/들어 올림 검출이 아니다. At의 1mm도 중심 Z 오차 조건이다. 포화가 안정적인 grasp 또는 손 release를 보장하지 않는다.

## 정책

```text
q_pre  = prerequisite edge들의 현재 raw phi 최솟값 (없으면 1)
q_term = TERM edge의 현재 raw phi (없으면 0)

Holding context = [1, phi_At]
At context      = [phi_Holding, 0]

SemanticEncoder: source16 + relation32 + target16 → 기존 shared MLP → 64
ContextEncoder:  2 → 32 → 64
FusionMLP:       concat(64,64) → 64 → 64
                 → layer/head bias → 기존 entity Transformer + GTA
```

PRE/TERM은 보상에 곱하지 않는다. 정책은 두 scalar로 관계의 현재 상태를 받아 순서를 학습한다. 순서 준수나 대기 행동을 강제하지 않는다. `q_term>=0.9`만으로 포화하지 않고, 참조 edge의 실제 S를 확인한다. 예를 들어 At phi=.95, Z 오차=2mm면 q_term=.95지만 T는 false다.

Task edge semantic은 fused 경로에서 한 번만 반영하고 같은 entity pair의 여러 edge bias는 합산한다. 별도의 edge-token Transformer나 E×E dependency attention은 없다. Actor/critic encoder는 서로 별도이며, 각 encoder의 모든 edge는 가중치를 공유한다.

관측 suffix는 `2*E`; 기본 M=2/O=3/E=4에서는 **595차원**이다. Context와 GTA pose는 RMS 및 일반 observation clipping을 우회한다. PPO는 저장된 rollout context를 읽으며 simulator의 최신 state를 재조회하지 않는다.

## 실행

저장소 루트에서 실행한다. 이 서버에서는 GPU 5·6 중 비어 있는 장치를 사용한다. 아래 예시는 GPU 6이다.

```bash
# GPU 번호는 실행할 장치를 골라 지정 (예시 6)
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_success_train.sh 2 2048 3

# 짧은 검증만: 기존 본학습 output과 분리, env 수는 유지
TOKENHSI_GPU=6 MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_distance_edge_context_success_check \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_success_train.sh 2 2048 3

# 같은 새 실험 checkpoint로 resume
CKPT='/absolute/path/to/ApproachDistanceEdgeContextSuccess.pth'
TOKENHSI_GPU=6 RESUME_CHECKPOINT="$CKPT" \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_success_train.sh 2 2048 3

# 로컬 viewer / 화면 없는 평가
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_success_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_success_test.sh "$CKPT" 2 16 3 1

# 서버 VNC: 기본 2명 / 1환경 / 3물체 / 10회, noVNC 6080
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_success_vnc.sh "$CKPT"
```

본학습 전 `MAX_ITERATIONS`, `RESUME_CHECKPOINT`, `OUTPUT_PATH` 잔여 설정을 확인한다. 처음 실행은 checkpoint를 로드하지 않는다. 새 구조와 기존 v0/OnTop checkpoint는 일반 resume/evaluate 호환되지 않는다.

## 다른 edge 수·사람 수로 평가

기본 `env.relationGraph: {template: independent_carry}`는 현재 사람 수에 맞는 graph를 생성하는 편의 설정이다. Generic reward/context/network는 E=2M를 가정하지 않는다. Explicit graph는 `edges` 아래 id/owner/src/relation/dst/pre/term/required_goal을 지정한다. 지원 relation은 HOLDING, AT이며 OnTop/Beside는 오류로 거부한다.

```bash
# 같은 가중치, 사람 3명/물체 4개/기본 edge 6개
TOKENHSI_GPU=6 HEADLESS=1 \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_success_test.sh "$CKPT" 3 16 4 1

# 같은 M=2/O=3에서 E만 4→3으로 변경하는 평가 예제
TOKENHSI_GPU=6 HEADLESS=1 \
  RELATION_GRAPH=tokenhsi/data/cfg/multi_agent/graphs/edge_context_three_edges.yaml \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_success_test.sh "$CKPT" 2 16 3 1
```

[3-edge 예제](../../../tokenhsi/data/cfg/multi_agent/graphs/edge_context_three_edges.yaml)는 A에 Holding/At, B에 At만 준다. 입력 크기 변경과 가중치 호환을 확인하는 예제이며, 해당 task 성능이 학습·검증됐다는 뜻은 아니다. `RELATION_GRAPH`는 VNC에도 전달된다. 한 rollout의 graph는 모든 환경에서 공유·고정한다.

H/O/G 인덱스는 **논리적 slot**이며 object reset 시 실제 box assignment/reordering을 따른다. Owner는 reward 수령자로 별도 지정한다. Goal 성공은 `required_goal`인 edge의 현재 own success로 집계한다. TERM은 edge당 최대 하나, PRE는 여러 개를 허용한다. 동일 pair에 여러 edge를 둘 수 있으나 sum bias만으로 모든 sequence를 구별한다고 보장하지 않는다.

Checkpoint에는 reward/schema/fusion 계약과 training task instance를 따로 기록한다. 평가에서는 같은 모델·reward 계약 내 E/M/O/graph 변경을 허용하고, training resume에서는 원래 graph와 M/O 일치를 요구한다. Graph tensor는 non-persistent buffer라 E 변경으로 strict weight load를 깨지 않는다.

## 지표와 검증

TensorBoard `relation/edge/holding/*`, `relation/edge/at/*`에 relation별 유효 edge의 평균을 기록한다. `phi_raw`, `progress_raw`, `q_pre`, `q_term`, `own_success`, `term_success`, `reward_saturated`와 weighted state/progress/success/total을 구분한다. `relation/agent/task_total`, `relation/goal/*`, `relation/penalty/*`도 기록한다. Task reward component는 `reward_terms/edge_state`, `edge_progress`, `edge_success`다.

`goal/current_*`는 rollout 평균이고 `goal/final_*`, `goal/ever_*`는 종료된 episode 기준이다. 여러 rollout의 episode 비율은 `goal/completed_scenes`로 가중한다. 종료 표본이 없는 구간은 0으로 표시된다. Required goal이 없는 agent는 성공률 분모에서 제외한다. Reset 성공은 포함한다.

`diagnostics/edge_context_steps.csv`는 기본 2환경의 모든 edge를 30 step마다 기록하며 raw 상태·좌표 오차·실제 성공·TERM·포화·지급 항·agent task/total·패널티를 함께 남긴다. CSV 간격은 학습 중 지표 계산 간격이 아니다. 예전 고정 Holding/At 순서의 CSV/timeline과 파일을 공유하지 않는다.

```bash
. tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python -m pytest -q tokenhsi/tests/test_edge_context_success.py
```

새 kernel/graph/network/normalizer/strict load/penalty 테스트와 기존 회귀 테스트를 제공한다. 실제 실행 검증 범위와 결과는 [changelog](../../../changelog.md)에 기록한다. Scratch의 짧은 smoke는 학습 수렴이나 자연스러운 접근→잡기→운반→배치→release 성공을 입증하지 않는다.
