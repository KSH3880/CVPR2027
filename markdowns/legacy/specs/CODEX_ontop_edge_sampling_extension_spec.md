# Codex 추가 수정 지시서: ON_TOP + Edge Composition Sampling + 약한 Reward Sharing + Viewer Presets

## 0. 작업 요청과 적용 기준

**이미 수정된 로컬 `scalar PRE/TERM context + edge-local current-success saturation` 구현을 보존하고, 그 위에 아래 기능을 추가 구현해줘. 분석/설명으로만 끝내지 말고 코드·config·실행 스크립트·테스트·문서까지 함께 수정한다.**

기존 지시서: `CODEX_approach_clean_edge_context_success_spec.md`.
기준 저장소: `KSH3880/CVPR2027`, 원래 작업 브랜치: `approach_clean`.

문서 작성 시 원격 `approach_clean` HEAD는 여전히 `34a8cb2e4f2482ace7183f71bd40bf32b74be56b`였다. 사용자는 이전 지시서대로 **로컬 코드를 수정 완료했다**고 밝혔다. 따라서 원격의 오래된 코드로 되돌리지 말고 **현재 작업 디렉터리의 실제 코드와 diff가 적용 기준**이다. 최신 로컬 구현의 함수명/schema 번호를 이 문서가 직접 검증했다고 가정하지 않는다.

시작 시 `git status --short`, 현재 브랜치/HEAD, `AGENTS.md`, `changelog.md`, `markdowns/structure.md`, `markdowns/config.md` 관련 부분을 확인한다. 기존 사용자의 변경·실행 중인 학습·checkpoint·output을 덮어쓰지 않는다. 임의 reset/checkout/pull/force push를 하지 않는다.

기존 실험을 보존하고 새 실험 이름은 기본적으로 다음을 사용한다.

```text
approach_distance_edge_context_ontop
```

이미 로컬에서 다른 명명 규칙을 사용 중이면 그 규칙에 맞추되, 변경 파일·실행 명령·이전 실험과의 차이를 최종 보고한다.

### 이번 추가 범위

- ON_TOP relation의 상태/진행/현재 성공 evaluator.
- 매 environment reset마다 primitive edge를 샘플링하는 generator.
- 고정 2-agent / 3-object 학습, agent당 1~2개 edge, scene당 총 2~4개 유효 edge.
- 환경별로 다른 graph를 observation/rollout/PPO에 정확히 전달.
- 모든 학습 episode에서 local task reward를 `0.9 self + 0.1 other`로 공유.
- Viewer에서는 `at_ontop`을 기본으로 하는 고정 graph preset과 random 모드.

**이번 요청은 ON_TOP 학습과 viewer 확장이다. 새로운 5/6-edge 정량평가 프로젝트, 새로운 reward gate, CLEAR_FROM skill, 별도 edge Transformer를 추가하라는 요청이 아니다.** 기존 가변 E/M/O policy 지원은 보존한다.

---

## 1. 변경하지 않을 기존 합의

이전 구현이 아래 계약을 만족하는지 먼저 확인하고, 이미 맞는 것은 재작성하지 않는다.

### 1.1 Edge reward

각 edge i의 실제 현재 상태를 `phi_i`, 진행 점수를 `P_i`, 실제 현재 성공을 `S_i`라고 한다.

```text
T_i = own_success[term_index[i]]     # TERM 없으면 False
F_i = S_i OR T_i                     # 보상 포화 flag

state_paid_i    = 1 if F_i else phi_i
progress_paid_i = 1 if F_i else P_i
success_paid_i  = float(F_i)

edge_reward_i = 0.2*state_paid_i + 0.2*progress_paid_i + 0.2*success_paid_i
```

모든 유효 edge는 최대 0.6이다. **TERM 성공도 success 항까지 포함해서 세 항을 모두 1로 포화**한다. Own과 TERM이 동시에 성공해도 중복 가산하지 않는다. Padding edge의 세 paid component와 total은 모두 0이다.

성공은 현재 성공을 유지하는 모든 step에서 지급한다. First-success/one-shot/latch/history 기반 보상, 성공 reset에 대한 보상 억제, 새 dwell/hysteresis는 넣지 않는다. TERM은 참조 edge의 **own_success**만 읽고 `F`, paid success, 과거 done을 읽지 않는다.

### 1.2 Reward gate 금지

```text
pre * reward
activation * state/progress
(1-term) * reward
(1-term)*raw + term*maximum
```

위와 같은 prerequisite/term 기반 연속 감쇠를 다시 넣지 않는다. State/progress 자체의 기하학적 정의와 valid/padding mask는 유지한다. 포화는 Boolean switch다.

### 1.3 Policy context

```text
q_pre(i)  = min(phi[j] for j in Pre(i)); Pre가 없으면 1
q_term(i) = phi[term(i)]; TERM이 없으면 0
ctx(i)    = [q_pre(i), q_term(i)]
```

`phi`는 실제 현재 상태다. Sigmoid gate, paid state, success latch로 대체하지 않는다.

```text
sem   = SemanticEncoder(source_type, relation_type, target_type)
ctx_z = ContextEncoder([q_pre, q_term])
fused = FusionMLP(concat(sem, ctx_z))
      -> layer/head별 bias
      -> 기존 entity attention + GTA
```

**신경망에 PRE/TERM E×E dependency matrix를 추가하지 않는다.** 내부 context 계산에 필요한 참조 metadata와, 신경망에 구조를 직접 입력하는 것은 다른 것이다. 이번에는 전자만 사용한다.

### 1.4 HOLDING, AT, progress

```text
phi_H = exp(-10 * ||mean(left_hand, right_hand) - object_center||²)
S_H   = phi_H >= 0.9

phi_AT = exp(-10 * ||object_center - goal_center||²)
S_AT   = (phi_AT >= 0.9) AND (abs(object_z - goal_z) <= 0.001)

P = 1 / (1 + max(d_xy - 0.5, 0) / 1.0)
```

HOLDING progress의 source는 human root이며 손 중점이 아니다. AT progress는 object→goal XY 거리다. HOLDING k=10, AT k=10, state/progress/success의 0.2 가중치를 유지한다.

Power/collision/box-speed penalty, AMP reward, PPO hyperparameter, fall/timeout, 학습 env=2048은 유지한다. Task success가 생겼다고 즉시 episode를 종료하지 않는다.

---

## 2. 학습 scene와 edge 표현

학습은 **항상 2 agents / 3 objects**다.

```text
Humans: H_A, H_B
Objects in logical order: O_A, O_B, O_X
Goals: G_A, G_B

H_A owns/manipulates O_A
H_B owns/manipulates O_B
O_X is unassigned; neither agent has a HOLDING edge to O_X
```

O_X는 agent의 조작 대상에 할당되지 않았다는 뜻이지, 움직이지 않는 static object라는 뜻이 아니다. 기존과 동일한 **dynamic rigid box**로 두며 물체 간 충돌을 유지한다. ON_TOP의 target이 되어도 ownership/HOLDING edge를 새로 붙이지 않는다.

각 agent i에는 반드시 다음 root edge가 있다.

```text
h_i: H_i --HOLDING--> O_i
```

이후 second edge를 최대 하나 추가한다.

```text
NONE:   추가 edge 없음; HOLDING only
AT:     p_i: O_i --AT--> G_i
ON_TOP: p_i: O_i --ON_TOP--> 선택된 support object
```

`NONE`은 policy relation type이 아니라 **edge의 부재**다. NONE task token을 유효한 edge로 넣지 않는다.

```text
유효 edge 수/agent ∈ {1,2}
유효 edge 수/scene ∈ {2,3,4}
```

G_A/G_B entity slot은 현재 scene 구조대로 유지해도 되지만, 해당 agent에게 AT가 없으면 그 goal에 연결되는 task edge나 AT reward를 생성하지 않는다. ON_TOP을 fake Object→Goal edge로 바꾸지 않는다.

---

## 3. 샘플 확률과 정확한 target 선택 규칙

### 3.1 Agent별 second relation 독립 샘플

각 environment의 episode reset에서 A와 B를 독립적으로 다음 분포로 샘플링한다.

```yaml
second_edge_probabilities:
  NONE: 0.20
  AT: 0.50
  ON_TOP: 0.30
```

Episode 중에는 샘플을 바꾸지 않는다. Success 발생 시 graph를 교체/삭제하거나 다음 edge를 생성하지 않는다.

먼저 두 agent의 second relation을 **둘 다** 정한 다음 ON_TOP target을 조건부로 결정한다. 학습 generator에서 scenario ID를 정책에 넣지 않는다. 조건부 target 결정은 물리적으로 유효한 조합을 만들기 위한 규칙이다.

### 3.2 ON_TOP이 1개인 경우

agent i가 ON_TOP, j가 상대 agent라고 하자.

| j의 second relation | i의 ON_TOP target | 확률 |
|---|---|---:|
| NONE | O_X | 100% |
| AT | O_X | 50% |
| AT | O_j | 50% |

상대가 HOLDING-only인 경우 O_j를 받침으로 삼지 않는다. 이 task 정의에서는 상대에게 계속 들고 있으라고 요구하면서 동시에 그 object를 배치된 받침으로 취급하지 않는다.

### 3.3 둘 다 ON_TOP인 경우

독립적으로 target을 뽑지 말고, 다음 **유효한 chain orientation**을 공정하게 하나 뽑는다.

```text
50%:
  p_A: O_A --ON_TOP--> O_X
  p_B: O_B --ON_TOP--> O_A

50%:
  p_B: O_B --ON_TOP--> O_X
  p_A: O_A --ON_TOP--> O_B
```

이때 second relation 종류 자체는 이미 샘플링한 ON_TOP/ON_TOP을 유지한다. 잘못된 target을 뽑았다고 relation 전체를 재샘플링하여 위의 0.2/0.5/0.3 분포를 바꾸지 않는다.

### 3.4 금지 조건

- Self support: `O_i ON_TOP O_i` 금지.
- Mutual/cyclic support: `O_A ON_TOP O_B`, `O_B ON_TOP O_A` 동시 금지.
- 같은 support 중앙에 두 object를 놓는 분기 금지: `O_A ON_TOP O_X`와 `O_B ON_TOP O_X` 동시 금지.
- HOLDING-only assigned object를 support로 참조하는 graph 금지.
- 존재하지 않는 edge/유효하지 않은 object를 참조하는 graph 금지.

**한 support당 direct ON_TOP child는 최대 하나**로 제한한다. 이것은 현재 중앙 적층 목표의 task 규칙이지, 넓은 받침 위에 두 물체를 놓는 모든 물리 상황이 불가능하다는 뜻이 아니다.

Cycle 검사는 ON_TOP object-support graph와 prerequisite graph에 한다. **PRE+TERM을 합친 graph 전체에 DAG 제약을 걸지 않는다.** `Pre(AT)={HOLDING}`, `Term(HOLDING)=AT`은 정상적인 반대 방향 참조이며, 이를 cycle 오류로 처리하면 안 된다.

### 3.5 최종 기대 분포

아래는 graph family를 직접 샘플링할 확률표가 아니라, 위의 **edge-first + conditional-target generator가 만들어야 할 결과 분포**다. A/B 대칭을 합친 값이다.

| 생성 결과 | 확률 |
|---|---:|
| HOLDING only / HOLDING only | 4% |
| HOLDING only / HOLDING+AT | 20% |
| HOLDING only / HOLDING+ON_TOP(O_X) | 12% |
| HOLDING+AT / HOLDING+AT | 25% |
| HOLDING+AT / HOLDING+ON_TOP(O_X) | 15% |
| HOLDING+AT / HOLDING+ON_TOP(상대 object) | 15% |
| ON_TOP(O_X) → ON_TOP(첫 agent object) chain | 9% |
| **합계** | **100%** |

추가 기대값:

```text
P(E=2) = 0.04
P(E=3) = 0.32
P(E=4) = 0.64
E[E]   = 3.6

ON_TOP이 하나 이상인 episode = 0.51
cross-agent preparation dependency가 있는 episode = 0.15 + 0.09 = 0.24
```

역할 대칭은 각각 절반이다. 예를 들어 AT_A→ON_TOP_B의 직접 협력 graph는 7.5%, 반대도 7.5%; ON_TOP_A(O_X)→ON_TOP_B(O_A) chain은 4.5%, 반대도 4.5%다.

---

## 4. PRE / TERM / required-goal 자동 생성

Sampled edges를 만든 후 참조를 규칙으로 생성한다. ID나 edge 배열 위치에 규칙을 숨기지 말고 object binding과 relation을 이용한다.

### 4.1 Placement lookup

```text
placement(O_i) = agent i가 가진 second edge
                 AT 또는 ON_TOP이면 그 edge
                 NONE이면 없음
```

Free object O_X는 preparation edge가 없다. Assigned object O_i의 placement가 없는 것과 free support의 preparation이 없는 것은 다르다. **HOLDING-only O_i를 free support로 처리하면 안 된다.**

### 4.2 규칙표

| Edge | Pre references | TERM reference | required_goal |
|---|---|---|---|
| HOLDING_i, second 없음 | [] | 없음 | True |
| HOLDING_i, second 있음 | [] | placement(O_i) | False |
| AT_i | [HOLDING_i] | 없음 | True |
| ON_TOP_i → O_X | [HOLDING_i] | 없음 | True |
| ON_TOP_i → O_j | [HOLDING_i, placement(O_j)] | 없음 | True |

Support preparation은 AT로 한정하지 않는다. **상대의 ON_TOP도 preparation edge가 될 수 있다.**

Raw scalar context 계산은 기존 규칙 그대로다.

```text
q_pre  = min(참조한 edge들의 raw phi); 참조가 없으면 1
q_term = TERM으로 지정된 edge의 raw phi; 없으면 0
```

예: `O_A ON_TOP O_X`, `O_B ON_TOP O_A`.

```text
Pre(HOLDING_A) = []
Term(HOLDING_A) = ON_TOP_A
Pre(ON_TOP_A) = [HOLDING_A]
Term(ON_TOP_A) = none

Pre(HOLDING_B) = []
Term(HOLDING_B) = ON_TOP_B
Pre(ON_TOP_B) = [HOLDING_B, ON_TOP_A]
Term(ON_TOP_B) = none
```

**ON_TOP_A.term = ON_TOP_B를 추가하지 않는다.** 밑에 있는 placement relation은 위에 쌓는 동안과 쌓은 뒤에도 실제로 유지되어야 한다. `AT_A.term = ON_TOP_B`도 추가하지 않는다. 종료되는 것은 물체를 잡고 있어야 한다는 HOLDING 요구이지, 이미 만든 받침의 배치 상태가 아니다.

### 4.3 Reward success와 context는 별개

ON_TOP의 own_success는 아래 5절의 자기 기하학적 성공으로 계산한다. Pre가 낮다는 이유로 own_success나 paid reward를 0으로 만들지 않는다. Pre/history 검증을 reward에 곱해 재도입하지 않는다.

모든 edge의 raw phi/own_success를 같은 물리 snapshot에서 계산한 다음, 별도 단계에서 Pre/Term context와 TERM saturation을 계산한다. Edge 배열 순서에 따라 phi가 먼저 갱신된 edge만 읽는 방식은 금지한다.

---

## 5. ON_TOP 물리 정의: 이번 구현의 기본안

대화에서 ON_TOP의 정확한 state/success 식까지 수치로 확정한 것은 아니므로, 아래는 **AT과 일관되게 맞춘 이번 명세의 기본 구현안**이다. HOLDING/AT의 기존 식은 변경하지 않는다. 새로운 contact 보상이나 velocity 보상은 추가하지 않는다.

### 5.1 Binding과 목표

```text
e: O_s --ON_TOP--> O_t
O_s = 올릴 source object
O_t = 받침 target object
```

Semantic은 반드시 Object→ON_TOP→Object다. 실제 target entity는 O_t이며, 새 fake Goal token을 생성하거나 기존 G_i를 O_t로 바꾸어 대신 사용하지 않는다.

ON_TOP 목표는 **받침의 현재 XY 중심 위에 source 바닥이 닿는 위치**다. 받침의 pose/크기는 매 step 실제 simulator 상태에서 읽는다. Reset 때의 pose나 상대 agent의 예정 AT goal에 고정하지 않는다.

### 5.2 Upright box에서의 식

Source/target 중심을 p_s, p_t, 실제 box 높이를 H_s, H_t라 하면:

```text
p_star = [p_t.x, p_t.y, p_t.z + H_t/2 + H_s/2]
d_xy   = ||p_s.xy - p_t.xy||
gap_z  = (p_s.z - H_s/2) - (p_t.z + H_t/2)
       = source bottom height - support top height

phi_ON_TOP = exp(-10.0 * (d_xy² + gap_z²))
P_ON_TOP   = 1 / (1 + max(d_xy - 0.5, 0) / 1.0)
S_ON_TOP   = (phi_ON_TOP >= 0.9) AND (abs(gap_z) <= 0.001)
```

즉 AT과 같은 3D 근접 state를, 실제 object에서 매 step 계산하는 목표 중심 p_star에 적용한다. 코드 주석은 기존 저장소의 한국어/영어 규약을 따른다.

### 5.3 회전을 포함한 실제 geometry 계산

실제 box는 동적으로 회전할 수 있으므로 reward 코드에서 항상 `size_z/2`만 쓰지 않는다. 이번 기본안은 **gravity/world-Z 기준 중심 적층 + 회전된 bounding box의 수직 extent**로 정의한다.

각 box의 half extents를 h, local→world 회전행렬을 R이라고 하면:

```text
z_extent = sum(abs(R[2,k]) * h[k], k=0..2)
z_top    = center.z + z_extent
z_bottom = center.z - z_extent
```

동일한 값을 기존 local 8개 bbox corner를 현재 quaternion으로 회전한 뒤 world Z의 max/min으로 구해도 된다. 구현은 프로젝트의 quaternion convention과 helper를 재사용한다.

```text
source_z_extent  = rotated_vertical_extent(source_rotation, source_half_size)
support_z_extent = rotated_vertical_extent(support_rotation, support_half_size)

gap_z = (p_s.z - source_z_extent) - (p_t.z + support_z_extent)
p_star = p_t + [0,0, support_z_extent + source_z_extent]
```

이 p_star/gap_z를 위 phi/progress/success 계산과 viewer 표시가 공유한다. Upright/yaw-only이면 정확히 H_s/2, H_t/2 식으로 환원되어야 한다.

**이것은 위치·bounding-envelope 기반 성공 proxy다.** 기울어진 모서리 접촉, 손으로 받치고 있는 상태, 매우 짧은 통과를 안정적 support contact와 완전히 구별하지 않는다. Contact/상대속도/tilt/maintain duration은 진단으로 기록하되 이번 reward에 몰래 추가하지 않는다. Reset에서는 받침이 upright·비관통·바닥 지지된 상태여야 한다. 큰 tilt에서 더 엄격한 top-face/contact 판정이 필요하면 실패 사례로 보고하고 별도 변경으로 분리한다.

### 5.4 공통 주의

- 1 mm는 **source bottom과 support top의 절대 오차**다. 두 물체의 center Z가 같은지를 검사하지 않는다.
- Signed gap을 보관하고 success에서는 abs를 쓴다. `gap_z <= 0.001`만 쓰면 깊게 관통한 상태도 성공하므로 금지한다.
- Source half-height와 target half-height를 빠뜨리거나 두 번 더하지 않는다.
- Logical object slot에서 실제 physical object size/rotation을 올바르게 gather한다.
- ON_TOP progress에는 Z/velocity/pre를 곱하지 않는다.
- `q_term = phi_ON_TOP`은 연속값이고, TERM saturation에는 위의 엄격한 S_ON_TOP을 쓴다.
- 새로운 ON_TOP threshold/default는 config에 명시하고, 1 mm가 실제 접촉 상태에서 얼마나 유지되는지는 simulator trace로 검증한다. 성공률을 높이려고 tolerance를 조용히 넓히지 않는다.

### 5.5 ON_TOP reward

새 특수 reward를 만들지 않고 공통 kernel을 사용한다.

```text
ON_TOP의 own_success=True
  -> ON_TOP state/progress/success 모두 1
  -> edge total = 0.6
  -> 해당 owner의 HOLDING도 TERM 성공으로 세 항 포화, 0.6
```

ON_TOP이 다시 깨지면 그 edge의 live saturation이 해제된다. 받침의 AT/ON_TOP은 자기 성공으로 계속 평가한다. 어떤 object가 target으로 쓰인다는 이유로 support owner에게 별도 bonus를 중복 가산하지 않는다.

---

## 6. 항상 적용하는 약한 task reward 공유

먼저 valid edge reward를 owner별로 합산한다.

```text
L_A = sum(edge_reward with owner A)
L_B = sum(edge_reward with owner B)
```

**모든 학습 episode**에서, graph 종류나 cross-agent dependency 존재 여부와 관계없이:

```text
R_A_task = 0.9 * L_A + 0.1 * L_B
R_B_task = 0.9 * L_B + 0.1 * L_A
```

Batch tensor가 `[N,2]`이면 다음과 같다.

```python
mixed_task = 0.9 * local_task + 0.1 * local_task.flip(dims=[1])
```

`flip`은 반드시 agent dimension에 한다. Flatten한 `[N*2]`에서 전체 reverse를 하면 서로 다른 environment 보상이 섞이므로 금지한다.

- AT/AT, HOLD/HOLD, free support ON_TOP에도 항상 적용한다.
- Dependency-neighbor 선택이나 task별 mixing ON/OFF를 구현하지 않는다.
- Local task reward는 edge 수로 나누지 않는다. 1-edge agent와 2-edge agent의 상한이 다른 것은 이번 설계에서 의도한 결과다.
- 혼합 후 **각 agent 자신의** power/collision/box-speed penalty를 기존처럼 합산한다.
- AMP와 task/disc weight 결합은 기존 학습 경로를 유지한다. AMP/penalty까지 서로 섞지 않는다.
- 섞인 task reward가 실제 PPO return/GAE 계산에 들어가야 한다. Logging만 섞거나 actor loss 합산만 바꾸는 것으로 대체하지 않는다.
- 별도 team success bonus, agent success bonus, CLEAR_FROM reward는 추가하지 않는다.

수치 테스트:

```text
L=(1.2,1.2) -> mixed=(1.2,1.2)
L=(0.6,1.2) -> mixed=(0.66,1.14)
L=(0.0,1.0) -> mixed=(0.10,0.90)

mixed.sum(agent_dim) == local.sum(agent_dim)
```

이것은 고정 mixing coefficient이며 pre/term 상태에 따라 raw reward를 줄이는 gate가 아니다. Edge 내부 0.2 가중치는 유지되지만, 최종 agent objective에서의 기여도는 0.9/0.1에 의해 바뀐다.

학습 sampler/mixing의 기본 범위는 M=2다. 추론은 policy action 계산에 reward가 필요하지 않으므로, 기존 generic M/E 확장을 이 2-agent mixing 함수로 막지 않는다. M>2 viewer/explicit graph가 기존에 가능했다면 reward 계산을 끈 action path는 그대로 동작하게 한다. 이번 작업에서 임의의 M-agent reward 공유 규칙을 새로 정하지 않는다.

---

## 7. 가장 중요한 확장: 환경마다 다른 graph를 저장하고 재생

이전 지시서는 한 rollout에 공통 graph 하나로 시작해도 된다고 했지만, **이번에는 환경별 독립 random graph가 필수**다. 이전 static-graph 허용 범위는 여기서 확장한다.

### 7.1 Padded graph runtime

Train에서 capacity는 4이며 valid edge 수만 2/3/4로 바뀐다.

```text
E_cap = 4  # 이 학습 config의 저장 capacity. learned parameter 크기가 아니다.

edge_valid      [N,E_cap] bool
edge_src        [N,E_cap] long
edge_dst        [N,E_cap] long
edge_relation   [N,E_cap] long
edge_owner      [N,E_cap] long
pre_mask        [N,E_cap,E_cap] bool   # runtime 전용
term_index      [N,E_cap] long        # 없으면 -1
required_goal   [N,E_cap] bool        # runtime/metrics 전용

phi             [N,E_cap]
progress_raw    [N,E_cap]
own_success     [N,E_cap] bool
q_pre, q_term   [N,E_cap]
```

형식은 현재 로컬 graph class를 확장해도 되지만, `[E]` metadata 하나를 모든 environment에 broadcast하는 것으로는 이번 요구를 충족하지 못한다.

Padding reference는 safe gather index로 대체한 후 mask한다. `-1`을 그대로 gather해 마지막 edge의 성공을 읽으면 안 된다. Padding edge는 어떤 encoder/bias/reward/metric에도 유효 기여가 없어야 한다.

### 7.2 PPO가 필요로 하는 graph binding도 observation에 포함

`[pre,term]`만 저장하면 부족하다. 같은 `q_pre=0.5`라도 `O_A AT G_A`와 `O_A ON_TOP O_X`는 다른 입력이다. Actor/critic이 과거 rollout sample의 source/target/relation을 정확하게 복원해야 한다.

현재 구현에 structured observation + typed graph side-buffer가 end-to-end로 이미 있으면 재사용한다. 없다면 이 repo의 flat observation 경로와 잘 맞는 **기본 구현은 packed graph record suffix**다.

```text
각 edge record의 고정 필드 순서:
[valid, src, dst, relation, owner, q_pre, q_term]

record shape [N,E_cap,7]
observation = [기존 clean-scene node/pose | flatten(records)]
```

- `q_pre,q_term`만 2차원 context MLP에 넣는다.
- `src/dst/relation`은 semantic binding과 gather/scatter에 쓰는 이산 metadata다.
- `owner`는 routing/검증 metadata이며, 숫자를 agent-ID embedding으로 학습시키지 않는다.
- `valid/src/dst/relation/owner`를 continuous context feature로 MLP에 넣지 않는다.
- 원래 2차원 execution context를 7차원으로 늘린 것이 아니다. **저장 packet의 metadata와 학습 feature를 구분**한다.
- PRE/TERM E×E matrix, raw phi, own_success/history/level을 새 policy feature로 추가하지 않는다.

Flat buffer에 metadata를 float32로 저장한다면 small integer를 손실 없이 저장하고, network에서 검증 후 integer/bool로 해석한다. RMS와 observation clipping은 packet 전체를 우회한다. Padding 필드는 안전한 canonical 값으로 저장한다.

Node/pose block 크기가 그대로인 M=2/O=3 clean-scene 기준으로:

```text
base width = 587
new graph packet = 4*7 = 28
total = 615
```

이 숫자는 현재 base node/pose schema가 유지될 때의 sanity check다. 실제 코드는 공통 layout helper에서 계산하고 615를 하드코딩하지 않는다. 기존 2E suffix와 새 7E packet을 schema로 명확히 구분한다.

### 7.3 Rollout 일관성

```text
obs_t = scene_t + graph_binding_of_episode + ctx_t
policy(obs_t) -> action_t
step -> reward_t, scene_(t+1)
obs_(t+1) = scene_(t+1) + 동일 episode graph + ctx_(t+1)
```

해당 environment가 reset될 때만 graph가 바뀐다. Reset된 일부 environment의 graph를 변경해도 다른 환경의 graph는 변하면 안 된다.

PPO minibatch forward는 **저장된 observation의 binding/context**를 사용한다. 현재 simulator의 최신 `graph`, `relation_matrix`, `term_index`, `logical_assignment`를 과거 sample에 가져다 쓰지 않는다.

`obses`와 `next_obses`, value bootstrapping, scene minibatch shuffle, actor/critic 모두 같은 규칙을 지킨다. Tensor view를 저장해 나중에 reset이 과거 buffer를 변경하지 않도록 buffer에 값을 복사한다.

### 7.4 GNN/추가 attention은 필요 없음

```text
sem:    [N,E_cap,64]   # graph가 환경마다 달라져 batch dimension이 필요
ctx_z:  [N,E_cap,64]
fused:  [N,E_cap,64]
bias_e: [layers,N,heads,E_cap]

scatter to (edge_src, edge_dst)
  -> bias_pair [layers,N,heads,L,L]
  -> 기존 GTA entity attention
```

NONE/SELF 배경 bias는 기존 방식을 보존해도 되지만, task edge 위치에 과거 static carry bias를 중복으로 더하지 않는다. ON_TOP sample에 숨은 `O_i AT G_i` edge가 남아서는 안 된다. 두 graph가 같은 물리 상태를 공유해도 task bias는 sample의 semantic/context를 반영해야 한다.

Task bias aggregation은 기존 합의대로 masked scatter-add다. Source/destination/edge order를 바꿔도 같은 규칙을 사용한다. Entity 수와 edge 수에 dependent한 learned weight나 learned edge-slot/agent-ID embedding은 추가하지 않는다.

### 7.5 Edge 순서

Episode reset마다 valid edge를 shuffle해도 되고 기존 shuffle 기능이 있으면 유지한다. `src/dst/relation/owner/valid/Pre/TERM/required_goal/context`를 같은 permutation으로 remap한다.

핵심 보장은 **permutation consistency test**다. 현 구조가 이미 shared per-edge encoder + scatter여서 수학적으로 순서에 독립적이면, shuffle은 그 구현 검증과 실수 방지 목적이다. Shuffle 자체를 별도의 추론 능력으로 과장하지 않는다.

---
## 8. 물리 reset / assignment / platform 처리

### 8.1 Logical ↔ physical binding

기존 object assignment randomization을 유지한다. `O_A/O_B/O_X`는 logical slot 이름이며 실제 simulator box actor ID와 같다고 가정하지 않는다.

기존 코드의 `_agent_box_assignment`, `_logical_box_order`, `_logical_box_values`와 동등한 현재 구현을 재사용한다. State/progress/size/bbox/rotation/visualization이 모두 동일한 logical binding을 따라야 한다.

특히 `ON_TOP -> O_X`는 `_assigned_box_values`만으로 계산할 수 없다. Unassigned object까지 포함한 logical object tensor에서 gather해야 한다. Source와 support의 크기를 owner index로 잘못 조회하지 않는다.

### 8.2 Reset 순서

다음 논리적 순서를 만족하도록 현재 reset 경로에 통합한다.

```text
1. 해당 env의 second relations/conditional support targets 샘플
2. graph compile: valid edges, Pre/TERM, required_goal
3. physical box assignment 및 기존 humanoid/box reference-state reset
4. AT가 있는 agent의 target/target platform만 활성화
5. O_X 및 object/support 초기 위치의 물리적 유효성 확인
6. root/dof state를 simulator에 반영하고 일관된 reset snapshot 확보
7. 모든 edge phi/P/own_success 계산
8. q_pre/q_term와 packed observation 생성
9. graph sampling statistics 및 viewer graph 표시 갱신
```

현재 reset 구현에 맞춰 함수 호출 순서는 조정 가능하지만, observation을 만든 뒤 graph를 바꾸거나 오래된 assignment/geometry로 phi를 만들면 안 된다.

Scene/agent sampling과 AMP reference-state sampling을 혼동하지 않는다. `NONE` relation은 humanoid motion skill 이름이 아니다. 기존 skillInitProb/skillDiscProb, 모션 데이터, AMP discriminator는 유지한다. ON_TOP 전용 모션 데이터가 없다는 이유로 policy 학습을 pre-scripted stacking으로 대체하지 않는다.

### 8.3 Source platform과 target platform 구분

기존 carry 환경은 box 초기 위치를 받치는 source platform과 AT goal을 받치는 target platform을 만들 수 있다.

- Box 초기 reference-state를 받치는 **source platform**은 필요하면 기존대로 유지한다.
- Agent에게 AT가 있을 때만 그 agent의 **target platform**을 기존 규칙으로 사용한다.
- HOLDING-only/ON_TOP agent의 오래된 AT target platform은 비활성 위치로 옮기거나 충돌하지 않는 비활성 상태로 둔다.
- ON_TOP의 p_star를 이용해 target platform을 이동시키거나 박스 밑에 따라다니는 invisible support를 만들지 않는다.
- O_X/받침 box를 fixed joint, kinematic actor, teleport로 고정하지 않는다.
- Source box와 support box 사이의 실제 PhysX collision/contact가 살아 있는지 확인한다.

ON_TOP에서 박스가 성공 위치에 있을 때 **진짜 support object가 지지하는지**를 시뮬레이션에서 확인한다. 숨은 AT platform이 적층을 대신 지지하면 잘못된 구현이다.

### 8.4 Goal/marker

AT 없는 agent의 G_i는 shape 유지를 위한 inactive/distractor slot으로 남아도 되지만 task edge/reward/target platform을 갖지 않는다. 슬롯 값은 finite하고 valid한 scene 좌표로 유지한다. `_tar_pos`를 ON_TOP target의 새로운 의미로 덮어쓰지 않는다.

Viewer에서는 사용하지 않는 goal marker를 숨기고, ON_TOP 목표 표시가 필요하면 별도의 collision 없는 디버그 표시를 사용한다. 실제 ON_TOP target entity와 pose는 항상 support object다.

### 8.5 물리적 유효성과 샘플 분포

Graph validity와 물리적으로 실행하기 좋은 초기 배치는 별개다. 다음을 확인한다.

- O_X는 upright/yaw-random, 바닥 지지, 초기 관통 없음.
- ON_TOP target과 주변에 접근 가능한 여유 공간이 있음.
- AT base goal이 다른 box/platform과 겹쳐 이미 불가능한 받침을 만들지 않음.
- AT→ON_TOP 및 ON_TOP→ON_TOP의 예상 최종 stack height를 실제 asset size로 계산하여 기록.
- 두 단계 stack을 기본 carry target 높이에 다시 얹어 작업 공간을 과도하게 높이는 배치를 피함.

필요한 위치/goal feasibility 재샘플링은 **이미 선택한 graph와 chain orientation을 유지한 채** 수행한다. 불리한 graph를 다른 relation로 바꾸는 silent fallback은 금지한다. Max retries와 실패 이유를 명시한다.

실제 asset 크기/높이 때문에 어떤 branch가 계속 유효한 scene을 만들 수 없다면 해결하지 않은 상태로 성공했다고 보고하지 않는다. 필요하면 **asset 생성 전**의 graph-compatible size sampling 또는 명시적 새 config 제약으로 해결하고 기존 분포와 달라진 점을 기록한다. 이미 생성한 rigid asset을 그대로 둔 채 `_box_size` 숫자만 바꾸는 것은 금지한다.

이번 기본 요청은 모션 curriculum이나 자동 난이도 curriculum이 아니다. 근거 없이 확률을 70/15/15로 바꾸거나 ON_TOP-only warmup을 추가하지 않는다.

### 8.6 성공 reset과 빈 목표 처리

이전 원격 구현의 `done.all()`에 의한 성공 reset 재시도 경로가 로컬 새 모드에 남아 있는지 확인한다. HOLDING-only episode는 reference reset에서 이미 HOLDING 성공일 수 있으며, 합의한 current-success reward에서는 정상 샘플이다.

새 모드에서 성공했다는 이유로 무조건 재샘플링하여 HOLDING-only를 지우지 않는다. AT edge가 하나도 없다고 빈 target tensor의 `all()`을 task 성공으로 오해하거나 reset을 무한 재시도해서도 안 된다. Task 성공은 9절의 explicit required_goal 기준으로 계산한다.

---

## 9. Task 성공과 진단

### 9.1 유지해야 하는 목표

```text
HOLDING-only agent:
    required_goal = 자신의 HOLDING

second edge 있는 agent:
    required_goal = 자신의 AT 또는 ON_TOP
    자신의 HOLDING은 required_goal 아님
```

Scene current success는 모든 valid required_goal edge의 **실제 own_success**의 AND다. Reward F를 task 성공으로 보고하지 않는다. 어느 단계까지 한 번 달성했는지를 나타내는 history metric이 있으면 현재 유지 성공과 구분한다.

예:

```text
A: HOLDING -> AT
B: HOLDING -> ON_TOP(O_A)

scene_success = S_AT_A AND S_ON_TOP_B
```

```text
A: HOLDING -> ON_TOP(O_X)
B: HOLDING -> ON_TOP(O_A)

scene_success = S_ON_TOP_A AND S_ON_TOP_B
```

밑의 placement가 깨져도 위의 ON_TOP만 현재 성립할 수 있으므로, 위 edge 하나만으로 전체 성공을 판정하면 안 된다.

### 9.2 기본 logging

기존 edge-local raw/paid/context logging을 유지하고 다음을 추가한다.

```text
sampling/second_relation/{NONE,AT,ON_TOP}
sampling/valid_edge_count/{2,3,4}
sampling/ontop_target/{free,teammate}
sampling/chain_first_owner/{A,B}
sampling/graph_family_frequency
sampling/layout_retry_count / failure_reason

edge/own_success, term_success, reward_saturated
edge/phi_raw, progress_raw, q_pre, q_term
edge/state_component, progress_component, success_component, total

ontop/xy_error
ontop/signed_gap_z, abs_gap_z
ontop/source_bottom_z, support_top_z
ontop/target_center_x/y/z
ontop/support_speed, relative_speed, source_tilt, support_tilt

agent/task_local
agent/task_mixed
agent/task_self_contribution
agent/task_other_contribution
penalty/power, collision, box_speed
scene/current_required_goals_success
```

구체적 trace에는 env/episode/edge ID, source/target logical/physical IDs, relation, owner, Pre/TERM 참조를 포함한다. GPU training hot path에서 전체 batch를 CPU로 복사하지 않고 기존 bounded diagnostics를 이용한다.

Sampling 분포는 **reset event 기준**으로 집계한다. Episode 길이가 다르면 per-step graph 점유 비율은 설계 확률과 달라질 수 있으므로 둘을 구분한다. A/B role frequency도 기록한다.

Relation별 metric 평균은 그 relation의 valid edge 수를 분모로 사용한다. Padding이나 해당 relation이 없는 sample을 0으로 평균하여 성능처럼 보고하지 않는다.

A가 비켜주는 행동을 확인하기 위한 근접/막힘/clearance 지표는 diagnostics-only로 추가할 수 있다. 그 값을 새 reward로 사용하거나 A를 강제로 움직이지 않는다.

---

## 10. Viewer: 학습과 분리된 고정 preset

**사용자가 말한 추론은 우선 viewer 시각화다.** 학습은 위 random edge sampling이고, viewer 기본값은 보고 싶은 협동 graph를 고정한다.

### 10.1 Preset 목록

| TASK_GRAPH | A edges | B edges | 용도 |
|---|---|---|---|
| `at_ontop` **기본값** | HOLDING_A, AT_A→G_A | HOLDING_B, ON_TOP_B→O_A | A 배치 후 B 적층 |
| `ontop_chain` | HOLDING_A, ON_TOP_A→O_X | HOLDING_B, ON_TOP_B→O_A | ON_TOP을 2단계 조합 |
| `independent_ontop` | HOLDING_A, ON_TOP_A→O_X | HOLDING_B only | Free support 위 primitive 확인 |
| `random` | 학습 generator | 학습 generator | 다양한 graph 디버깅 |

Preset은 별도 policy나 다른 reward가 아니라 **같은 graph compiler에 넣는 explicit edge binding**이다. PRE/TERM을 viewer용으로 다르게 구현하지 않는다.

필요하면 `TASK_ROLE_SWAP=1`로 A/B 역할만 바꿀 수 있게 한다. Default는 역할 고정이다. Object physical assignment, 실제 위치/rotation randomization은 graph 의미와 분리한다.

`at_ontop`/`ontop_chain` graph는 viewer의 매 reset에서도 유지한다. `random`에서만 매 reset graph를 재샘플링한다. 명시하지 않은 graph를 성공 중간에 교체하지 않는다.

### 10.2 Viewer 동작

- 새 ON_TOP taxonomy/schema로 학습한 checkpoint를 로드한다.
- Policy는 scene/edge semantics/context로 actions를 출력한다. Reward로 다음 action을 검색·선택하거나 online update하지 않는다.
- 환경이 모니터링을 위해 reward를 계산하는 것은 허용하지만 action 결정에 사용하지 않는다.
- 기본 num_agents=2, num_objects=3, viewer num_envs=1.
- `HOLDING→AT→release→상대 ON_TOP` 전체를 보기 위해 success 직후 자동 reset하지 않는다. 기존 timeout/fall은 유지한다.
- 기존 evaluation의 locomotion 시작 설정을 기본으로 두고, RSI 중간 상태/성공 상태로부터의 디버그 시작은 명시적 옵션으로 분리한다. 본 시연을 이미 쌓인 상태에서 시작하여 성공처럼 보이게 하지 않는다.
- 실패, 넘어짐, 받침 이동도 정상적으로 표시한다. 비키기/놓기/쌓기를 scripted controller로 대신하지 않는다.

### 10.3 표시

최소한 reset 시 콘솔에서 다음을 확인할 수 있어야 한다.

```text
selected preset / seed / graph signature
edge list: owner, source, relation, target
Pre references / TERM reference
```

가능한 기존 viewer 디버그 경로로 목표와 상태를 표시한다. 무리한 새 UI 라이브러리는 추가하지 않는다.

- AT goal만 실제 goal marker 표시.
- ON_TOP은 support object와 현재 목표 중심을 구분해서 표시.
- A/B와 O_A/O_B의 기존 소유자 색상 유지, O_X는 unassigned 색상 유지.
- 두 ON_TOP chain은 첫째/둘째 relation이 혼동되지 않게 trace 이름 표시.
- Camera는 agent뿐 아니라 O_X와 최상단 물체까지 프레임에 들어오게 한다. 기존 `_video_focus_points`가 humanoid만 본다면 ON_TOP 시각화에서 support/stack points를 포함하는 옵션을 둔다.

### 10.4 Scripts

새 환경 YAML과 필요한 train config/schema 전달을 기존 스크립트 구조로 연결한다.

```text
tokenhsi/data/cfg/multi_agent/approach_distance_edge_context_ontop.yaml

tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_train.sh
tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_test.sh
tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_vnc.sh

output/approach_distance_edge_context_ontop/
```

Train script의 graph mode 기본값은 `random`이다. Viewer script의 기본값만 `at_ontop`이다. 공통 환경변수 기본값 때문에 학습도 고정 시나리오만 나오게 만들지 않는다.

구현 후 다음 인터페이스가 동작하도록 한다. 아래는 **구현할 명령 인터페이스 예시**이며 현재 파일이 이미 존재하거나 실행 검증되었다는 뜻이 아니다.

```bash
# GPU 번호는 실행자가 nvidia-smi로 비어 있는 장치를 확인한 뒤 지정한다.
# 학습: 2 agents / 2048 envs / 3 objects
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_train.sh 2 2048 3

# 새 실험에서 저장된 실제 checkpoint로 설정
CKPT='/path/to/new_ontop_checkpoint.pth'

# 기본 viewer: AT -> ON_TOP
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_vnc.sh "$CKPT"

# ON_TOP -> ON_TOP chain
TASK_GRAPH=ontop_chain TOKENHSI_GPU=5 \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_vnc.sh "$CKPT"

# Free support 위 ON_TOP + 상대 HOLDING only
TASK_GRAPH=independent_ontop TOKENHSI_GPU=5 \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_vnc.sh "$CKPT"

# 학습과 같은 random graph viewer
TASK_GRAPH=random TOKENHSI_GPU=5 \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_ontop_vnc.sh "$CKPT"
```

VNC/runtime wrapper, GPU 지정, headless/GUI, PORT, checkpoint path 검사 등은 기존 안전한 처리 흐름을 재사용한다. Unknown preset이나 부적절한 M/O는 명확한 오류를 낸다.

---

## 11. Config 계약 예시

아래 key는 이번에 추가할 **의미상 제안**이다. 현재 로컬 schema 명칭과 충돌하면 일관되게 조정하고 validator/runtime/network/player/docs를 같이 수정한다. Unknown field를 무시하는 방식으로 구현하지 않는다.

```yaml
env:
  numAgents: 2
  numObjects: 3
  numEnvs: 2048

  relationGraph:
    mode: edge_composition
    sampler: two_agent_three_object
    max_edges_per_agent: 2
    edge_capacity: 4
    second_edge_probabilities:
      NONE: 0.20
      AT: 0.50
      ON_TOP: 0.30
    ontop:
      free_support_probability_when_other_at: 0.50
      first_agent_probability_when_both_ontop: 0.50
      max_children_per_support: 1
      reject_cycles: true
    shuffle_edge_order: true

  relationReward:
    # 기존 local edge-context 모드/schema를 확장한다.
    # 구조/packet/taxonomy 변경은 metadata에 새 contract로 기록한다.
    state_reward_weight: 0.2
    progress_reward_weight: 0.2
    success_reward_weight: 0.2
    satisfaction_threshold: 0.9
    holding:
      hand_distance_scale: 10.0
    at:
      state_definition: box_near
      near_distance_scale: 10.0
    ontop:
      state_definition: centered_stack_world_z
      near_distance_scale: 10.0
      z_tolerance: 0.001
      vertical_extent: rotated_bbox
    progress:
      kind: distance
      delta: 0.5
      sigma: 1.0
    context:
      kind: pre_term_scalar
      prerequisite_reduction: min
    success:
      at_z_tolerance: 0.001
      saturation: own_or_term_success
      terminate_when_all_subgoals_done: false
    observation:
      edge_context_fields: [pre, term]
      graph_packet_fields: [valid, src, dst, relation, owner, pre, term]

  rewardSharing:
    enabled: true
    kind: fixed_two_agent_task_mix
    self_weight: 0.9
    other_weight: 0.1
    apply_to: task_edges_only
    apply_on_all_training_graphs: true

  viewerGraph:
    default_preset: at_ontop
    role_swap: false
```

Sampling 확률은 유한·비음수이고 합계가 1인지 검증한다. Mixing weight는 지정한 0.9/0.1을 사용하고 임의로 재정규화하지 않는다.

Edge 수가 같아도 relation/target이 environment마다 바뀌므로 metadata에는 단순한 E뿐 아니라 binding이 포함되어야 한다. Topology는 재생성 가능한 runtime data이며 checkpoint의 learned parameter shape를 결정하지 않는다.

---

## 12. Checkpoint / relation vocabulary / 호환성

- 현재 로컬 relation enum에서 사용하지 않는 ON_TOP ID를 배정한다. 기존 HOLDING/AT ID를 재번호화하지 않는다.
- ON_TOP type pair는 Object→Object다. Semantic encoder의 relation embedding 크기, taxonomy validation, checkpoint metadata, player allowlist를 함께 갱신한다.
- 기존 H/AT-only checkpoint에는 학습된 ON_TOP relation이 없으므로 그대로 ON_TOP을 수행할 수 있다고 가정하지 않는다.
- 새 relation row나 packet schema가 필요한데 `strict=False`로 mismatch를 숨기지 않는다.
- 기본 학습 검증은 **새 ON_TOP config로 새 학습**이다. Old checkpoint warm-start는 이번 필수 기능이 아니며, 필요하면 별도의 명시적 변환/초기화 경로로만 처리한다.
- 새 ON_TOP checkpoint로 viewer preset/random을 바꾸는 것은 정상 evaluation override다. Reward/model contract는 유지하면서 task graph instance/sampling mode만 바뀌므로 old-style config exact-equality 검사로 차단하지 않는다.
- Training resume는 model/obs/taxonomy/reward/mixing contract를 검증한다. Sampling probability 변경도 실제 실험 설정에 기록한다.
- Graph binding/mask/capacity는 runtime에서 재구성할 수 있어야 한다. 모델 parameter가 E_cap에 묶여 viewer의 가변 edge 지원을 깨뜨리면 안 된다.
- Observation packet version, graph record width, context feature width=2, taxonomy, ON_TOP geometry definition, success tolerances, mixing weights를 metadata에 기록한다.
- 새 graph packet을 쓸 때 env/actor/critic/player/normalizer/rollout buffer 모두 같은 layout helper를 사용한다.

---

## 13. 수정 파일 지도

아래는 기존 저장소에서 확인한 파일 지도다. 사용자가 이미 로컬에서 구조를 바꿨다면 실제 대응 모듈에 적용한다. 새 이름은 구현 제안이며 지금 존재한다고 가정하지 않는다.

| 위치 | 추가/수정 내용 |
|---|---|
| `tokenhsi/utils/relation_task_spec.py` 또는 현재 graph compiler | ON_TOP taxonomy, batched graph, Pre/TERM 규칙, required_goal, 유효성 검사, schema/metadata |
| 새 `relation_graph_sampling.py` 등 적절한 pure module | Edge-first sampler, conditional target rules, preset generator, 확률/RNG |
| `tokenhsi/env/tasks/multi_agent/relation_reward.py` | Pure ON_TOP geometry evaluator, 기존 success saturation 재사용, raw state/context 분리 |
| `tokenhsi/env/tasks/multi_agent/relation_task.py` | 전 logical object gather, relation dispatch, batched phi/context, owner sum, task mixing, diagnostics |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py` | Reset sampler 통합, assignment, AT-only target platform, observation packet, viewer target/metrics |
| `tokenhsi/learning/multi_agent/amp_network_builder_ma.py` | Batch-dependent semantic binding, packed graph parse, masked bias scatter; GTA/ctx fusion 유지 |
| `tokenhsi/learning/multi_agent/scene_normalizer.py` | 새 metadata/context packet passthrough, count-independent RMS 유지 |
| `tokenhsi/learning/multi_agent/ma_agent.py` | Env info/schema, 저장된 rollout graph/next graph 사용, scene minibatch 정합성, checkpoint |
| `tokenhsi/learning/multi_agent/ma_players.py` | Preset override와 checkpoint contract, action-only inference, changed counts 처리 |
| `tokenhsi/env/tasks/multi_agent/relation_diagnostics.py` 등 | Relation별/padding-safe logging, current required-goal metrics |
| 신규 env YAML/train/test/VNC script | Random training, fixed viewer presets, 독립 output |
| `tokenhsi/tests/` | 아래 pure CPU/network/integration tests |

공용 helper를 만들되 잘 동작하는 기존 scalar-context reward/fusion 구현을 통째로 다른 방법론으로 바꾸지 않는다. Baseline mode의 state/progress/obs/checkpoint 동작은 유지한다.

---

## 14. 필수 테스트와 acceptance criteria

아래는 구현 후 수행할 검사다. 이 문서 작성 단계에서 repository test나 simulator가 실행됐다는 뜻은 아니다.

### 14.1 Sampler 구조와 분포

- 어떤 sampled graph도 M=2/O=3에서 agent당 edge 1~2, 총 valid E=2~4를 넘지 않음.
- Agent마다 HOLDING이 정확히 하나, assigned source object가 중복되지 않음.
- NONE sample은 second valid edge가 없음. Padding relation을 task로 인코딩하지 않음.
- 상대 NONE이면 ON_TOP target은 O_X뿐.
- 상대 AT이면 free/teammate target 각각 50%.
- 양쪽 ON_TOP이면 O_X-rooted chain 두 orientation뿐.
- Self/cycle/두 child가 같은 support를 쓰는 graph는 생성되지 않음.
- Assigned HOLDING-only support graph는 validator가 거부.
- PRE+TERM union의 정상 양방향 참조는 거부하지 않음.
- Fixed seed로 graph sequence 재현 가능. Partial reset은 해당 env만 변경.
- 충분한 reset sample(예: 100,000)에서 second marginal .2/.5/.3, E 분포 .04/.32/.64, 3.5절 family 분포 확인. 검증 허용 오차는 sample 수에 맞게 정하고 test가 flaky하지 않도록 한다.
- 별도의 작은 exhaustive test로 orientation 포함 12개의 최종 graph outcome과 정확한 확률 합을 확인.
- Per-step가 아닌 per-reset 분포로 검증.

### 14.2 PRE/TERM와 목표

- HOLDING-only: pre=1, term=0, required_goal=True.
- HOLDING→AT: AT pre는 own HOLDING, HOLDING term은 AT.
- ON_TOP→O_X: pre는 own HOLDING 하나.
- ON_TOP→AT-prepared O_j: pre=min(own HOLDING phi, AT_j phi).
- ON_TOP→ON_TOP-prepared O_j: pre=min(own HOLDING phi, ON_TOP_j phi).
- Placement edge는 term 없음. 위에 다른 child가 있어도 자동 retire되지 않음.
- Example prerequisite phi=.8,.3이면 q_pre=.3.
- phi_ON_TOP=.95, abs_gap=.002이면 q_term=.95일 수 있지만 own/term success=False.
- TERM으로 보상이 포화되어도 raw phi/own_success는 바뀌지 않음.
- Edge permutation 후 references를 remap하면 context도 같은 permutation이고 owner reward는 동일.

### 14.3 ON_TOP 기하학

- Upright 0.4m support: center z=.2, source 높이 .4m이면 목표 source center z=.6.
- 위 상태에서 같은 XY이면 phi=1, gap=0, own_success=True.
- 다른 source/support 크기에서도 두 half-height 보정이 맞음.
- Gap +.001와 -.001은 같은 기준으로 성공 판정하며, +.00101와 -.00101은 실패. Boundary rounding은 helper의 dtype에 맞는 tolerance로 테스트.
- Source center와 support center를 같은 높이에 놓으면 실패. 깊은 penetration도 실패.
- XY/Z 멀어지면 phi가 감소. Progress는 현재 XY 거리식 그대로이며 Z 변화만으로 바뀌지 않음.
- Support를 이동시키면 같은 step의 p_star/phi/next observation이 바뀜. Reset 위치나 G_j를 잘못 참조하지 않음.
- Scene 전체 동일 translation/yaw rotation에서 phi/P/S 불변.
- Rotated vertical extent는 8-corner 직접 계산과 일치. Non-cubic box 회전 fixture 포함.
- Logical/physical assignment permutation 후 같은 물리 relation에 대한 결과 동일.
- O_X는 assigned object만 gather하는 함수의 범위를 넘어도 정상 계산.

### 14.4 Reward/saturation/mixing

- 성공 전 phi=.8, P=.6 -> edge reward=.28.
- Own 또는 TERM 성공 -> 세 weighted component 각각 .2, 합 .6.
- TERM 성공에도 success component .2 포함.
- 유효하지 않은 edge는 paid component/bias 모두 0.
- 성공 상태가 여러 step 유지되면 매 step 지급. 성공이 깨지면 바로 raw로 복귀.
- ON_TOP 성공 뒤 손을 놓아 HOLDING phi가 낮아져도 ON_TOP이 유지되면 HOLDING은 TERM 포화.
- Lower ON_TOP이 깨지고 upper ON_TOP만 유지되면 전체 current success=False.
- Pre 값만 바뀌고 해당 edge의 raw/S/TERM을 고정하면 reward는 동일.
- Mixing은 모든 graph에서 .9/.1. E2/E3/E4 케이스 모두 확인.
- `L=(.6,1.2)`이면 mixed=(.66,1.14). 별도의 edge-count normalization 없음.
- 서로 다른 env에 매우 다른 reward를 넣어도 같은 env의 A/B끼리만 공유.
- Penalty는 각자 유지하고 AMP 결합도 기존대로. Saturation/mixing이 penalty를 없애지 않음.
- 실제 rollout reward/GAE가 혼합된 값으로 계산되는지 integration test.

### 14.5 Batch graph / policy / replay

- 한 batch 안에 E2 HOLD/HOLD, E3 free ON_TOP, E4 AT/AT, E4 AT→ON_TOP, E4 ON_TOP chain을 섞어 forward.
- 동일한 physical scene/context 숫자에서 relation/target binding만 다른 sample을 넣고 각 fused embedding/bias 위치가 올바른지 확인.
- Output 차이 자체는 zero-init 때문에 초기에는 없을 수 있다. 학습 가능한 projection을 업데이트한 뒤 graph-conditioned 경로의 gradient/출력을 검사한다.
- `graph_A`로 저장한 obs를 고정한 뒤 simulator를 `graph_B`로 reset해도 old obs의 forward 결과는 변하지 않음.
- Saved old context 역시 최신 phi로 재계산되지 않음.
- Partial reset 후 미reset env의 graph/obs 불변.
- Padding 위치 이동/edge permutation 후 action/value는 수치 tolerance 안에서 동일.
- Metadata/context/pose는 normalizer와 clipping을 통과해도 보존.
- Learned parameter shape가 E_cap에 독립적. 다른 E_capacity의 synthetic graph forward는 가능해야 함. 이것을 실제 더 큰 물리 task 성공으로 주장하지 않음.
- Actor/critic/semantic/context/fusion/ON_TOP relation embedding의 유한 gradient 확인. Zero-init projection의 첫 step upstream gradient=0은 기존 설계대로 처리.
- New checkpoint save/reload, viewer preset override 성공; incompatible old schema는 명확히 거부.

### 14.6 Reset와 viewer smoke

- HOLDING-only에서 target tensor가 비어도 reset/metric/observation이 정상.
- 성공 reset에서 reward latch/suppression 또는 무한 retry 없음.
- ON_TOP agent의 target platform이 비활성이고 실제 box-box contact가 가능.
- 그래프와 관계없는 AT marker/reward가 남지 않음.
- `at_ontop`, `ontop_chain`, `independent_ontop`, `random` preset이 정확한 graph 생성.
- Fixed preset의 reset은 같은 graph 의미 유지. Random만 재샘플.
- VNC wrapper까지 `TASK_GRAPH`, role swap, GPU, checkpoint가 전달됨.
- Camera에 support/stack이 포함되고 source/target을 확인 가능.
- 정책 forward 과정에 reward/online PPO 호출이 필요하지 않음.

---

## 15. 실행 검증과 마무리 보고

먼저 pure CPU tests → network/rollout tests → 실제 환경에서 가능한 simulator smoke 순서로 진행한다.

학습 smoke는 기존 운영 규칙대로 **2048 environments를 유지하고 iteration만 줄인다.** 실행할 때는 `nvidia-smi`로 점유를 확인하고 사용자가 사용 중인 GPU/process를 종료하거나 재시작하지 않는다. GPU를 확보할 수 없으면 해당 검증은 미실행이라고 정확하게 보고한다. 장기 본학습을 자동 시작하지 않는다.

```text
별도 smoke output:
output/approach_distance_edge_context_ontop_check/
```

Simulator smoke에서 확인할 것은 생성/shape/finite forward/reward/contact/reset/checkpoint/save-load이다. 몇 iteration으로 학습 성공이나 비켜주기 수렴을 확인했다고 주장하지 않는다.

Behavior 검증용 trace는 다음 구간을 구분한다.

```text
AT -> ON_TOP:
  A 접근/잡기 -> A 배치 -> A HOLDING TERM 포화
  B 접근/잡기 -> A가 배치를 유지하는 동안 B 적층 -> release/maintain

ON_TOP -> ON_TOP:
  첫 agent가 O_X 위에 배치
  두 번째 agent가 첫 object 위에 배치
  두 placement의 현재 성공이 동시에 유지되는지
```

Task reward sharing은 A가 비켜줄 유인을 학습 목표에 추가하지만, 이동 동작 자체를 보장하지 않는다. No reward gate 설계에서는 PRE가 strict execution constraint도 아니다. 따라서 early placement, support 이동, hand-release 전에 geometry success가 켜지는 경우, 성공 threshold jitter, 계속 막고 서 있는 행동을 각각 raw trace/영상으로 구분한다. 문제를 감추려고 implicit gate, latch, CLEAR_FROM, scripted movement를 추가하지 않는다.

완료 보고에는 다음을 포함한다.

1. 기존 로컬 구현에서 재사용한 부분과 실제 변경 파일.
2. 새 ON_TOP geometry/success 식과 설정값.
3. Edge sampling 및 target 조건부 분포, 실제 샘플 빈도 검증.
4. Batch graph packet/rollout 저장과 PPO replay 처리.
5. 모든 graph에 .9/.1 sharing이 적용된 실제 reward 경로.
6. 학습·viewer 실행 명령과 preset 목록.
7. CPU/network/simulator tests의 실행 결과와 미실행 항목.
8. Checkpoint 호환 범위와 알려진 한계.

`changelog.md`, `markdowns/config.md`, `markdowns/structure.md`를 실제 변경/검증 결과에 맞춰 갱신한다. 새 지시서나 긴 대화 전체를 changelog에 복사하지 않는다.

---

## 16. 최종 금지/필수 사항 요약

```text
[필수]
Train: 2 agents, 3 objects, 2048 envs
Per agent: HOLDING 필수 + optional second 최대 1개
Second: NONE .2 / AT .5 / ON_TOP .3, A/B 독립 샘플
ON_TOP target: 조건부 유효 binding
Both ON_TOP: O_X-rooted chain, orientation .5/.5
Pre/TERM: sampled edges에서 규칙으로 생성
State/progress/success: 각각 .2, own OR term 성공이면 세 항 전부 포화
Reward sharing: 모든 학습 graph에서 .9 own + .1 other
Policy: sem + ctx=[pre,term], 기존 GTA/entity attention
Random graph binding/context를 sample과 함께 rollout에 저장
Viewer default: at_ontop; ontop_chain/independent_ontop/random 선택 가능

[금지]
기존 수정 되돌리기
Scenario ID를 policy에 입력
PRE/TERM dependency matrix를 새 network feature로 추가
Prerequisite/term reward multiplication 또는 interpolation 재도입
ON_TOP을 fake AT/Goal edge로 구현
Support를 reset pose/예정 goal에 고정
두 ON_TOP이 같은 O_X 중앙을 target하거나 mutual support cycle 생성
HOLDING-only assigned object를 free support로 취급
밑의 placement를 위 placement 성공으로 retire
TERM 성공에서 success reward만 빼기
과거/current graph 혼용, padding success 참조, env 사이 reward 혼합
고정 platform/teleport/scripted action으로 적층 대체
Viewer 변경을 위해 별도 학습 model/다른 reward 적용
실행하지 않은 수렴/성공/성능을 검증했다고 보고
```

### 작성 근거와 확인 범위

기존 설계 계약은 앞서 전달한 `CODEX_approach_clean_edge_context_success_spec.md`와 이번 대화의 확정사항을 기준으로 한다. 이번 문서는 그 구현 완료를 전제로 하는 **추가 변경 명세**이며, 최신 로컬 코드를 직접 읽거나 수정한 결과물이 아니다.

원격 기준 commit `34a8cb2e4f2482ace7183f71bd40bf32b74be56b`에서 확인한 통합 지점:

```text
tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py
  object assignment/logical slot gather, clean scene observation,
  reset/AT target platform, metrics, video focus

tokenhsi/learning/multi_agent/ma_agent.py
  clean-scene rollout storage, scene minibatch grouping, PPO reward/GAE path

이전 지시서에서 확인한 모듈:
  relation_task_spec.py / relation_reward.py / relation_task.py
  amp_network_builder_ma.py / scene_normalizer.py
  관련 config/scripts/tests
```

문서 작성 단계에서는 repository 수정, 학습 실행, GPU/프로세스 제어, simulator smoke를 수행하지 않았다. 수치표는 정의한 확률과 보상식의 계산 결과이며 학습 성능 측정값이 아니다.
