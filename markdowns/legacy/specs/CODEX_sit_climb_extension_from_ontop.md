# Codex 수정 지시서: `approach_distance_edge_context_ontop` 기반 SIT / CLIMB relation 추가

## 0. 작업 기준

저장소: `KSH3880/CVPR2027`  
브랜치: `approach_clean_context`

기준 실험:

`tokenhsi/data/cfg/multi_agent/approach_distance_edge_context_ontop.yaml`

현재 이 실험에는 이미 다음이 구현되어 있다.

- 2 agents / 3 objects (`O_A`, `O_B`, free object `O_X`)
- environment별 sampled edge graph
- edge capacity 4, agent당 최대 2 edge
- relations: `HOLDING`, `AT`, `ON_TOP`
- replayable graph packet: `[valid, src, dst, relation, owner, pre, term]`
- scalar context `[q_pre, q_term]`
- semantic + context fusion으로 entity-attention bias 생성
- state/progress/success 각 0.2
- own success 또는 TERM-edge success 시 edge reward 0.6 포화
- task reward sharing: `R_A = 0.9 L_A + 0.1 L_B`, `R_B = 0.9 L_B + 0.1 L_A`
- prerequisite를 reward에 곱하지 않음
- ON_TOP target binding / chain / cycle rejection
- edge order shuffle
- viewer preset

이 구조를 되돌리거나 다시 설계하지 말고, 그 위에 `SIT`과 `CLIMB` relation을 추가한다.

기존 `approach_distance_edge_context_ontop` 실험과 checkpoint는 보존하고, 새 실험을 별도 config / script / output으로 만든다.

추천 이름:

```text
approach_distance_edge_context_interaction
```

새 relation mode / schema를 분리한다. 예:

```text
mode: state_relation_edge_interaction_v1
schema_version: 4
```

현재 schema 3 checkpoint를 새 schema 4로 조용히 로드하지 않는다.

---

# 1. Relation vocabulary 확장

현재 relation ID를 유지한다.

```text
HOLDING = 6
AT      = 7
ON_TOP  = 8
SIT     = 9
CLIMB   = 10
```

semantic binding:

```text
HOLDING : H -> O
AT      : O -> G
ON_TOP  : O -> O
SIT     : H -> O
CLIMB   : H -> O
```

`SIT` / `CLIMB`도 일반 Object token을 target으로 사용한다. 별도의 Chair token, ClimbObject token을 추가하지 않는다.

---

# 2. Toy object geometry

이번 실험 목적은 object diversity가 아니라 relation composition 검증이다.

SIT/CLIMB용 별도 curated asset taxonomy를 추가하지 말고, 현재 multi-agent의 일반 box entity를 그대로 사용한다.

새 실험에서는 geometry confound를 줄이기 위해 우선 box를 고정 크기로 사용한다.

권장:

```yaml
box:
  build:
    baseSize: [0.5, 0.5, 0.4]
    randomSize: false
```

세 box 모두 `0.5 x 0.5 x 0.4 m`.

이유:
- HOLDING 가능한 일반 object 유지
- SIT 가능한 chair-height에 가까운 support
- CLIMB 가능한 낮은 step/platform
- ON_TOP 3-box chain 높이 = 1.2m
- relation/context 조합 자체를 우선 검증

기존 ON_TOP random-size config는 보존한다.

---

# 3. Agent별 edge sampling

현재 sampler는 HOLDING을 항상 만들고 second edge를 뽑는다. 새 실험에서는 agent별 local pattern 자체를 샘플링한다.

agent당 최대 2 edge.

```text
HOLDING only          : 0.10
SIT only              : 0.10
CLIMB only            : 0.10
HOLDING + AT          : 0.25
HOLDING + ON_TOP      : 0.20
HOLDING + CLIMB       : 0.15
HOLDING + SIT         : 0.10
```

합 = 1.0.

Config 예:

```yaml
relationGraph:
  mode: edge_composition
  sampler: two_agent_three_object_interaction
  max_edges_per_agent: 2
  edge_capacity: 4
  pattern_probabilities:
    HOLDING: 0.10
    SIT: 0.10
    CLIMB: 0.10
    HOLDING_AT: 0.25
    HOLDING_ON_TOP: 0.20
    HOLDING_CLIMB: 0.15
    HOLDING_SIT: 0.10
```

시나리오 이름을 먼저 뽑지 말고, agent별 local edge composition을 샘플하고 그 결과로 PRE/TERM과 cross-agent dependency를 compile한다.

---

# 4. 기본 binding

Agent `i`의 assigned object를 `O_i`, 다른 agent object를 `O_j`, free object를 `O_X`라고 한다.

```text
HOLDING: H_i -> O_i
AT:      O_i -> G_i
ON_TOP:  O_i -> support_object
SIT:     H_i -> support_object
CLIMB:   H_i -> support_object
```

---

# 5. SIT / CLIMB target sampling

SIT/CLIMB target은 세 종류 허용:

```text
1. 자기 object O_i
2. 다른 agent object O_j
3. free object O_X
```

## 5.1 SIT-only / CLIMB-only

1-edge이면 기본 후보 `{O_i, O_j, O_X}` 중 valid target을 균등 샘플링.

단 `O_j`가 다른 agent에 의해 이동되는 object이면 target-preparation rule을 적용한다.

## 5.2 HOLDING + SIT / HOLDING + CLIMB

이 경우 `O_i`는 현재 agent가 들고 있으므로 target으로 금지.

금지:

```text
HOLD(O_i) + SIT(O_i)
HOLD(O_i) + CLIMB(O_i)
```

허용 target:

```text
O_j
O_X
```

기본 50/50. 단 `O_j`가 안정적으로 준비되지 않는 graph면 `O_X`로 fallback.

---

# 6. Target preparation rule

다른 agent object `O_j`를 SIT/CLIMB/ON_TOP target으로 사용할 때, 그 object가 현재 episode에서 placement edge로 이동된다면 그 placement edge를 prerequisite에 자동 추가한다.

`placement(O_j)`는:

```text
AT_j
ON_TOP_j
```

예:

```text
A:
H_A --HOLDING--> O_A
O_A --AT-------> G_A

B:
H_B --CLIMB----> O_A
```

이면:

```text
pre(CLIMB_B) = AT_A
q_pre(CLIMB_B) = phi_AT_A
```

B가 자기 물체도 들고 있다면:

```text
Pre(CLIMB_B) = { HOLDING_B, AT_A }
q_pre(CLIMB_B) = min(phi_HOLDING_B, phi_AT_A)
```

SIT도 동일.

Target이 ON_TOP으로 준비되면 AT 대신 해당 ON_TOP edge를 prerequisite로 사용한다.

다른 agent object가 HOLDING-only 등으로 계속 이동 중이고 안정적 placement edge가 없으면 SIT/CLIMB/ON_TOP support로 사용하지 말고 `O_X`로 fallback한다.

---

# 7. PRE compile 규칙

PRE는 random sampling하지 않는다. sampled edges + target binding을 보고 자동 생성한다.

```text
Root HOLDING/SIT/CLIMB: PRE 없음 -> q_pre=1
HOLDING + AT:          Pre(AT) = {HOLDING}
HOLDING + ON_TOP:      Pre(ON_TOP) = {HOLDING} + optional target_placement
HOLDING + SIT:         Pre(SIT) = {HOLDING} + optional target_placement
HOLDING + CLIMB:       Pre(CLIMB) = {HOLDING} + optional target_placement
```

Scalar context:

```text
q_pre(i) = min(phi_j for j in Pre(i))
Pre 없음 -> 1
```

Reward에는 q_pre를 곱하지 않는다.

---

# 8. TERM compile

후속 대화에서 의미를 고정했다. `HOLDING+AT/ON_TOP`은 배치 성공 뒤 release를 허용하므로
`term(HOLDING)=AT/ON_TOP`이다. `HOLDING+SIT/CLIMB`은 물체를 든 채 앉거나 오르는
동시 과제이므로 HOLDING과 SIT/CLIMB가 모두 required goal이고 TERM은 없다.

```text
HOLDING + AT/ON_TOP:  term(HOLDING) = terminal relation
HOLDING + SIT/CLIMB: term(HOLDING) = None, both edges required
term(AT/ON_TOP/SIT/CLIMB) = None
```

---

# 9. 3-edge inference generic path 유지

학습은 agent당 최대 2 edge만 사용한다.

하지만 explicit graph inference에서는:

```text
H_A --HOLDING--> O_A
H_A --CLIMB----> O_B
O_A --ON_TOP---> O_B
```

을 지원해야 한다.

```text
Pre(CLIMB)  = {HOLDING}
Pre(ON_TOP) = {HOLDING, CLIMB}
term(HOLDING) = ON_TOP
term(CLIMB)   = ON_TOP
```

이건 training sampler가 생성하지 않고 explicit graph inference preset으로만 검증한다.

---

# 10. SIT / CLIMB progress

현재 공통 distance progress 철학을 유지한다. 원본 TokenHSI velocity progress를 다시 넣지 않는다.

SIT:

```text
source = Human root
target = object-local tarSitPos transformed to world
d_xy = ||root_xy - sit_target_xy||
```

CLIMB:

```text
source = Human root
target = climb Object center for progress only
d_xy = ||root_xy - object_xy||
```

둘 다 `P = 1 / (1 + max(d_xy - 0.5, 0) / 1.0)`을 쓴다.

---

# 11. CLIMB state

원본 TokenHSI climb의 terminal geometry를 box에 맞게 이식한다.

Box top surface는 rotated bbox vertical extent로 계산한다.

```text
top_z = object_center_z + rotated_vertical_half_extent
```

Toy target:

```text
climb_target_xy = object center xy
feet_target_z   = object top_z
root_target_z   = object top_z + char_h
```

State는 root target 하나만 사용한다.

```text
phi_climb = exp(-10 * ||climb_target - root_pos||^2)
```

feet height는 dense state가 아니라 success validation에만 쓴다.

```text
feet_error = abs(mean(feet_z) - top_z)
S_climb = (phi_climb >= 0.9) AND (feet_error <= 0.05)
```

5cm tolerance는 config에서 조절한다. current/live success이며 one-shot latch는 없다.

---

# 12. SIT state

원본 TokenHSI sit의 terminal 핵심은:

```text
reward_near = exp(-10 * ||tar_pos - root_pos||^2)
```

far approach velocity는 이번 framework의 공통 distance progress로 대체한다.

Generic box에는 원본 chair dataset의 object별 `tarSitPos`가 없으므로 train object 49개의
`tarSitPos` 중앙값 `[0, 0, 0.1381430834425038]`을 config에 기록한다.

SIT target:

```text
sit_target = object_position + rotate(object_rotation, target_local_offset)
```

새 config:

```yaml
sit:
  state_definition: tokenhsi_tar_sit_pos
  near_distance_scale: 10.0
  target_local_offset: [0.0, 0.0, 0.1381430834425038]
```

```text
phi_sit = exp(-10 * ||sit_target - root_pos||^2)
S_sit = (phi_sit >= 0.9)
```

초기 toy에서는 chair facing/orientation term은 추가하지 않는다.

---

# 13. Reward / saturation

기존 그대로:

```text
state weight    = 0.2
progress weight = 0.2
success weight  = 0.2
```

Own success 또는 TERM target success이면:

```text
state_paid = 1
progress_paid = 1
success_paid = 1
edge reward = 0.6
```

추가 금지:

```text
q_pre * reward
activation * reward
(1-term) * reward
soft interpolation
one-shot success bonus
```

---

# 14. Task reward sharing

현재 ON_TOP 실험의 0.9 / 0.1 sharing을 모든 새 graph에 그대로 적용.

```text
R_A_task = 0.9 * L_A + 0.1 * L_B
R_B_task = 0.9 * L_B + 0.1 * L_A
```

Power / collision / box-speed / AMP 경계는 그대로 유지.

---

# 15. AMP motion pool

SIT/CLIMB relation을 추가하면서 carry-only AMP motion pool을 그대로 쓰지 않는다.

branch에 이미 존재하는:

```text
tokenhsi/data/dataset_loco_sit_carry_climb.yaml
```

을 우선 활용한다.

기존 multi-agent loader가 combined dataset category를 처리할 수 있는지 먼저 확인한다.

Discriminator sampling에 최소한:

```text
loco
sit
climb
carry-related motion
```

이 포함되어야 한다.

Task relation과 AMP skill ID를 1:1 architecture dependency로 만들지 않는다. AMP는 motion prior, task edge는 별도다.

---

# 16. Graph packet / network

현재 packet 형식 유지:

```text
[valid, src, dst, relation, owner, pre, term]
```

relation vocabulary만 10까지 확장.

- context는 계속 `[pre, term]` 2 scalar
- 같은 context encoder / fusion MLP 공유
- SIT/CLIMB 전용 context MLP 금지
- A2 semantic relation embedding vocabulary 확장
- 새 schema checkpoint 사용

---

# 17. Validation rules

공통:
- src/dst type 검증
- invalid edge PRE/TERM 참조 금지
- PRE cycle 금지
- TERM self-reference 금지

SIT/CLIMB:
- `H -> O`만 허용
- `HOLD(O_i)+SIT(O_i)` reject
- `HOLD(O_i)+CLIMB(O_i)` reject
- other-agent target이 moving-only이고 안정적 placement가 없으면 reject/fallback
- target placement가 있으면 PRE 자동 연결
- target placement edge에는 dependent SIT/CLIMB을 TERM으로 자동 추가하지 않음

ON_TOP 기존 규칙 그대로 유지.

---

# 18. Viewer preset

기존 preset 유지:

```text
at_ontop
ontop_chain
independent_ontop
random
```

추가:

```text
sit_only
climb_only
hold_sit
hold_climb
at_then_sit
at_then_climb
hold_climb_ontop
```

`at_then_climb`:

```text
A: HOLD_A -> AT_A
B: CLIMB(O_A)
pre(CLIMB_B)=AT_A
```

`at_then_sit` 동일.

`hold_climb_ontop`은 inference-only 3-edge demo:

```text
H_A --HOLDING--> O_A
H_A --CLIMB----> O_B
O_A --ON_TOP---> O_B
```

---

# 19. 필수 테스트

Sampling frequency:

```text
HOLDING           ~10%
SIT               ~10%
CLIMB             ~10%
HOLDING_AT        ~25%
HOLDING_ON_TOP    ~20%
HOLDING_CLIMB     ~15%
HOLDING_SIT       ~10%
```

PRE:

```text
CLIMB(O_X)                     -> q_pre=1
HOLD + CLIMB(O_X)              -> q_pre=phi_HOLD
A:AT(O_A), B:CLIMB(O_A)        -> q_pre=phi_AT_A
A:AT(O_A), B:HOLD+CLIMB(O_A)   -> q_pre=min(phi_HOLD_B, phi_AT_A)
```

SIT도 동일.

TERM:
- HOLD+X 50/50 release sampling 확인
- TERM 있음: X success 후 HOLD reward 포화 유지 가능
- TERM 없음: X success 후 HOLD가 깨지면 HOLD reward raw로 복귀

Invalid:

```text
HOLD(O_A)+CLIMB(O_A) reject
HOLD(O_A)+SIT(O_A) reject
```

Network:
- relation 9/10 forward finite
- graph packet replay 안정성
- edge shuffle equivariance
- actor/critic gradient finite
- explicit 3-edge inference forward 가능

---

# 20. Simulation smoke

CPU test 후 짧게:

1. SIT-only
2. CLIMB-only
3. HOLDING+SIT
4. HOLDING+CLIMB
5. AT(A) -> CLIMB(B)
6. AT(A) -> SIT(B)

기록:

```text
relation id
src/dst
phi
progress
q_pre
q_term
own_success
term_success
saturation
edge reward
agent local reward
shared task reward
```

Cross-agent target에서 A의 placement phi가 B의 q_pre에 정확히 반영되는지 확인.

PRE는 hard action gate가 아니므로 B가 미리 접근하는 것은 허용한다. 처음부터 reward gating을 다시 넣지 않는다.

---

# 21. 마무리

새 config / train / test / VNC scripts / tests / docs를 함께 갱신한다.

예상 새 config:

```text
tokenhsi/data/cfg/multi_agent/approach_distance_edge_context_interaction.yaml
```

기존 ON_TOP experiment와 checkpoint는 변경하지 않는다.

작업 시작 시 현재 branch의 실제 diff와 코드를 읽고, 이미 더 최신 구현이 있으면 기존 변경을 보존하면서 적용한다. 분석만 하고 끝내지 말고 코드/config/scripts/tests/docs까지 수정하고 실제 수행한 검증과 미실행 항목을 구분해서 보고한다.
