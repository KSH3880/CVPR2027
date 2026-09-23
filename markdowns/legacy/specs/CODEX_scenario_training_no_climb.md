# Codex 구현 지시서: Context-free scenario 학습 — CLIMB 제외

작성일: 2026-09-22  
저장소: `KSH3880/CVPR2027`  
작업 브랜치: `approach_clean_scenario`

> **목표:** 에이전트당 최대 2개 edge로 구성된 joint scenario를 학습한다. Object/goal binding은 랜덤화하고, 유효하지 않은 조합만 제외한다. 정책에 temporal context를 넣지 않는다.
>
> **CLIMB은 현재 리워드 항이 불안정하므로 이번 새 실험에서 완전히 제외한다. 리워드 안정화 후 별도 단계에서 재도입한다.**

이 문서는 새 실험의 구현 명세다. 아래 변경이 이미 구현·실행되었다는 의미는 아니다. 기존 23/24번 실험, checkpoint, 실행 중인 학습과 사용자의 미커밋 변경은 보존한다.

---

## 1. 구현 기준과 범위

- **24번에서 가져올 것:** context-free architecture, semantic-only graph 입력, 기존 entity token/GTA/actor·critic 구조. CLIMB 학습 설정은 가져오지 않는다.
- **23번에서 가져올 것:** HOLDING/SIT 상태함수와 RSI 처리, 가변 박스 설정, combined AMP 구성의 비-CLIMB 부분.
- **새로 구현할 것:** 4-template scenario sampler, 전체 object/goal 랜덤 binding, validity filter, placement에 연결된 HOLDING 포화, template-conditioned RSI, 공유 물체의 일관된 reset.
- 23/24번은 현재 agent당 **단일 edge** 실험이다. 새 실험에서 **최대 2 edge/agent**로 확장한다. 기존 파일의 이름이나 설명만 바꾸는 작업이 아니다. [기준 config: 23번][src23], [24번][src24]

### CLIMB 제외 범위

새 실험의 template, 허용 relation, reward 실행 경로, RSI, AMP motion pool, 기본 평가 preset, 성공률 집계에서 모두 제외한다. `climb`, `climbNoRSI`는 새 motion YAML에서도 제거하여 불필요하게 로드되지 않게 한다.

기존 실험을 위한 CLIMB 코드·데이터·relation ID를 저장소 전체에서 삭제하거나 재번호화하지 않는다. 다만 새 실험에 CLIMB graph가 입력되면 명확한 validation error를 낸다. 현재 공용 evaluator는 SIT 계산 중에도 `config['climb']`를 참조하므로, 새 경로에서는 CLIMB config나 계산을 요구하지 않도록 분리한다. 빈 CLIMB 설정을 끼워 넣어 우회하지 않는다. [현재 evaluator][srcinteraction]

## 2. 학습 template과 시나리오 의미

학습 환경은 **2 humans / 3 objects / 2 goals**, agent당 최대 2 edge, scene 전체 최대 4 edge로 구성한다.

| Agent template | 생성 edge | 최종적으로 유지해야 하는 성공 조건 |
|---|---|---|
| `HOLDING` | `H_i → O_j : HOLDING` | 해당 HOLDING의 실제 current success |
| `SIT` | `H_i → O_j : SIT` | 해당 SIT의 실제 current success |
| `HOLDING_AT` | `H_i → O_j : HOLDING`, `O_j → G_k : AT` | AT의 실제 current success |
| `HOLDING_ON_TOP` | `H_i → O_j : HOLDING`, `O_j → O_k : ON_TOP` | ON_TOP의 실제 current success |

두 agent의 edge 집합 전체가 **하나의 joint scenario**다. 에피소드 중 graph는 고정한다. 명시적인 PRE/START/KEEP/TERM, pending/active/done phase, context encoder 및 context auxiliary reward는 추가하지 않는다.

`HOLDING+SIT`는 **학습에서 제외**한다. 추후 `held object != sit support`인 unseen intra-agent composition 평가 대상으로 남긴다. Zero-shot 성공은 검증할 가설이지 보장되는 결과가 아니다. 이번 기본 평가에는 넣지 않아도 된다.

**미지정 수치의 초기 기본값:** 네 template을 각 `0.25`로 샘플링한다. 이는 대화에서 확정한 최적 비율이 아니라 구현 시작값이며, config에서 변경 가능하게 한다.

## 3. Random binding과 sampling 순서

```text
A/B template 샘플
    ↓
전체 object/goal 중에서 binding 샘플
    ↓
structural / semantic validity 검사
    ↓
template-conditioned RSI skill 샘플
    ↓
reference object-write 충돌 검사
    ↓
공유 scene을 한 번만 초기화하고 기하학 검사
    ↓
joint scenario 학습
```

### Binding 규칙

1. `O0/O1/O2`에 agent 전용 또는 free-object 같은 고정 의미를 부여하지 않는다. 모든 agent는 어느 object든 선택할 수 있다.
2. AT의 goal도 전체 goal 집합에서 선택한다. goal index를 owner index와 동일하다고 가정하지 않는다.
3. Placement pair 내부만 묶는다: **`HOLDING.dst == placement.src`이고 두 edge의 owner가 같아야 한다.**
4. `HOLDING/HOLDING_AT/HOLDING_ON_TOP`의 HOLDING source human은 해당 owner 자신이어야 한다. SIT도 동일하다.
5. Independent/coupled 비율을 별도로 강제하지 않는다. Binding 결과에 따라 coupling이 자연스럽게 생기도록 한다.
6. Agent 역할, object/goal index, edge 순서의 permutation을 일관되게 처리한다. 관측·보상·RSI·성공 판정 모두 동일 binding을 사용한다.

가능하면 **template pair를 먼저 고정하고 valid binding 중에서 균등 샘플링**한다. Rejection으로 구현해도 되지만 모든 것을 매번 다시 뽑아 특정 template이 과도하게 사라지지 않도록 한다. Symbolic validity와 physical reset 실패는 별도로 집계한다.

기존 `own_object`, `free_object`, `allow_teammate_object: false` 제약을 새 실험에 그대로 적용하지 않는다. 특히 logical-to-physical mapping과 `_agent_box_assignment`를 통한 숨은 owner 고정도 점검한다.

## 4. Invalid scenario filter

### 4.1 구조·목표 충돌

| 제외 조건 | 예시 / 판단 기준 |
|---|---|
| 동일 object를 두 agent가 HOLDING | `H_A HOLDING O0`, `H_B HOLDING O0` — co-manipulation은 범위 밖 |
| ON_TOP self-loop | `O0 ON_TOP O0` |
| ON_TOP cycle | `O0 ON_TOP O1`, `O1 ON_TOP O0` |
| 한 source object에 복수의 placement 요구 | `O0 AT G0`와 `O0 ON_TOP O1` 등; 이번 실험은 source당 placement 하나 |
| 서로 다른 object의 같은 single-occupancy goal 점유 | `O0 AT G0`, `O1 AT G0` |
| 같은 support의 상단 공간을 둘 이상이 요구 | 두 human의 SIT, 두 object의 centered ON_TOP, 또는 SIT와 centered ON_TOP의 중복 |
| HOLDING-only 대상이 다른 task의 support | `A: HOLDING(O0)`, `B: SIT(O0)` 또는 `B: HOLDING(O1)+ON_TOP(O1,O0)` |
| 타입·owner·binding 불일치 | H→O가 아닌 HOLDING, O→H인 ON_TOP, owner가 다른 placement pair 등 |

상단 점유 제한과 HOLDING-only support 제한은 **현재 작은 박스 토이 실험의 보수적 범위 제한**이다. 모든 물리 환경에서 절대 불가능하다는 주장은 하지 않는다.

### 4.2 기하학적 validity

실제 asset 크기, 최종 stack 높이, support 안정성, SIT 목표 높이, goal 간 최종 점유 공간, 초기 침투·바닥 아래 body 등을 검사한다. Goal ID가 달라도 최종 박스 공간이 겹치면 그 물리 배치를 재샘플링한다.

ON_TOP chain은 cycle이 없고 실제 기하학 범위가 허용하면 유지한다. Size는 기존 asset 생성 방식에 맞춰 처리하며, reset마다 물리 asset 크기를 바꿀 수 있다고 가정하지 않는다. 기하학 재시도만으로 해결할 수 없는 경우에는 binding을 다시 선택하고 사유를 기록한다.

**의도한 손–물체 접촉, SIT 접촉, 적층 접촉까지 일괄 충돌로 reject하지 않는다.** Filter 통과가 학습 가능성이나 물리적 실행 성공을 보장하는 것은 아니다.

### 4.3 유지해야 하는 valid 예시

```text
A: HOLDING(O0) + AT(O0,G0)
B: SIT(O0)

A: HOLDING(O0) + AT(O0,G0)
B: HOLDING(O1) + ON_TOP(O1,O0)

A: HOLDING(O0) + ON_TOP(O0,O1)
B: SIT(O0)                         # 위에 놓이는 O0의 상단 사용

A: HOLDING(O0) + ON_TOP(O0,O1)
B: HOLDING(O1) + ON_TOP(O1,O2)      # acyclic stack, 높이 검사 필요
```

같은 object를 참조한다는 이유만으로 reject하지 않는다. 반대로 첫 번째 예시가 반드시 “A가 먼저 배치한 후 B가 앉는다”는 시간 순서를 명령받은 것은 아니다. 순서를 강제해야 하는 긴 task는 추론 시 서로 다른 scenario로 나눈다.

## 5. Reward와 HOLDING saturation

### 5.1 유지할 기존 정의

Edge별 `state/progress/success` 가중치는 각각 **0.2**로 유지한다. HOLDING/AT/ON_TOP/SIT의 기존 geometry와 current-success 정의를 재사용하며, 이번 작업에서 새 reward 설계를 하지 않는다.

| Relation | 유지할 핵심 정의 |
|---|---|
| HOLDING | 손 중점–물체 중심 기반 `exp(-10 d²)`, success `phi >= 0.9` |
| AT | 기존 box-near state, success `phi >= 0.9`와 중심 Z 오차 `<= 0.001 m` |
| ON_TOP | 기존 centered stack/world-Z geometry, success `phi >= 0.9`와 source-bottom/support-top gap 절댓값 `<= 0.001 m` |
| SIT | 실제 박스 윗면 중심 + pelvis clearance `0.12 m`를 root target으로 사용, `exp(-10 d²)`, success `phi >= 0.9` |

Progress는 현재 branch의 **distance progress**를 유지한다. 과거 대화의 velocity-dot reward로 되돌리지 않는다. 기본식은 `1 / (1 + max(d_xy - 0.5, 0) / 1.0)`이며 relation별 기존 source/target 정의를 따른다. [23번 설정][src23], [reward kernel][srcreward], [SIT evaluator][srcinteraction]

### 5.2 추가할 고정 룰

> 같은 owner이고 `HOLDING.dst == AT/ON_TOP.src`인 pair에서, placement의 **실제 current success**가 참이면 해당 HOLDING의 세 reward component를 모두 최대값으로 포화한다.

```python
# 개념식. 그래프 순서나 edge index의 인접성을 가정하지 않는다.
paired_placement_success[h] = success[p] if paired_placement_exists(h) else False
saturated[e] = success[e]
if relation[e] == HOLDING:
    saturated[e] = success[e] or paired_placement_success[e]

state_component[e]    = 0.2 * (1.0 if saturated[e] else phi[e])
progress_component[e] = 0.2 * (1.0 if saturated[e] else progress[e])
success_component[e]  = 0.2 * float(saturated[e])
# padding / invalid edge는 전부 0.
```

- 기존 **자기 current success에 의한 포화**도 유지한다.
- Placement가 성공 상태를 벗어나면, placement에 의한 HOLDING 포화는 즉시 해제한다. First-success latch나 과거 성공 이력을 사용하지 않는다.
- 다른 owner의 HOLDING, ON_TOP support 쪽 HOLDING, 단독 HOLDING에는 이 추가 룰을 적용하지 않는다.
- Raw `phi`, raw `success`, reward용 `saturated`는 분리한다. 포화 때문에 물리적 HOLDING이 실제 성공했다고 덮어쓰지 않는다.
- Pair index/mask는 graph로부터 계산하는 reward 내부 정보다. 정책 입력 context로 추가하지 않는다.
- Reward는 owner별 edge 합으로 계산한다. 1-edge 최대 0.6, 2-edge 최대 1.2다. 임의의 edge-count 정규화나 `.9 own + .1 other` 공유/팀 보상은 추가하지 않는다. Power/collision/box-speed penalty와 PPO/AMP 결합은 기존 기준을 유지한다.

**포화는 손을 놓아도 HOLDING 보상 손실이 없도록 하는 장치다. 실제 release나 비켜주기 자체를 보장하는 보상은 아니다.** 별도 release reward는 이번 범위에 추가하지 않고 행동·진단으로 확인한다.

### 5.3 시나리오 완료 판정

Placement pair의 HOLDING은 운반을 위한 edge이므로 **최종 required-goal에서는 제외하고 placement를 required-goal로 둔다.** 그렇지 않으면 정상적으로 내려놓고 손을 뗀 상태를 실패로 판정하게 된다.

단독 HOLDING/SIT는 실제 current success를 요구한다. Scene success는 두 agent의 required-goal이 **동시에 현재 성공**일 때다. Edge별 과거 성공의 OR를 모아 scene success로 계산하지 않는다. Empty required-goal 집합도 성공으로 보지 않는다.

Success 유지 학습을 위해 기존 `terminate_when_all_subgoals_done: false`를 유지한다. Training의 episode 종료와 추론 scheduler의 scenario 전환은 구분한다. 현재 `required_goal` 기반 판정 경로를 참고한다. [기존 판정 코드][srcreward]

## 6. Template-conditioned RSI

### 6.1 최종 샘플링 확률

| Template | loco | pickUp | carryWith | putDown | sit |
|---|---:|---:|---:|---:|---:|
| HOLDING | 1/2 | 1/2 | 0 | 0 | 0 |
| SIT | 1/2 | 0 | 0 | 0 | 1/2 |
| HOLDING_AT | 0.5 | 0.1 | 0.3 | 0.1 | 0 |
| HOLDING_ON_TOP | 5/9 | 1/9 | 3/9 | 0 | 0 |

`omomo`는 모든 template에서 RSI 확률 0이다.

**원본/변형 구분:** SIT 및 HOLDING_AT의 skill 비율은 저장소 내 원본 TokenHSI SIT/Carry config를 따른다. HOLDING-only의 `loco/pickUp=.5/.5`는 23번 방식이다. ON_TOP은 원본 carry에서 `putDown`을 제외한 나머지를 조건부 정규화한 **이번 실험의 adaptation**이다. 원본 ON_TOP RSI라고 표현하지 않는다. [원본 Carry config][srccarrycfg], [원본 SIT config][srcsitcfg], [23번][src23]

기존 23/24번의 `max(relation_id)`로 HOLDING/SIT/CLIMB 세 종류를 고르는 로직을 그대로 사용하지 않는다. **Owner의 unordered edge set 전체를 분류하여 네 template을 구분**한다. 알 수 없는 조합을 HOLDING으로 조용히 fallback하지 않는다.

### 6.2 원본 RSI에서 유지할 처리

- `stateInit: Random`을 사용한다. **loco도 reference motion에서 시작하는 RSI**이며 default pose 초기화가 아니다. `hybridInitProb`를 추가적인 50% 확률로 곱하지 않는다.
- Skill 선택 → `sample_motions()` → `sample_time_rsi()` → `get_motion_state()` 순서를 유지한다. 임의의 첫 frame이나 균등 frame sampler로 바꾸지 않는다.
- Human의 root pose·속도, DOF pose·속도, kinematic body state, reference motion ID/time, AMP history를 동일 reference 기준으로 일관되게 초기화한다.
- Scene translation/yaw transform을 적용하면 관련 human/object pose와 world velocity에도 일관되게 적용한다. 손–물체 정렬을 유지한다. [원본 Carry reset][srccarryreset]

| Sampled skill | Object/goal 초기화 |
|---|---|
| loco | Human은 loco reference. Object는 공통 scene initializer에서 랜덤 배치. 다른 agent가 reference로 고정한 object는 덮어쓰지 않음 |
| pickUp / carryWith | HOLDING target object를 **같은 motion ID/time**의 `get_obj_motion_state()`로 복원. AT goal은 랜덤 |
| putDown | Human/source object는 같은 motion ID/time. 연결된 AT goal은 **같은 clip의 마지막 object 위치**로 정함. 원본처럼 goal Z는 실제 source box 반높이로 조정 |
| sit | SIT target object의 XY/rotation은 `get_obj_motion_state_single_frame()` 기준. 23번처럼 실제 박스 바닥 배치 높이를 적용하고 최종 기하학 검사 |

원본 Carry의 `putDown`은 goal까지 ground placement로 복원한다. 그래서 **이번 ON_TOP RSI에는 putDown을 넣지 않는다.** Source object만 위로 이동시키거나 human 전체를 support 높이만큼 띄우는 식의 임시 retargeting도 하지 않는다. Support-aware putDown RSI는 별도 후속 작업이다. [원본 Carry target reset][srccarrytarget], [원본 SIT object reset][srcsitreset]

SIT reference frame은 반드시 이미 앉은 frame이라는 뜻이 아니다. 원본 시간 샘플러가 반환한 유효 frame을 사용한다. 가변 박스와 reference geometry가 맞는지는 별도로 검사한다.

## 7. Shared-object RSI와 물리 reset

### 7.1 Reference-write 충돌만 먼저 검사

```text
loco       → reference object-pose 요구 없음
pickUp     → HOLDING target object
carryWith  → HOLDING target object
putDown    → HOLDING target object + 연결된 AT goal
sit        → SIT target object
```

**같은 physical object를 둘 이상의 RSI가 서로 독립적인 reference pose로 복원하려면 skill 조합을 재샘플링한다.** Scenario graph는 그대로 유지한다. Graph의 공유 connected component당 advanced RSI를 1명으로 제한하지 않는다.

```text
A: HOLDING(O0)+AT(O0,G0), B: SIT(O0)
A RSI=carryWith, B RSI=sit
→ O0 reference-write 중복: RSI 조합 재샘플링

A: HOLDING(O0)+AT(O0,G0), B: HOLDING(O1)+ON_TOP(O1,O0)
A RSI=carryWith, B RSI=carryWith
→ O0/O1을 각각 복원: reference-write 충돌은 없음
→ 실제 배치의 기하학 검사는 여전히 필요
```

유효한 RSI joint distribution은 위 template별 분포를 proposal로 쓰고 충돌 조합을 제외한 **조건부 분포**다. Filter 이후 agent별 실현 비율이 원래 표와 정확히 같다고 주장하지 않는다. Proposal/accepted 비율을 따로 기록한다.

### 7.2 중요한 구현 정정: loco도 원본에서는 object를 랜덤으로 쓴다

위의 “loco → 없음”은 **reference pose 제약이 없다는 뜻**이다. 원본 `_reset_boxes()` / `_reset_objects()`에는 loco 대상 object의 랜덤 배치 코드가 있으므로, 기존 per-agent reset을 순서대로 실행하면 나중 agent가 앞 agent의 reference object를 덮어쓸 수 있다. [원본 Carry box reset][srccarryboxes], [원본 SIT reset][srcsitreset]

따라서 새 실험은 다음 순서를 따른다.

1. Graph와 RSI skill을 먼저 확정하고, object/goal별 reference 요구를 모은다.
2. Reference가 요구한 pose를 우선 확정한다. Reference가 없는 object만 공통 scene sampler가 **물체당 한 번** 랜덤 배치한다.
3. Loco agent의 위치는 이미 확정된 shared object와 다른 human을 고려하여 배치한다. Loco라는 이유로 shared object를 다시 옮기지 않는다.
4. Placement용 goal/platform, 다른 object, human을 포함한 전체 배치를 검사한다. 각 agent가 참조하는 object/goal은 실제 graph binding으로 찾는다.
5. Accepted scene state를 취합한 뒤 기존 단일-commit 방식으로 simulator에 적용한다. Rejected 시도에서 나온 AMP metadata를 남기지 않는다.

Ref-write 충돌, 물리 침투, asset/reference 크기 불일치는 별도 실패 사유다. 재시도 횟수는 유한하게 두고, 한도를 넘기면 진단 가능한 오류/실패를 기록한다. 숨은 all-loco fallback이나 graph 변경으로 성공한 것처럼 처리하지 않는다.

RSI가 이미 일부 목표를 달성한 상태를 만드는 것은 허용한다. 특히 putDown의 후반부 reference를 “초기 success”라는 이유로 모두 제거하지 않는다. 이는 모든 학습 샘플을 처음부터 실행시키는 평가와 구분해야 한다.

23번에는 carryWith 초기화의 box-speed penalty가 컸던 진단 기록이 있다. 이번 placement template에서 재사용하더라도 해결된 문제라고 간주하지 말고, reference 정렬과 초기 box speed를 검증한다. [기록][srcchangelog]

## 8. AMP motion pool — CLIMB 완전 제외

RSI 분포와 AMP discriminator demo 분포는 별개로 유지한다.

```yaml
# 새 실험의 skill 순서
skill: [loco, sit, omomo, pickUp, carryWith, putDown]

# 초기 기본값: 23번 combined AMP에서 climb의 0.2를 제외하고 재정규화.
# 원본 단일-task TokenHSI의 AMP 비율과 같다는 뜻이 아니다.
skillDiscProb: [0.25, 0.25, 0.125, 0.125, 0.125, 0.125]

templateRsi:
  HOLDING:        [0.5, 0.0, 0.0, 0.5, 0.0, 0.0]
  SIT:            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
  HOLDING_AT:     [0.5, 0.0, 0.0, 0.1, 0.3, 0.1]
  HOLDING_ON_TOP: [0.5555555555555556, 0.0, 0.0, 0.1111111111111111, 0.3333333333333333, 0.0]
```

위 `templateRsi`는 **새로 구현할 설정 계약**이며, 기존 parser가 이미 지원한다는 뜻이 아니다. AMP 비율은 별도 최적값이 확정되지 않아 정한 초기 기본값이다. PPO/AMP loss coefficient는 이 표 때문에 바꾸지 않는다.

Motion YAML의 각 key와 실제 reference 파일을 확인한다. `climb/climbNoRSI`를 `skill` 목록에서만 빼고 combined YAML에는 남겨 loader가 계속 읽는 구현은 금지한다. `omomo/pickUp/carryWith/putDown`은 carry용 원본 pool을 유지한다. [23번 AMP 설정][src23]

## 9. 구현 산출물과 기존 코드 점검

별도 새 실험을 만든다. 제안 이름은 `approach_scenario_no_climb`이며, 기존 명명 규칙에 맞춰 조정하되 23/24번 config를 덮어쓰지 않는다.

| 대상 | 요구사항 |
|---|---|
| 새 환경 YAML | 4 templates, 2 humans/3 objects/2 goals, max 2 edges/agent, capacity 4, semantic-only, templateRsi |
| Motion YAML | CLIMB 없는 combined pool; 실제 데이터 연결 확인 |
| Graph sampler/validator | 전체 binding 랜덤, invalid filter, owner/source 일관성, required-goal 및 paired-placement mapping |
| Reward runtime | 기존 current success 유지 + paired placement current-success 포화; raw 상태 분리 |
| Reset | Template 분류, ref-write 검사, object별 단일 초기화, 전체 물리 배치 검사 |
| Network/rollout | `[valid, src, dst, relation, owner]` 5-field packet, context encoder/aux loss 없음, obs/next_obs 및 minibatch에 graph binding 보존 |
| 실행 스크립트 | 전용 train/test/VNC와 분리된 output |
| Checkpoint | 새 schema/version·task metadata; 기본 scratch, 기존 23/24 checkpoint의 silent resume/transfer 금지 |
| 문서 | `changelog.md`, `markdowns/config.md`, 필요 시 `markdowns/structure.md` 갱신 |

우선 확인할 코드 위치:

```text
tokenhsi/utils/edge_stage1_spec.py
tokenhsi/utils/edge_ontop_spec.py
tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py
tokenhsi/env/tasks/multi_agent/edge_ontop_task.py
tokenhsi/env/tasks/multi_agent/edge_stage1_reward.py
tokenhsi/env/tasks/multi_agent/edge_context_reward.py
tokenhsi/env/tasks/multi_agent/edge_interaction_reward.py
tokenhsi/learning/multi_agent/amp_network_builder_ma.py
tokenhsi/learning/multi_agent/scene_normalizer.py
```

현재 schema 7/semantic-only가 CLIMB-only variant에 묶인 부분, primitive-only relationRsi validation, `max(relation_id)` 분기, owner 전용 box/goal reset, 공용 evaluator의 무조건적인 CLIMB config 접근을 함께 점검한다. 필요한 새 경로를 만들되 기존 실험의 결과 정의는 변경하지 않는다. [구조 안내][srcstructure]

## 10. 검증 및 완료 조건

### CPU / 단위 테스트

- 네 template 생성, edge 상한, pair의 동일 owner/source, permutation 불변성, padding 0 처리.
- Invalid filter의 각 조건과 §4.3 valid 예시를 fixture로 검증. 같은 object를 공유한다는 이유만으로 모든 coupling을 제거하지 않음.
- 자기 success 포화, paired placement 포화, success 이탈 시 해제, 다른 owner/support로 잘못 전파되지 않음, raw success 불변.
- AT/ON_TOP 성공 후 raw HOLDING이 false여도 placement-pair의 완료 판정이 유지됨. 서로 다른 시점의 과거 success를 scene success로 합치지 않음.
- Template별 RSI proposal 확률·0 확률 skill·sum=1, edge 순서와 무관한 template 분류.
- 같은 object의 두 reference-write는 reject, 서로 다른 object의 reference-write는 허용. **loco reset이 이미 확정된 reference pose를 덮어쓰지 않음.**
- AT putDown target은 같은 clip의 마지막 object 위치와 지정 goal binding을 사용. ON_TOP putDown은 0 확률.
- 새 실험에서 CLIMB graph를 거부하고 CLIMB motion 로드·reward 계산·AMP demo sampling이 0임.
- Semantic-only network forward/backward, 저장된 graph packet의 minibatch 일관성, 새 checkpoint 저장/로드와 legacy 분리.

### 짧은 simulator 검증

실행 가능하고 명시된 운영 범위에서만 수행한다. 학습 smoke는 기존 기준인 **2048 environments**를 유지하고 반복 수만 줄인다. 전용 `_check` output을 사용한다. 기존 프로세스를 임의 종료·재시작하거나 본학습을 자동 시작하지 않는다. GPU 번호는 사용자가 지정한 값 또는 기존 운영 지침을 따르고 실행 전 점유를 확인한다. [운영 규칙][srcagents]

로그에는 template proposal/accepted 비율, object·goal binding 비율, shared-object 비율, invalid 사유, RSI proposal/accepted 비율, reference-write 충돌률, 물리 reset 재시도·실패, 초기 box speed·penetration, raw success·saturation·required-goal scene success를 구분해 남긴다.

짧은 실행 성공을 장기 수렴이나 coordination 성능 검증으로 보고하지 않는다. 검증하지 못한 항목과 실제 실행 결과를 구분하여 전달한다.

## 11. 추론과 후속 범위

추론에서는 같은 policy에 **현재 scenario graph**를 주고, 완료 후 다음 graph로 교체하여 긴 task를 구성한다. 전환 시 물리 world state는 유지하며, graph 관련 mapping·success buffer만 새 scenario 기준으로 갱신한다. 미래 전체 graph, LTL/automaton, temporal context 학습을 추가하지 않는다.

이번 단계는 네 template의 joint scenario 학습 및 기본 평가를 우선 완성한다. 긴 sequence의 전환 상태에서 다음 scenario가 실제로 수행되는지는 별도 검증 대상이다. 학습에서 본 시작 상태와 다를 수 있으므로 단일 scenario 성공만으로 긴 sequence 성공을 주장하지 않는다.

**CLIMB은 현재 리워드 불안정으로 보류한다. 리워드 및 단독 수행을 안정화한 뒤에만 template·RSI·AMP·binding validity·평가를 함께 재검토하여 재도입한다. 이번 작업에서는 CLIMB과 HOLDING+CLIMB을 구현·학습·평가하지 않는다.**

---

## 참조 코드

위 링크는 구현 근거를 확인하기 위한 경로다. `src23/src24`는 `approach_clean_scenario`의 변경 전 기준이며, 원본 RSI 링크는 대화에서 확인한 저장소 내 `main/TokenHSI` 복사본이다. 실행 전 실제 checkout의 코드와 대조하고, 구현 결과에는 사용한 commit을 기록한다.

[src23]: https://github.com/KSH3880/CVPR2027/blob/approach_clean_scenario/tokenhsi/data/cfg/multi_agent/approach_distance_edge_context_stage1_primitives_rsi.yaml
[src24]: https://github.com/KSH3880/CVPR2027/blob/approach_clean_scenario/tokenhsi/data/cfg/multi_agent/approach_distance_stage1_climb_rsi.yaml
[srcreward]: https://github.com/KSH3880/CVPR2027/blob/approach_clean_scenario/tokenhsi/env/tasks/multi_agent/edge_context_reward.py
[srcinteraction]: https://github.com/KSH3880/CVPR2027/blob/approach_clean_scenario/tokenhsi/env/tasks/multi_agent/edge_interaction_reward.py
[srccarrycfg]: https://github.com/KSH3880/CVPR2027/blob/main/TokenHSI/tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_carry.yaml
[srcsitcfg]: https://github.com/KSH3880/CVPR2027/blob/main/TokenHSI/tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_sit.yaml
[srccarryreset]: https://github.com/KSH3880/CVPR2027/blob/main/TokenHSI/tokenhsi/env/tasks/basic_interaction_skills/humanoid_carry.py#L974-L1014
[srccarrytarget]: https://github.com/KSH3880/CVPR2027/blob/main/TokenHSI/tokenhsi/env/tasks/basic_interaction_skills/humanoid_carry.py#L435-L501
[srccarryboxes]: https://github.com/KSH3880/CVPR2027/blob/main/TokenHSI/tokenhsi/env/tasks/basic_interaction_skills/humanoid_carry.py#L723-L797
[srcsitreset]: https://github.com/KSH3880/CVPR2027/blob/main/TokenHSI/tokenhsi/env/tasks/basic_interaction_skills/humanoid_sit.py#L717-L757
[srcchangelog]: https://github.com/KSH3880/CVPR2027/blob/approach_clean_scenario/changelog.md
[srcstructure]: https://github.com/KSH3880/CVPR2027/blob/approach_clean_scenario/markdowns/structure.md
[srcagents]: https://github.com/KSH3880/CVPR2027/blob/approach_clean_scenario/AGENTS.md
