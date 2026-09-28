# Codex 구현 지시서 — Stage 3: Long-Horizon Cooperative Fine-tuning

**작성일: 2026-09-25**  
**문서 상태: 구현 요구사항. 이 문서 작성 과정에서 저장소 코드 수정이나 policy 학습을 실행한 것은 아니다.**  
**함께 읽을 문서: `CODEX_STAGE2_COORDINATION_HANDOFF.md`**

> **Stage 1은 이미 구현되어 있다. Stage 2의 모델·학습 경로를 보존한 채, Stage 2 체크포인트에서 시작하는 선택적 Stage 3 학습과 multi-phase 평가를 추가한다.**
>
> **Stage 2:** 하나의 cooperative phase 안에서 다른 역할을 고려하는 행동을 학습한다.  
> **Stage 3:** 같은 모델로 여러 cooperative phase를 reset 없이 연결하여, 이전 행동의 실제 종료 상태에서 다음 행동으로 넘어가는 능력을 추가로 학습한다.
>
> 개발/config 명칭은 **Stage 3**로 명확하게 둔다. 논문에서는 **Long-horizon cooperative adaptation / sequential composition**이라는 확장 실험으로 설명할 수 있다. 단계 이름과 논문의 핵심 contribution은 별개다.

---

## 0. 먼저 읽을 최종 결정

| 항목 | Stage 3 요구사항 |
|---|---|
| 시작 checkpoint | 사용자가 지정한 **Stage 2 checkpoint**. Stage 1에서 다시 시작하지 않는다. |
| Network | Stage 2와 동일. 새 planner, phase token, dependency DAG, stand-up skill을 기본으로 추가하지 않는다. |
| Frozen | Stage 1에서 물려받은 actor tokenizer·edge encoder·Transformer와 actor observation RMS. |
| Trainable | Stage 2 grounded-edge projection, human-to-edge cross-attention, **action head 전체**, critic, 기존 AMP 학습 경로. |
| Head 초기화 | **Stage 2에서 학습된 값 그대로** 복사. `c` 입력 weight를 다시 0으로 만들지 않는다. |
| 학습 단위 | 하나의 episode 안에 여러 cooperative phase. 첫 toy는 2 agents, 2 phases. |
| Phase 내부 | 모든 참여 agent의 현재 role edge를 **처음부터 함께** 입력한다. Agent별 순차 지급은 금지. |
| Phase 사이 | 주어진 순서에 따라 전체 edge set을 교체한다. Physical state는 그대로 유지한다. |
| RSI | **Episode 시작 시 한 번만**. Phase 전환 시 RSI, teleport, object reset, AMP history 재초기화 금지. |
| Phase 완료 | 해당 phase의 required goals가 같은 시점에 만족되는지 판정하고, 짧은 유지 조건 후 자동 전환. |
| PPO | 중간 `phase_completed`와 `episode_done`을 분리한다. Phase 경계 자체로 return을 끊지 않는다. |
| 첫 LR | Coordination과 action head 모두 `2e-5`를 출발점으로 사용. 검증 없이 10배 낮추지 않는다. |
| 비교 | 동일한 phase runner에서 **Stage 2 zero-shot**과 **Stage 3 fine-tuned**를 비교한다. |
| 확장 | Agent/edge 수 증가와 phase 수 증가를 각각 평가하고, 이후 두 축을 결합한다. |

### 기존 Stage 2 문서와의 관계

Stage 2 문서 11절은 long-horizon을 “별도 평가/선택적 확장”으로 남겨두었다. **이 문서는 그 선택적 확장의 구체적인 구현 요청**이다. Stage 2에 Stage 3를 강제로 합치라는 뜻이 아니다. [S2: §11]

다음 세 실행 모드를 분리해 구현한다.

```text
stage2_zero_shot_eval: Stage 2 weights 그대로 multi-phase 평가, 추가 학습 없음
stage3_train:          Stage 2 weights에서 시작해 multi-phase fine-tuning
stage3_eval:           학습된 Stage 3 weights로 평가
```

**숫자 기본값과 새로운 API/config 이름은 첫 구현을 위한 제안이다.** 기존 프로젝트 schema에 맞게 구현하고 실제 채택값을 보고한다. 이미 합의된 설계와 검증되지 않은 하이퍼파라미터를 구분한다.

---

## 1. 구현 전 확인 및 범위

### 1.1 Stage 2를 선행 조건으로 사용

1. 실제 working tree, Stage 2 실행 config와 checkpoint의 architecture metadata를 확인한다.
2. Stage 2가 아직 구현되지 않았다면 먼저 Stage 2 지시서를 적용해야 한다. Stage 1 모델로 Stage 3를 대체 구현하지 않는다.
3. 기존 Stage 1/2 config, checkpoint loading, single-phase 실행 경로를 그대로 보존한다.
4. 사용자의 수정사항을 checkout/reset하거나 오래된 원격 snapshot으로 덮어쓰지 않는다.
5. Stage 3 기능은 별도 config/flag로 활성화한다.

Stage 2 문서의 코드 참조 기준은 `KSH3880/CVPR2027`, branch `approach_clean_scenario_2`, commit `1c119b41b47bdbe49ad0a0fea7802da3cafca5ef`이다. **이는 선행 문서의 snapshot이며, 현재 로컬 Stage 2 구현을 이번 문서에서 다시 확인했다는 뜻은 아니다.** [S2: §1]

### 1.2 Stage 3가 배우는 것과 배우지 않는 것

학습 대상:

- SIT를 마친 자세에서 다음 CARRY를 시작하기.
- CLIMB를 마친 실제 위치/접촉 상태에서 다음 과업에 접근하기.
- CARRY/STACK 이후 다음 task에 맞게 놓기·이동·자세 전환하기.
- Phase가 바뀌어 역할과 공유 대상이 바뀌어도 현재 cooperative task 수행하기.

기본 범위에서 제외:

- 어떤 phase를 먼저 할지 자동 계획하기.
- Agent의 역할을 자동 배정하기.
- Actor가 입력받지 않은 미래 phase를 예측해 미리 준비하기.
- 없던 물리 스킬을 phase runner가 animation으로 대신 수행하기.

**Actor에는 현재 phase의 edge set만 제공한다.** 미래 역할을 현재 행동에 반드시 반영해야 하는 과업은 이 첫 설계의 범위를 벗어난다. 이를 몰래 미래 정보를 주거나 보상만으로 해결했다고 주장하지 않는다.

---

## 2. 모델은 Stage 2 그대로 이어받는다

현재 Stage 2 설계를 재사용한다. [S2: §3–5]

```text
현재 scene observation + 현재 phase의 전체 edge packet
                         ↓
             Frozen Stage-1 actor encoder
                         ↓
                전체 post-TF entity tokens
                   /                 \
             Human h_i          src/tgt tokens
                  |                  + edge64
                  |                     ↓
                  Q             grounded edge e_bar
                   \                  K,V
                    shared softmax cross-attention
                                 ↓
                      coordination context c_i
                                 ↓
                            [h_i | c_i]
                                 ↓
                trainable action head 128→1024→512→32
                                 ↓
                         각 human의 action
```

수식:

```text
z, h = frozen_actor_encoder(obs, current_edges)
e_bar_j = phi([edge64_j, z_src_j, z_dst_j])
c_i = cross_attention(Q=h_i, K=e_bar, V=e_bar)
a_i = action_head([h_i, c_i])
```

### 2.1 Freeze/train 범위

| 구성 요소 | Stage 3 처리 |
|---|---|
| Actor entity tokenizer / type embedding / semantic edge encoder / GTA Transformer | Stage 2 checkpoint의 값으로 load, 계속 freeze |
| Actor observation RMS | Stage 2 통계를 load, 계속 update 정지 |
| Grounded-edge projection `phi` | Stage 2에서 load, 계속 train |
| Human-to-edge softmax cross-attention | Stage 2에서 load, 계속 train |
| Action head 전체 | Stage 2에서 load, 계속 train |
| Actor sigma | Stage 2의 fixed/learnable 설정 유지 |
| Critic encoder / value head | 호환 Stage 2 값을 load하고 Stage 3 return에 맞게 train |
| AMP discriminator / AMP RMS / motion library | 기존 Stage 2 경로 유지. 임의로 교체·freeze하지 않는다. |

Frozen actor encoder만 `no_grad()`/detach 경계에 둔다. Cross-attention과 head를 함께 `no_grad()`에 넣으면 안 된다. `model.train()`이 actor RMS 업데이트를 다시 켜는지도 검사한다. Gradient 기록과 parameter freeze의 구분은 PyTorch autograd 문서를 참고한다. [T2]

### 2.2 Stage 1 → 2 초기화 코드를 다시 실행하지 않는다

Stage 1에서 Stage 2를 처음 만들 때만 첫 weight를 `[W_stage1, 0]`으로 확장했다.

**Stage 2 → 3에서는 이미 학습된 128D-input head를 그대로 가져온다.**

```text
금지:
  W_c = 0으로 재초기화
  action head를 Stage 1 값으로 덮어쓰기
  cross-attention / phi를 random init
  input dimension을 128→192로 다시 확장
```

동일한 현재 obs/graph/RMS에서, optimizer update 전 Stage 3의 action mean은 Stage 2와 수치 허용오차 내에서 같아야 한다.

---

## 3. Phase는 하나의 완전한 cooperative scenario다

### 3.1 한 phase에 모든 역할을 동시에 준다

```text
Phase 1:
  A: HOLDING(Ha,Oa) + AT(Oa,Ga)
  B: SIT(Hb,Oa)

동시에 제공하는 입력:
  { HOLDING(Ha,Oa), AT(Oa,Ga), SIT(Hb,Oa) }
```

A 완료 후 B의 SIT를 새로 켜는 것이 아니다. Phase 1이 진행되는 동안 세 edge를 함께 준다. A/B 중 누가 접근·대기·유지·이탈하는지는 policy가 수행한다.

**다음 cooperative scenario로 넘어갈 때만** edge set 전체를 교체한다.

### 3.2 Agent당 하나의 bundle 규칙 유지

```text
HOLDING             : 1 edge
SIT                 : 1 edge
CLIMB               : 1 edge
CARRY               : HOLDING + AT       = 2 edges
STACK               : HOLDING + ON_TOP   = 2 edges
```

- 한 phase의 같은 agent에게 `SIT + CLIMB` 같은 두 bundle을 동시에 주지 않는다.
- `SIT → CLIMB`는 서로 다른 두 phase에 놓는다.
- Phase가 여러 개여도 **미래 phase의 edge를 현재 입력에 누적하지 않는다.**
- `CARRY/STACK`은 sampler용 bundle 명칭이지 새로운 relation token이 아니다.

### 3.3 첫 toy의 실제 sequence 예시

**예시 A — 앉은 상태에서 운반 역할로 전환**

```text
Phase 1: place_sit
  A: HOLDING(Ha,Oa) + AT(Oa,Ga)
  B: SIT(Hb,Oa)

  → Phase 1 공동 완료. B는 실제로 Oa에 앉아 있는 상태.
  → reset 없이 다음 phase.

Phase 2: place_climb, role 교환
  B: HOLDING(Hb,Ob) + AT(Ob,Gb)
  A: CLIMB(Ha,Ob)
```

B는 앉은 자세에서 일어나 Ob에 접근해 운반해야 한다. A는 운반하던 역할에서 Ob를 이용하는 역할로 바뀐다. **Phase 2에서도 세 edge는 처음부터 동시에 입력한다.**

**예시 B — 오른 상태에서 다음 과업으로 전환**

```text
Phase 1: place_climb
  A: HOLDING(Ha,Oa) + AT(Oa,Ga)
  B: CLIMB(Hb,Oa)

Phase 2: place_sit, role 교환
  B: HOLDING(Hb,Ob) + AT(Ob,Gb)
  A: SIT(Ha,Ob)
```

두 번째 예시는 B가 실제로 내려올 수 있는 낮은 support, 접근 가능한 Ob, 충분한 공간을 갖춘 경우에만 사용한다. “토큰으로 표현 가능”하다고 물리적 실행 가능성이 보장되는 것은 아니다.

첫 예시들은 Oa/Ob/Ox 세 object를 계속 유지하고, 필요한 object는 episode 시작부터 scene에 존재한다. 중간에 object를 생성하거나 이동시켜 전환을 쉽게 만들지 않는다.

---

## 4. Program / phase specification

### 4.1 외부 runner가 순서를 보유한다

```text
Program = [Phase_1, Phase_2, ..., Phase_P]
Phase_k = 현재 cooperative edge set + required goals + transition rule
```

Program 순서와 phase index는 **환경 실행기·sampler·로그용 metadata**다. 이를 actor의 scenario-ID embedding이나 phase positional encoding으로 넣지 않는다.

새 spec 예시 — 기존 graph parser에 맞게 실제 schema를 구현할 것:

```yaml
program_id: sit_to_carry_role_swap
phases:
  - name: place_sit
    edges:
      - {id: p1_hold_a, src: Ha, dst: Oa, relation: HOLDING, owner: Ha}
      - {id: p1_at_a, src: Oa, dst: Ga, relation: AT, owner: Ha}
      - {id: p1_sit_b, src: Hb, dst: Oa, relation: SIT, owner: Hb}
    required_goal_edges: [p1_at_a, p1_sit_b]

  - name: place_climb_role_swap
    edges:
      - {id: p2_hold_b, src: Hb, dst: Ob, relation: HOLDING, owner: Hb}
      - {id: p2_at_b, src: Ob, dst: Gb, relation: AT, owner: Hb}
      - {id: p2_climb_a, src: Ha, dst: Ob, relation: CLIMB, owner: Ha}
    required_goal_edges: [p2_at_b, p2_climb_a]
```

Placement bundle의 HOLDING은 배치 완료 후 release를 허용하는 Stage 2 saturation 규칙을 그대로 따른다. Required goal에 모든 HOLDING을 무조건 포함해 끝까지 잡고 있도록 바꾸지 않는다.

### 4.2 Entity identity는 episode 전체에서 유지

- Ha/Hb/Oa/Ob/Ga/Gb는 episode 전체에 걸쳐 동일한 실제 entity를 가리킨다.
- Phase가 바뀌어도 actor에 맞춰 physical object index를 재배열하거나 ownership reset을 하지 않는다.
- 바뀌는 것은 **task의 src/dst/owner 연결**이다.
- 기존 `_logical_box_order` 등 logical/physical mapping은 episode 시작에 정한 것을 유지한다.
- Goal reference를 바꾸는 것과 physical object/support platform을 옮기는 것은 다르다. Virtual goal/marker 변경은 명시된 program에 따라 가능하지만 collidable geometry를 teleport하지 않는다.
- 첫 toy는 모든 goal 위치를 episode 시작에 정하고 phase마다 binding만 선택하는 구성이 권장된다.

### 4.3 Capacity는 phase 개수의 합이 아니다

Run별 entity 수와 graph padding capacity는 고정하고, phase마다 valid mask와 edge 내용을 바꾼다.

```text
Train: 2 humans, 3 objects, 2 goals, E_capacity=4
       phase가 2개여도 현재 입력 capacity는 4

Eval:  4 humans, 4개 이상의 objects, 4 goals, E_capacity=8
       phase가 4개여도 현재 입력 capacity는 8
```

`E_capacity >= max_k(|E_k|)`이면 된다. 모든 phase의 edge 개수를 합친 길이로 actor 입력을 펼치지 않는다. N/E가 늘어나는 평가에서는 Stage 2의 variable-count parser와 shared network를 그대로 사용한다.

---

## 5. 완료, 유지, 새 목표의 의미

### 5.1 Phase 완료 조건

해당 phase의 required goal들이 **같은 control step에서 모두 만족**해야 한다. 과거 어느 시점에 A가 성공했고 다른 시점에 B가 성공했다는 것만으로 완료하지 않는다.

첫 구현 기본값:

```text
joint_success = all(current_success[required_valid_edges])
성공 유지 시간 = 0.2 seconds
필요 step 수 = max(1, ceil(0.2 / control_dt))
```

기존 Stage 2 evaluator에 확정된 유지 시간이 있으면 그것을 우선하며, Stage 2 zero-shot과 Stage 3 평가에 동일하게 적용한다. 0.2초는 제안 기본값이다.

- 유지 조건을 채우면 다음 control action 전 phase를 자동 전환한다.
- A만 성공했다고 A를 멈추거나 controller를 끄지 않는다.
- 모든 agent는 매 step 자기 action을 계속 출력한다.
- 실패/낙상 판정이 같이 발생했다면 성공보다 우선한다.
- Joint success와 timeout이 같은 step에 발생하면, failure가 없는 경우 성공 유지 조건 충족 여부를 먼저 판정한다.

### 5.2 과거 완료와 계속 유지할 목표를 구분

```text
Phase 1의 SIT 완료
→ Phase 2에서 B가 일어나는 것은 정상이다.
```

따라서 모든 과거 relation을 계속 성공 상태로 요구하면 안 된다.

반대로 이전 object 배치를 이후에도 유지해야 한다면 그 요구는 task specification에 명시되어야 한다. 첫 구현에서는:

1. 다음 phase에도 유지할 관계를 **유효한 현재 bundle 안에서** 표현할 수 있으면 포함한다.
2. 그 때문에 agent당 한 bundle/최대 2-edge 규칙과 충돌하면 해당 program은 첫 toy에서 제외한다.
3. Actor가 모르는 숨은 유지 constraint를 reward에만 추가하지 않는다.

Old phase 완료 여부는 program progress로 기록한다. 이것이 old relation reward의 영구 latch나 ongoing success를 의미하지는 않는다.

### 5.3 새 phase가 이미 완료 상태인 경우

- 실제 state에서 새 relation success를 평가한다. 강제로 false로 만들지 않는다.
- 처음부터 모두 만족하는 phase는 `pre_satisfied_on_entry`로 기록한다.
- 다음 physics/control step들에서 같은 유지 조건을 확인한 뒤 전환하며, 한 step에서 while-loop로 여러 phase를 연쇄 통과시키지 않는다.
- 샘플러는 trivially pre-satisfied인 sequence를 가능한 한 피하되, 진행 중인 물리 상태를 resample해서 고치지 않는다.
- 평가에서 pre-satisfied phase와 실제 수행한 phase를 분리한다. 아무 행동 없이 통과한 것을 transition learning의 증거로 세지 않는다.

---

## 6. Phase runner — 물리 reset과 완전히 분리

### 6.1 Env별 상태

Vectorized env마다 독립적으로 관리한다.

```text
program / program_id
phase_index
phase_steps
episode_steps
phase_success_streak
phase_ever_success / phase_final_success
program_completed
current graph packet / current required-goal mask
termination_reason
```

여러 env가 서로 다른 phase에 있어도 하나의 batch로 처리해야 한다. 일부 env가 phase를 전환했다고 batch 전체 graph를 갈아끼우지 않는다.

### 6.2 한 control step의 처리 순서

```text
1. obs_t에는 E_k가 들어 있다.
2. policy가 obs_t로 action_t / logprob_t / value_t를 계산한다.
3. action_t를 적용하고 physics를 한 step 진행한다.
4. 실제 s_{t+1}를 refresh한다.
5. E_k 기준 reward_t, success, failure, phase timeout을 계산한다.
6. E_k에서 phase 완료가 확정되면:
   a. 마지막 phase → program success / episode termination.
   b. 다음 phase 존재 → physical state 그대로 E_{k+1}로 교체.
7. 교체한 E_{k+1} 기준 relation state와 task-specific baseline을 초기화한다.
8. obs_{t+1}를 새 graph로 생성한다.
9. PPO에 (obs_t, action_t, reward_t, true obs_{t+1}, masks)를 저장한다.
```

**Boundary transition의 reward는 E_k, 다음 action과 next value의 observation은 E_{k+1}에 대응한다.** 새 graph로 이전 action의 reward를 계산하면 안 된다.

### 6.3 Phase 전환 시 바꿔도 되는 것

- Current graph packet와 required-goal mask.
- Graph에서 유도되는 owner/bundle/target binding과 관련 reward-sharing mask.
- Phase index, phase timer, success streak, phase-local diagnostics.
- 새 relation의 현재 phi/success 및 필요한 task-specific progress baseline.
- Program이 명시한 virtual target/reference metadata.

### 6.4 Phase 전환 시 보존할 것

- Human root pose/velocity, DOF position/velocity, object pose/velocity.
- 현재 grasp/contact/support 상태와 물리 actor identity.
- Actor observation RMS와 policy weights.
- **AMP observation history** 및 실제 motion의 연속성.
- Episode timer와 program progress.
- 물리 속도/가속도 penalty에 필요한 실제 이전 state history.

**Graph-relative progress history**는 새 target에 맞춰 다시 잡지만, object velocity 추정처럼 **물리 identity 기준 history**까지 지우지는 않는다. 목표 변경 자체로 가짜 progress가 생기거나, 반대로 실제 velocity가 0으로 리셋되는 일을 막는다.

### 6.5 Reset 함수를 재사용하지 않는다

`_reset_envs`, `_reset_actors`, `_reset_boxes`, `_init_amp_obs` 등 physical reset 경로를 phase 전환 함수에서 호출하지 않는다.

새 graph용 cache만 갱신하는 API를 둔다. 예를 들어 `apply_phase_graph(env_ids, phase_specs, preserve_physics=True)`처럼 역할을 제한한다. 함수명은 제안이며 실제 최신 Stage 2 코드에 맞춘다.

Phase 완료가 PPO rollout 마지막 step에 발생해도 예외는 없다. 반환한 next observation은 전환 완료된 새 phase여야 한다.

---

## 7. RSI — Stage 3의 핵심 규칙

### 7.1 Episode 시작

Program의 **첫 phase**를 먼저 선택한 뒤 Stage 2의 joint-consistent RSI를 적용한다. Stage 2에서 정한 bundle별 확률과 공유 물체 초기화 규칙을 재사용한다. [S2: §7]

```text
program 선택
→ Phase 1 graph 확정
→ Phase 1 bundle 기준 joint RSI
→ shared object를 한 번만 초기화
→ AMP history / relation baseline 초기화
→ episode 시작
```

첫 phase의 downstream/full success를 RSI로 미리 완성해놓지 않는 기본 평가 원칙도 유지한다. Carry warm-start의 초기 HOLDING 성공은 허용한다.

### 7.2 두 번째 phase 이후

**RSI를 하지 않는다.**

```text
Phase 1에서 B가 SIT 완료
→ B는 앉아 있는 실제 상태
→ Phase 2 CARRY edge가 들어옴
→ 동일 policy가 일어서기·접근·집기를 실제로 수행
```

CLIMB 이후도 동일하다. 내려오기, 놓기, 이동 같은 전환 동작이 자동으로 학습된다고 보장하지 않는다. **이런 이전 종료 상태에서 다음 task를 수행하는 경험을 추가하는 것이 Stage 3의 목적**이다.

### 7.3 금지되는 우회

- Phase 시작마다 loco RSI를 재추출.
- Human을 서 있는 기본 pose로 순간 이동.
- 손을 강제로 release하거나 object를 goal에 snap.
- 자동 STAND_UP phase / scripted stand-up animation 삽입.
- Phase가 바뀌었다는 이유로 AMP history를 motion clip으로 덮어쓰기.
- 기존 skill이 다음 task를 시작하기 어려우면 scene을 다시 배치해서 성공 처리.

### 7.4 전환 구간 경험이 부족하면

우선 두-phase program을 짧게 고정하고 자연스럽게 첫 phase를 완료하는 비율을 높인다. 그래도 transition 표본이 부족하면 **선택적 후속 기능**으로 실제 rollout의 phase-end snapshot에서 시작하는 초기화를 검토할 수 있다.

조건:

- 사용 시 episode **시작점**에서만 복원한다. 진행 중 phase 경계에서 복원하지 않는다.
- Scene 전체 human/object state, velocity, graph binding, AMP history를 함께 보존한다.
- Snapshot은 초기 state용이다. 오래된 action/logprob로 PPO update를 수행하지 않는다.
- Full sequence 평가와 snapshot-start 진단을 분리한다.
- Joint state를 안전하게 복원할 수 있는지 검증 전에는 필수 구현으로 넣지 않는다.

---

## 8. Stage 3 reward

### 8.1 현재 phase의 Stage 2 reward를 재사용

```text
current-phase local edge reward
+ current shared-object 관계에 따른 기존 reward sharing
+ 기존 AMP / collision / power 등
```

Stage 2의 state/progress/success, placement-HOLDING saturation, raw local reward 기반 sharing을 유지한다. 새로운 phase에서는 새 graph로 sharing neighborhood도 갱신한다. 이전 phase partner를 계속 섞지 않는다. [S2: §8]

첫 구현에서는 phase bonus, program bonus, 별도 stand-up reward를 추가하지 않는다. 필요하면 별도 config/ablation으로 넣는다.

### 8.2 끝난 phase 보상을 계속 지급하지 않는다

전환을 만든 마지막 action은 old phase reward를 한 번 받는다. 다음 action부터는 new phase reward만 계산한다.

- 과거 완료 relation의 saturation을 새 edge slot에 그대로 복사하지 않는다.
- 변경된 slot 번호를 동일한 relation instance로 취급하지 않는다.
- Program completion을 위해 phase마다 success를 조작하지 않는다.

### 8.3 진행을 늦추는 정책도 진단한다

Dense per-step reward가 높으면 일부 상태에 오래 머물 유인이 생길 수 있다. 따라서 total return만 보지 말고 phase completion 수와 시간도 함께 확인한다.

첫 대응은 phase completion 자동 전환, timeout, joint success 구현을 검증하는 것이다. 이후 필요 시 completion event bonus 또는 작은 time cost를 실험하되, 값과 변경 이유를 기록하고 모든 평가 조건을 고정한다. 전환 보상은 phase마다 한 번만 지급해야 한다.

---

## 9. PPO / GAE — 중간 phase는 terminal이 아니다

### 9.1 두 신호를 분리

```text
phase_completed: 이번 cooperative scenario 완료
terminated:     최종 program 완료 또는 실제 terminal failure
truncated:      외부 계산/시간 예산 때문에 episode 중단
```

중간 phase가 끝나면 `phase_completed=True`여도 `terminated=False`, `truncated=False`다. Env 전체 reset mask도 false다.

Termination에서는 value bootstrap을 하지 않고 외부 truncation에서는 실제 마지막 state value로 bootstrap하는 원칙을 따른다. 현재 rl_games/IsaacGym API가 Gymnasium과 다를 수 있으므로 필드 이름을 그대로 가정하지 말고 의미를 매핑한다. [T1]

### 9.2 TD bootstrap과 GAE trace를 구분

구현 의도:

```text
delta_t = reward_t + gamma * bootstrap_mask_t * V(true_next_obs_t) - V(obs_t)
adv_t   = delta_t + gamma * gae_lambda * trace_mask_t * adv_{t+1}
```

| 상황 | bootstrap_mask | trace_mask |
|---|---:|---:|
| 보통 step | 1 | 1 |
| 중간 phase 전환 | 1 | 1 |
| 최종 program success / terminal failure | 0 | 0 |
| 외부 cap 때문에 reset되는 truncation | 1 | 0 |

Truncation 후 새 episode의 advantage를 이전 episode로 이어 붙이지 않는다. Reset 뒤 observation을 `true_next_obs` 대신 쓰지 않는다. PPO rollout segment 마지막에서는 실제 다음 observation으로 value를 bootstrap하고, 아직 수집하지 않은 advantage까지 이어 붙인다고 가정하지 않는다.

**PPO horizon이 32라고 32 step마다 물리 episode를 reset하거나 phase를 종료하지 않는다.** Long episode는 여러 rollout/update를 거쳐 진행할 수 있다.

### 9.3 Reward와 graph packet 정합성

모든 sample은 rollout 당시의 `obs_t`와 graph packet을 저장한다. Phase 전환 후 live env graph로 예전 sample을 다시 인코딩하면 안 된다.

```text
transition 직전 sample:
  obs_t.graph        = E_k
  reward_t           = E_k 기준
  next_obs_t.graph   = E_{k+1}
  phase_completed    = True
  episode_done       = False
```

Buffer에 저장한 graph가 live tensor와 alias되지 않도록 복사/ownership을 확인한다. Minibatch 안에서 여러 phase의 sample이 섞여도 각자의 graph를 재구성해야 한다.

### 9.4 Timeout 의미를 명시

첫 toy의 phase/episode step cap은 **외부 rollout 예산**으로 취급한다. 도달 시 program 평가 결과는 미완료/실패로 집계하되, learner에서는 truncation으로 처리한다.

과업 정의 자체에 엄격한 deadline을 넣는 별도 실험은 terminal 의미와 필요한 시간 observation을 다시 설계해야 한다. 두 timeout 의미를 혼용하지 않는다. [T1]

### 9.5 미래 program이 보이지 않는 critic의 한계

첫 toy는 Stage 2 critic 구조를 유지한다. 하지만 같은 현재 obs/graph라도 남은 program이 다르면 이후 return이 달라질 수 있다. 따라서 이 critic은 숨겨진 program에 대해 평균적인 value를 학습할 수 있으며 완전한 program-state critic이라고 주장하지 않는다.

초기에는 짧고 명시적인 sequence family로 경향성을 확인한다. Value error가 phase/remaining-program별로 체계적으로 커지면, **critic에만** remaining-phase/program context를 주는 옵션을 별도 ablation으로 검토한다. Actor 입력 변경과 혼동하지 말고, context를 실제 rollout sample에 저장한다. 이것은 첫 toy의 필수 architecture 변경은 아니다.

---

## 10. 학습 분포와 curriculum

### 10.1 첫 구현 기본값

```text
Train humans / objects / goals: 2 / 3 / 2
Phase당 agent별 bundle:         1개
Phase당 edge capacity:          4
Episode 구성:                  2-phase program 80%, single-phase retention 20%
Coordination / action-head LR:  2e-5 / 2e-5
Critic / AMP:                  Stage 2 기본값에서 시작
```

80/20과 LR은 출발점이며, 사용자 확정 실험값이 아니다. 실제 Stage 2 config가 다르면 상속값과 override를 모두 저장한다. Single-phase retention도 **현재 policy로 새 rollout**을 생성한다. Stage 2의 오래된 PPO trajectory를 그대로 재학습하지 않는다.

### 10.2 Curriculum 순서

```text
1. 동일 phase runner에서 Stage 2 checkpoint zero-shot baseline 수집
2. 2 agents / 2 phases의 실행 가능한 전환부터 Stage 3 학습
3. Stage 2 single-phase cooperative 성능이 유지되는지 확인
4. 필요하면 학습 sequence 종류 확대
5. 3~4 agents 및 더 긴 phase sequence를 별도 held-out 평가
```

첫부터 agent 수, edge 수, scene geometry, sequence 길이를 모두 동시에 늘리지 않는다. 무엇 때문에 실패했는지 분리할 수 있어야 한다.

### 10.3 Sequence sampling

- Phase별 graph validity뿐 아니라 **이전 phase의 end state에서 다음 phase가 실행 가능한지** 검사한다.
- 사람/object/goal ID binding과 role swap을 적용하되 episode 전체 identity mapping을 유지한다.
- 다음 과업에 필요한 object는 사전에 존재해야 한다.
- Same object를 새 phase에서 다른 agent가 쓰는 것은 허용할 수 있으나, 기존 접촉/위치에서 실제로 전환 가능해야 한다.
- 이전 support가 필요한 상태에서 support를 강제로 제거해야 하는 조합은 첫 toy에서 제외한다.
- 불가능한 graph/전환과 단지 정책이 아직 못 하는 전환을 분리해 로그한다.
- 사람이 정한 explicit evaluation program을 지원한다. Sampling 오류를 live graph 재추출로 조용히 숨기지 않는다.

### 10.4 시간 예산

Phase timeout은 Stage 2의 single-phase budget에서 시작한다. 예를 들어 그것이 600 control steps라면 2-phase episode cap은 기본적으로 1200 steps다. 이는 예시이며 실제 Stage 2 값을 상속한다.

- Phase timer는 phase 시작에만 0으로 갱신한다.
- Episode timer는 phase 전환으로 초기화하지 않는다.
- Episode cap은 기본적으로 각 phase cap의 합이다.
- Phase가 길어졌다고 minibatch/horizon 크기를 자동으로 배수 확대하지 않는다.
- LR을 작게 하는 대신 actual KL, clip fraction, value loss, phase success를 보고 조절한다.

---

## 11. Checkpoint / initialization / resume

### 11.1 세 경로를 명확히 구분

| 실행 | 로딩 및 optimizer |
|---|---|
| Stage 2 zero-shot multi-phase 평가 | Stage 2 weights/RMS load. 모든 학습·통계 update 정지. |
| Stage 2 → 새 Stage 3 학습 | Stage 2 전체 호환 weights/RMS load. **Stage 3 optimizer 새로 생성**. |
| Stage 3 resume | Stage 3 weights, optimizer, scheduler, train step, curriculum 상태 복원. |

다른 head 초기화, layer 확장, Stage 1 import 함수가 실행되지 않게 한다. Compatible weight를 `strict=False`로 조용히 누락하지 않는다.

### 11.2 Normalization

- Actor observation RMS는 Stage 2 값으로 고정한다.
- Value RMS는 Stage 2 값을 가져와 Stage 3 return 분포에 맞춰 update하는 것을 기본으로 한다. Reset 옵션은 별도로 기록한다.
- Graph index/type/valid, GTA pose 등 기존 passthrough 필드는 normalize하지 않는다.
- AMP RMS 동작은 Stage 2 규칙을 유지한다.
- 평가 시 actor뿐 아니라 value/AMP/RMS도 update하지 않는다.

### 11.3 저장할 metadata

```text
stage = stage3
source_stage2_checkpoint + file hash
source/config/code revision
coordination variant / architecture dimensions
freeze policy / optimizer LR groups
train_num_agents / objects / goals / edge_capacity
train_phase_count distribution
program families / train-validation-test split manifest
RSI policy + phase-reset prohibition
phase completion / timeout definitions
additional_finetuning_steps / environment steps
```

일반 학습 checkpoint에 simulator state를 저장하지 않는다면 resume 시 episode를 새로 시작하는 사실을 명시한다. Exact mid-episode resume를 지원한다고 주장하려면 모든 human/object state, phase metadata, AMP history와 RNG까지 복원하는 별도 기능이 필요하다.

---

## 12. 평가 설계

### 12.1 핵심 비교

| Model | Agents / phases | 목적 |
|---|---|---|
| Stage 2 | 학습 규모 / 1 phase | 기본 cooperative 성능 기준 |
| Stage 2, 추가 학습 없음 | 학습 규모 / 2+ phases | Zero-shot sequential baseline |
| Stage 3 | 학습 규모 / 2+ phases | 전환 fine-tuning의 효과 |
| Stage 3 | 3~4 agents / 1 phase | 기존 cardinality 성능 유지 |
| Stage 3 | 3~4 agents / 여러 phases | Spatial/relational + temporal 확장 |

Stage 2와 Stage 3에 동일한 program, seed, 초기 state, timeout, success 유지 조건을 사용한다. Policy가 달라 발생하는 이후 state 차이는 정상이다.

### 12.2 Generalization 축을 구분

```text
Agent-count:      학습보다 많은 human/edge
In-phase graph:   학습에서 보지 않은 동시 role composition
Transition:       학습에서 보지 않은 phase/bundle 전환
Sequence:         익숙한 전환의 새로운 전체 순서 조합
Length:           학습보다 많은 phases
```

Agent ID permutation이나 object binding 변경만으로 새로운 sequence structure라고 세지 않는다. Train/test program을 manifest로 고정하고, test에서 fine-tune하면 그 결과를 zero-shot이라고 부르지 않는다.

3명 이상이 작은 support를 동시에 사용해 물리적으로 모순되는 higher-order graph를 long-horizon 평가에 넣지 않는다. 필요한 feasible-region reward/scene 확장은 별도의 실험 변경이며, Stage 3의 필수 조건이 아니다.

### 12.3 필수 metrics

- **Full-program success rate:** reset 없이 요구된 phase들을 순서대로 완료한 비율.
- Phase reach rate와 reached phase의 conditional success rate를 각각 기록.
- 완료한 phase 수, 실패한 phase index, failure reason.
- Transition별 success: 예를 들어 `SIT→CARRY`, `CLIMB→CARRY`, `CARRY→CLIMB`.
- Phase 진입 후 첫 일정 구간의 fall/collision rate와 다음 task 첫 성공까지 시간.
- Stage 2 single-phase skill/cooperation retention.
- Pre-satisfied-on-entry 비율, RSI 종류별 시작 분포.
- Wall-clock, env steps, PPO updates, physics time / policy time / update time.

Program success는 각 phase를 완료한 event를 기준으로 한다. **과거의 SIT까지 최종 시점에 동시에 만족하라는 뜻은 아니다.** 반면 task가 명시한 지속 유지 조건은 별도로 충족해야 한다.

Conditional phase success만 보고하면 쉬운 첫 phase를 통과한 일부 episode만 평가하게 된다. 전체 시작 episode를 분모로 한 program success와 함께 제시한다. Training seed 변동성과 test episode 수를 기록하고, 수치/영상은 실제 실행 결과만 보고한다.

### 12.4 영상

Phase 번호와 현재 edge/role을 overlay하고, phase 경계에서 끊김 없는 영상을 저장한다. 컷 편집·teleport·scene reset으로 transition을 가리지 않는다. Stage 2 zero-shot과 Stage 3의 동일 program 영상을 나란히 비교할 수 있게 한다.

---

## 13. 변경 파일 / API 후보

아래 경로는 **배치 제안**이다. Stage 2 구현 결과를 먼저 확인하고 이름·상속 관계를 맞춘다. 이미 존재한다고 가정하지 않는다.

```text
신규 후보:
  tokenhsi/utils/edge_stage3_spec.py
    - Program/phase spec, validation, transition sampling, split manifest

  tokenhsi/env/tasks/multi_agent/phase_runtime.py
    - env별 phase index/timer/transition, physical-reset-free graph 교체

  tokenhsi/data/cfg/train/rlg/amp_ma_stage3_long_horizon.yaml
  tokenhsi/data/cfg/multi_agent/approach_stage3_long_horizon.yaml
  tokenhsi/data/cfg/multi_agent/programs/stage3_*.yaml

  tokenhsi/scripts/multi_agent/approach_stage3_train.sh
  tokenhsi/scripts/multi_agent/approach_stage3_test.sh
  tokenhsi/scripts/multi_agent/approach_stage2_long_horizon_eval.sh
  tokenhsi/scripts/multi_agent/approach_stage3_smoke.sh

필요한 경우에만 기존 경로 수정:
  tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py
  tokenhsi/env/tasks/multi_agent/edge_context_task.py
  tokenhsi/env/tasks/multi_agent/edge_ontop_task.py
    - reward → phase transition → new obs 순서, reset/history 분리

  tokenhsi/learning/multi_agent/ma_agent.py
  tokenhsi/learning/multi_agent/ma_players.py
    - transfer/resume, true next obs, GAE masks, metrics

  Stage 2의 graph compiler / reward runtime / checkpoint metadata
    - program-aware 사용, sample별 packet 정합성
```

Network는 Stage 2와 동일하므로, 새 stage 이름만으로 model layer를 복제하지 않는다. Stage 2 graph·reward·RSI 함수를 가능한 한 재사용하되, physical reset 함수에 graph 전환을 억지로 끼워 넣지 않는다.

---

## 14. Config 설계안

다음은 새 옵션의 의미를 보여주는 예시이며, 현재 코드에 바로 전달 가능한 확정 schema는 아니다.

```yaml
stage3:
  enabled: true
  mode: train  # train | stage2_zero_shot_eval | eval
  stage2_checkpoint: null  # 실행 시 명시. 새 학습/zero-shot 평가에서 필요
  stage3_checkpoint: null  # eval/resume에서 명시
  resume: false

  model:
    inherit_stage2_architecture: true
    freeze_actor_encoder: true
    freeze_actor_obs_rms: true
    train_coordination: true
    train_entire_action_head: true
    reinitialize_context_weights: false
    add_phase_token: false
    expose_future_program_to_actor: false

  optimization:
    learning_rate: 2.0e-5
    coordination_lr_scale: 1.0
    action_head_lr_scale: 1.0
    reset_optimizer_on_stage2_import: true
    critic_and_amp: inherit_stage2
    gamma_and_gae_lambda: inherit_stage2

  sampling:
    train_num_agents: 2
    train_num_objects: 3
    train_num_goals: 2
    edge_capacity_per_phase: 4
    one_bundle_per_agent: true
    episode_phase_count_probabilities: {1: 0.2, 2: 0.8}
    program_manifest: null
    random_role_and_entity_binding: true
    preserve_entity_identity_across_phases: true

  transition:
    completion: simultaneous_required_goals
    success_hold_seconds: 0.2
    max_transitions_per_control_step: 1
    switch_only_after_joint_phase_completion: true
    per_phase_step_cap: inherit_stage2_episode_length
    episode_step_cap: sum_phase_caps
    caps_are_external_truncations: true
    failure_precedes_success: true
    reset_between_phases: false
    teleport_between_phases: false
    preserve_amp_history: true
    seed_new_relation_history_from_current_state: true
    preserve_physics_history: true

  rsi:
    episode_start: inherit_stage2_joint_consistent
    between_phases: disabled
    phase_end_snapshot_starts: false

  reward:
    current_phase: inherit_stage2
    recompute_sharing_from_current_graph: true
    phase_completion_bonus: 0.0
    program_completion_bonus: 0.0
    additional_standup_reward: false

  rollout:
    store_current_and_next_graph: true
    bootstrap_across_phase_boundaries: true
    preserve_trace_across_phase_boundaries: true
    use_true_final_obs_for_truncation: true

  evaluation:
    stage2_zero_shot_first: true
    use_same_program_runner_for_all_models: true
    include_single_phase_retention: true
    report_initial_and_entry_success: true
    save_phase_transition_video: true
```

`program_manifest: null` 등의 필수 인자는 실행 시 검증해서 빠진 경우 명확하게 오류를 낸다. 임의의 checkpoint/program을 자동 선택하지 않는다. 실제 생성한 config와 실행 가능한 command를 Codex 완료 보고에 포함한다.

CLI 인터페이스 제안:

```bash
# 아래 script는 이번 작업에서 생성할 후보다. 현재 존재/실행 완료를 뜻하지 않는다.
STAGE2_CHECKPOINT="/actual/path/to/stage2.pth" \
PROGRAM_CONFIG="/actual/path/to/programs.yaml" \
bash tokenhsi/scripts/multi_agent/approach_stage2_long_horizon_eval.sh

STAGE2_CHECKPOINT="/actual/path/to/stage2.pth" \
PROGRAM_CONFIG="/actual/path/to/train_programs.yaml" \
bash tokenhsi/scripts/multi_agent/approach_stage3_train.sh

STAGE3_CHECKPOINT="/actual/path/to/stage3.pth" \
PROGRAM_CONFIG="/actual/path/to/test_programs.yaml" \
bash tokenhsi/scripts/multi_agent/approach_stage3_test.sh
```

Stage 3 resume와 새 Stage 2 import는 별도 인자로 구분한다. Source checkpoint 파일은 덮어쓰지 않는다.

---

## 15. 검증 항목 — 완료 조건

### A. 기존 기능과 checkpoint

- [ ] Stage 3 flag가 꺼지면 Stage 1/2 실행·평가·checkpoint loading에 영향이 없다.
- [ ] Stage 2 → Stage 3 load 직후 같은 입력에 대한 action mean/sigma가 일치한다.
- [ ] 학습된 `W_c`, `phi`, cross-attention이 그대로 보존된다.
- [ ] Frozen actor parameter와 RMS는 update 후 불변이고, coordination/head/critic에는 gradient가 흐른다.
- [ ] Stage 3 resume 시 optimizer·scheduler·train step이 복원된다.

### B. Phase 전환의 정합성

- [ ] 단위 테스트에서 physics step 없이 phase 전환 함수만 호출하면 human/object pose·velocity·DOF가 완전히 불변이다.
- [ ] AMP history와 logical/physical object mapping이 유지된다.
- [ ] Phase boundary에서 reward는 old graph, next obs는 new graph에 대응한다.
- [ ] 새 graph success를 실제 state에서 다시 계산하고 old success latch를 복사하지 않는다.
- [ ] Task-relative progress는 새 기준으로 초기화하고 physical velocity history는 보존한다.
- [ ] 중간 phase 완료 시 episode reset/termination이 발생하지 않는다.
- [ ] 최종 성공/실패/외부 timeout을 구분한다.
- [ ] Pre-satisfied phase가 한 step 안에 연쇄 auto-skip되지 않는다.

### C. Vectorization / PPO

- [ ] Batch 내 env들이 서로 다른 phase에 있어도 각자의 packet을 사용한다.
- [ ] Live graph 변경이 이미 저장한 rollout sample의 내용/결과를 바꾸지 않는다.
- [ ] Phase 경계에서 bootstrap/trace mask는 1이다.
- [ ] Terminal에서는 bootstrap=0, external truncation에서는 실제 final observation을 사용한다.
- [ ] Truncation 후 새 episode advantage가 이전 episode로 이어지지 않는다.
- [ ] Rollout 마지막 step에서 phase가 전환되는 사례를 테스트한다.
- [ ] Invalid/padded edge가 늘어나도 action에 영향을 주거나 NaN을 만들지 않는다.
- [ ] 동일 checkpoint를 2/3/4-agent 및 서로 다른 edge capacity로 평가할 수 있다.

### D. RSI / physical smoke

- [ ] Full program episode에서 RSI 호출 횟수는 시작 시 한 번이다.
- [ ] 공유 물체 초기화는 Stage 2의 joint-consistent 규칙을 유지한다.
- [ ] 실제 SIT 종료 상태를 그대로 받아 다음 CARRY를 시작하는 장면을 저장한다.
- [ ] 실제 CLIMB 종료 상태에서 다음 task로 가는 시도도 접근 가능한 scene에서 확인한다.
- [ ] 실패 시 reset/teleport로 보정하지 않고 실패 원인을 로그한다.

### E. 최소 학습/평가

- [ ] Stage 2 zero-shot multi-phase baseline을 저장한다.
- [ ] Stage 3 짧은 smoke 학습에서 loss/gradient/phase reach 수가 유한하고 증가 가능한지 확인한다.
- [ ] Stage 3 학습 전후 single-phase retention을 같은 seed로 비교한다.
- [ ] 2-phase training과 더 긴 sequence 평가 split이 실제로 분리되어 있다.
- [ ] 실행하지 않은 simulator test나 학습 성능을 완료로 보고하지 않는다.

---

## 16. Codex 작업 순서와 완료 보고

1. 실제 Stage 2 코드·checkpoint contract 확인.
2. Phase spec/runtime과 **reset 없는 multi-phase evaluator**부터 구현.
3. State preservation, graph switching, PPO mask unit test.
4. Stage 2 zero-shot baseline 실행 경로 확보.
5. Stage 2 → Stage 3 weight import, optimizer, on-policy fine-tuning 연결.
6. 2-agent / 2-phase small smoke test와 single-phase retention 검사.
7. Variable-N/E 및 긴 program evaluator 연결.
8. 실제 config·script·테스트 결과·남은 제한을 보고.

완료 보고에 포함:

```text
변경 파일 목록
확인한 Stage 2 checkpoint / 실제 config / commit
실제 frozen/trainable 범위와 LR
stage3_transfer_report.json
program spec 및 train/validation/test manifest
phase transition / RSI / AMP history / PPO mask 검증 결과
zero-shot과 fine-tuned 결과의 구분
실행한 명령과 테스트 / 실패·미실행 항목
전환 실패 유형과 남아 있는 물리적 제약
```

**금지:** Stage 3를 구현했다고 Stage 2/Stage 1 코드를 새로 설계하거나, action head를 다시 zero-padding 초기화하거나, agent별로 edge를 하나씩 켜주는 scheduling으로 바꾸는 것.

---

## 17. 한 장 요약

```text
Stage 2 checkpoint
  ├─ Frozen actor encoder 그대로
  ├─ 학습된 human-to-edge cross-attention 그대로
  └─ 학습된 128D-input action head 그대로
                  ↓
          Stage 3 fine-tuning
                  ↓
Episode 시작: Program 선택 + Phase 1 joint RSI
                  ↓
Phase 1: 전체 cooperative edge set 동시 입력
                  ↓ 공동 완료
          Physical state 그대로
                  ↓
Phase 2: 새 cooperative edge set 동시 입력
                  ↓ 공동 완료
          Physical state 그대로
                  ↓
                ...
                  ↓
최종 program 완료 / 실제 실패 / 외부 cap에서 episode 종료
```

**Stage 3는 “새로운 dependency 구조를 하나 더 붙이는 단계”가 아니라, 이미 학습한 cooperative controller가 이전 phase의 실제 결과에서 다음 phase를 수행하도록 적응시키는 단계다.**

**핵심 평가:** Stage 2 zero-shot과 Stage 3 fine-tuned를 같은 runner로 비교하고, 이후 **더 많은 agent + 더 많은 phase**로 확장한다.

---

## 근거 문서와 외부 기술 참고

**[S2] `CODEX_STAGE2_COORDINATION_HANDOFF.md` (2026-09-25).**

- §1: 기준 코드와 로컬 구현 우선 원칙.
- §2–5: phase semantics, model, full-head fine-tuning, freeze/checkpoint 규칙.
- §7–9: joint RSI, reward, PPO graph packet 정합성.
- §10–11: variable-N/E와 선택적 long-horizon 확장.
- 이 Stage 3 문서는 해당 결정을 계승하며, 새 runtime·curriculum·config 기본값을 제안한다.

**[T1] Farama Foundation, Gymnasium — Handling Time Limits.**

```text
https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
```

Termination/truncation과 value bootstrap의 구분을 참고했다. 현재 프로젝트를 Gymnasium으로 이식하거나 최신 버전으로 업데이트하라는 지시가 아니다.

**[T2] PyTorch — Autograd mechanics.**

```text
https://docs.pytorch.org/docs/2.14/notes/autograd.html
```

Frozen parameter와 gradient/no-grad 경계를 구분하는 참고 자료다. 실제 프로젝트의 PyTorch/IsaacGym 호환 버전은 유지한다.
