# Codex 수정 지시서: Scalar PRE/TERM Context + Edge-local Success Saturation

## 0. 작업 대상과 목적

저장소 `KSH3880/CVPR2027`, 브랜치 `approach_clean`의 `approach_distance_success`를 기준으로 아래 설계를 실제 코드에 구현해줘. 이 문서는 구현 요청이며, 분석이나 제안만 하고 끝내지 말고 관련 코드·config·테스트·문서를 함께 수정해줘.

확인한 기준 HEAD는 `34a8cb2e4f2482ace7183f71bd40bf32b74be56b`다. 로컬 HEAD가 이후 버전이면 현재 diff와 실제 코드를 확인하고 변경을 맞춰 적용한다. 사용자의 기존 수정은 덮어쓰거나 되돌리지 않는다.

핵심 변경은 다음과 같다.

1. HOLDING 상태함수의 `hand_distance_scale`을 새 실험에서 5.0 → 10.0으로 변경한다.
2. prerequisite가 state/progress/reward에 곱해지는 경로를 새 실험에서 제거한다.
3. 정책에는 edge마다 `ctx = [q_pre, q_term]`만 execution context로 제공한다. Semantic과 context를 따로 인코딩한 뒤 shared MLP로 결합한다.
4. 성공 보상을 agent 단위가 아니라 edge 단위로 통일한다. 각 edge는 state/progress/success 가중치가 모두 0.2다.
5. 자기 edge가 현재 성공했거나, 그 edge의 TERM으로 지정된 다른 edge가 현재 성공했으면 해당 edge의 세 reward component를 모두 1로 포화한다. 가중합은 0.6이다.
6. graph/reward/observation/network 경로를 `E = 2 * num_agents`라는 가정에서 분리한다. 현재 학습 task는 2-edge/agent carry를 유지하되, 추론에서 edge 수를 바꾸어도 같은 weight를 사용할 수 있게 한다.

**이번에는 PRE/TERM dependency matrix를 policy attention에 넣지 않는다.** 내부에서 참조를 계산하기 위한 graph metadata/mask는 필요하지만, 이것을 신경망 입력으로 사용하는 2번 방식과 혼동하지 않는다.

## 1. 먼저 확인할 파일과 현재 구현

작업 시작 시 `git status --short`를 확인하고 `AGENTS.md`, `changelog.md`, `markdowns/structure.md`, `markdowns/config.md`의 관련 부분을 읽는다. 실제 코드와 YAML을 기준으로 수정한다.

| 파일 | 확인한 현재 구현 / 수정 초점 |
|---|---|
| `tokenhsi/data/cfg/multi_agent/approach_distance_success.yaml` | state/progress 0.2, HOLDING k=5, AT k=10, agent 단위 current success +0.2 |
| `tokenhsi/utils/relation_task_spec.py` | `CarryGraph`, `compile_carry_subgoal`, config 검증, checkpoint metadata. 현재 E=2M, target=2h+1 고정 |
| `tokenhsi/env/tasks/multi_agent/relation_reward.py` | `evaluate_holding`, `evaluate_at`, `relation_step`, `RelationRuntime`. prerequisite multiplication 및 agent 단위 saturation 제거/대체 |
| `tokenhsi/env/tasks/multi_agent/relation_task.py` | `_evaluate_relations`, `_compute_relation_reward`, 진단. 현재 Holding/At를 stack하고 `0::2`, `1::2`로 접근 |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py` | runtime 초기화, observation width, `get_relation_suffix_size`, reward 로그 이름 |
| `tokenhsi/learning/multi_agent/amp_network_builder_ma.py` | `EdgeEncoder`, `RelationEncoder`, `build_dynamic_relation_bias`, `set_entity_counts`. semantic/context 결합 및 E 독립성 |
| `tokenhsi/learning/multi_agent/scene_normalizer.py` | 새 context suffix RMS 우회 |
| `tokenhsi/learning/multi_agent/ma_agent.py`, `ma_players.py` | environment 정보 전달, rollout/replay observation, normalizer 크기, checkpoint 로드/평가 |
| `tokenhsi/tests/test_relation_distance.py` 및 관련 `test_relation_*.py` | 기존 실험 보존과 새 실험 테스트 분리 |

현재 `relation_step`의 핵심은 다음과 같다.

```python
activation = prerequisite_minimum(relation_gate(phi_prev), graph.prereq_mask)
state = state_weight * activation * phi_next * reward_mask
progress_reward = progress_weight * activation * pinned_progress * reward_mask
```

현재 saturation은 `current_at_success`가 만든 `[N,M]` flag를 `[:, graph.edge_owner]`로 `[N,E]`에 broadcast한다. 현재 성공 보상은 agent당 `current_success_reward=0.2`다. 새 방식은 이 둘을 edge-local 규칙으로 바꾼다.

현재 네트워크는 **entity token Transformer + edge-conditioned attention bias** 구조다. 별도의 edge-token Transformer가 아니다. 이 구조는 유지한다.

현재 state-relation 경로의 dynamic 입력은 edge당 `[phi, gate, satisfied, achieved, owner_done]` 5개이며, static semantic bias와 dynamic bias를 따로 계산하여 더한다. 새 실험에서는 이것을 아래의 `sem + [pre,term] -> fused edge embedding -> bias`로 교체한다.

## 2. 실험 분리와 변경 범위

기존 실험과 checkpoint를 보존하기 위해 기존 `approach_distance_success`와 `approach_distance_success_no_sat` YAML은 baseline으로 남긴다. 새 실험명은 `approach_distance_edge_context_success`로 한다.

새 환경 YAML과 train/test/VNC script를 만들고 output도 별도로 분리한다. 공용 모듈은 mode/schema로 분기하여 baseline 동작을 유지한다. 새 모드는 예를 들어 `state_relation_v1`, schema version은 2로 정의한다. 이름보다 중요한 것은 기존 schema와 명시적으로 구분하고 조용히 호환시키지 않는 것이다.

`STATE_MODE` 상수 한 곳만 추가하고 끝내지 말고, 기존 `== STATE_MODE`, 허용 mode 검사, environment/network/player/normalizer 경로를 모두 점검한다. 기존 v0만 허용하는 validator를 우회하지 않는다.

이번 작업에서 task state evaluator는 현재 존재하는 HOLDING/AT만 구현 대상으로 한다. ON_TOP/BESIDE의 새 물리적 성공 조건을 임의로 만들어 추가하지 않는다. 다만 generic graph와 evaluator dispatch가 나중의 relation 추가를 막지 않게 한다. 미지원 relation은 명확한 오류로 처리한다.

## 3. 용어와 tensor shape

- `N`: parallel environment 수 / minibatch에서는 scene batch 수.
- `M`: agent 수.
- `O`: object 수.
- `L`: entity token 수. 현재 carry scene에서는 `L = 2M + O`.
- `E`: task edge 수. **E를 2M로 가정하지 않는다.**

새 reward kernel은 최소한 다음을 받는다.

```text
phi                 [N,E]       실제 현재 relation state
progress_raw        [N,E]       현재 XY distance progress
own_success         [N,E] bool  실제 현재 성공 조건
edge_src/dst        [E]         entity index
edge_relation       [E]         relation type
edge_owner          [E]         reward 수령 agent
edge_valid          [E] 또는 [N,E]
prereq_mask         [E,E] bool  내부 context 계산용
term_index          [E]         없으면 -1, 있으면 참조 edge index
```

Tensor 이름은 기존 구조에 맞춰 조정해도 되지만 의미와 shape는 명확하게 분리한다. `edge_mask`가 기존에는 paid-edge 의미였다면 structural validity와 reward eligibility가 같은 것인지 확인한다. Padding edge가 보상을 받거나 성공 source가 되어서는 안 된다.

## 4. 상태함수: 형태는 유지하고 HOLDING k만 변경

### 4.1 HOLDING

기존 손 중점과 object 중심 거리식을 그대로 사용한다.

```text
hand_midpoint = mean(left_hand_position, right_hand_position)
d_H = ||hand_midpoint - object_position||_2
phi_H = exp(-10.0 * d_H^2)
```

새 실험 config는 반드시 다음 값을 전달한다.

```yaml
holding:
  hand_distance_scale: 10.0
```

reset/evaluation/reward/context 계산이 동일한 k를 사용하도록 한다. 공용 함수의 default를 바꾸어 baseline을 암묵적으로 바꾸기보다는, 새 모드의 기본값과 명시적인 config 전달을 일치시킨다. 기존 v0 실험의 k=5는 보존한다.

손별 contact, 높이, 속도, grasp force 조건은 이번 변경에 임의로 추가하지 않는다. 이 함수는 현재 구현상 hand-midpoint proximity proxy이며 실제 물리 grasp를 보장하는 검출기는 아니라는 점은 진단에서 구분한다.

### 4.2 AT

현재 `box_near` 상태함수를 유지한다.

```text
phi_AT = exp(-10.0 * ||object_position - goal_position||_2^2)
z_error_AT = abs(object_position.z - goal_position.z)
```

**1 mm 조건은 object의 world Z 자체가 아니라 목표 중심과의 절대 Z 오차다.** 기존 `_tar_pos`의 box-center 목표 convention을 그대로 사용한다. 플랫폼 높이나 box half-height를 이 과정에서 중복 보정하지 않는다.

## 5. Progress: approach_distance_success의 distance 식 유지

```text
d_xy = ||source_xy - target_xy||_2
P_e = 1 / (1 + max(d_xy - 0.5, 0) / 1.0)
```

HOLDING progress는 human root → object 중심, AT progress는 object 중심 → goal이다. HOLDING state의 손 중점과 HOLDING progress의 human root를 혼동하지 않는다.

`delta=0.5`, `sigma=1.0`을 유지한다. velocity/direction progress, near blending, 별도 pinning 함수는 재도입하지 않는다. `P_e`에 prerequisite를 곱하지 않는다. State도 동일하게 prerequisite를 곱하지 않는다.

## 6. Edge-local current success

모든 성공 상태는 **현재 step의 물리 상태로 재계산하는 live condition**이다.

```text
S_H(e)  = (phi_e >= 0.9)
S_AT(e) = (phi_e >= 0.9) AND (abs(z_error_e) <= 0.001)
```

AT own success에 HOLDING 성공이나 prerequisite history 조건을 추가하지 않는다. 원래 사용자가 지정한 AT/Z 판정만 적용한다.

`own_success`는 `[N,E]`로 계산한다. `subgoal_target`, `2*a+1`, 마지막 edge라는 가정으로 계산하지 않는다. relation type과 edge의 source/target binding으로 계산한다.

보상용 성공에는 one-shot, latch, achieved-history, initially-successful suppression, dwell-time, hysteresis를 넣지 않는다. Reset 시 이미 성공한 상태라면 첫 reward step부터 그 현재 성공 상태에 해당하는 보상을 지급한다. 과거 성공했다가 지금 실패하면 현재 성공은 false다.

평가용 `ever_achieved` 또는 first-success 시간 기록은 별도로 남겨도 되지만 새 reward나 policy context의 대체값으로 쓰지 않는다.

## 7. TERM 참조와 최종 reward saturation

### 7.1 방향을 하나로 고정

`term(e_i) = e_j`의 뜻은 다음과 같다.

> e_j가 현재 성공 상태이면 e_i의 state/progress/success 보상을 모두 포화한다.

내부 mask를 쓰면 다음 convention을 사용한다.

```text
prereq_mask[i,j] = True  <=>  j는 i의 prerequisite
term_mask[i,j]   = True  <=>  i.term은 j
```

즉 `HOLDING -> AT` carry에서는:

```text
pre(AT)       = [HOLDING]
term(HOLDING) = AT
pre(HOLDING)  = []
term(AT)     = 없음
```

PRE와 TERM은 서로 자동으로 생성되는 역관계가 아니다. 이 carry template에는 둘 다 명시적으로 설정한다. 일반 graph에서는 prerequisite이더라도 term으로 지정하지 않았으면 해당 edge를 자동으로 retire시키지 않는다.

이번 명세는 edge당 TERM이 0개 또는 1개인 경우를 정의한다. 여러 TERM의 AND/OR 의미는 합의하지 않았으므로 임의로 정하지 말고, 복수 TERM을 입력하면 명확히 검증 오류를 낸다. Prerequisite는 여러 개를 허용한다.

### 7.2 Saturation flag

```text
T_i = S_term(i),  term이 없으면 false
F_i = S_i OR T_i
```

여기서 TERM source는 **참조 edge의 실제 own_success S_j**다. 참조 edge의 `F_j`, 포화된 success reward, `done` latch를 참조하지 않는다. 이번 구현에 재귀적인 TERM 전파를 추가하지 않는다.

`F_i`는 물리적 성공이 아니라 **보상 포화 flag**다. `S_i`와 F_i는 별도로 보관한다. TERM으로 포화됐다고 실제 phi, own_success, achieved-history를 성공으로 덮어쓰지 않는다.

### 7.3 모든 edge에서 동일한 세 항

```python
state_paid    = torch.where(F, torch.ones_like(phi), phi)
progress_paid = torch.where(F, torch.ones_like(progress_raw), progress_raw)
success_paid  = F.to(phi.dtype)

edge_reward = (
    0.2 * state_paid
    + 0.2 * progress_paid
    + 0.2 * success_paid
)
edge_reward = torch.where(edge_valid, edge_reward, torch.zeros_like(edge_reward))
```

따라서:

```text
F_i == False:  R_i = 0.2 * phi_i + 0.2 * P_i
F_i == True:   R_i = 0.2 + 0.2 + 0.2 = 0.6
```

**Own success일 때도, TERM success일 때도 success component까지 1로 포화한다.** TERM 성공이라고 state/progress만 포화하고 success를 0으로 두지 않는다. Own과 TERM이 동시에 성공해도 0.6이지 0.8이나 1.2가 아니다.

보상은 매 step 지급한다. One-shot bonus가 아니다.

### 7.4 Agent reward aggregation

```text
R_agent(a) = sum(R_i for edges with owner(i) == a)
```

구현은 `edge_owner`에 대한 scatter-add/reduction을 사용한다. Edge 개수로 평균내거나 0.6으로 재정규화하지 않는다. 이번 설계는 edge별 공통 scale을 사용하는 합산 reward다.

```text
edge 1개 포화:              0.6
agent의 edge 2개 모두 포화:  1.2
agent의 edge 5개 모두 포화:  3.0
2-agent / 4-edge scene 합:   2.4  (scene 전체를 합산해 표시할 때)
```

기존 agent 단위 `current_success_reward=0.2`를 위 합에 다시 더하지 않는다. `subgoal_success_bonus`도 새 경로에서는 지급하지 않는다. 새 `success_reward_weight=0.2` 하나로 통일한다.

### 7.5 포화 범위

포화 대상은 task edge의 **state/progress/success** 세 항뿐이다. Power penalty, collision penalty, box-speed penalty, AMP reward, fall/timeout termination은 포화하거나 제거하지 않는다.

PPO의 `task_reward_w`, `disc_reward_w`, reward shaper scale도 임의로 재조정하지 않는다. 기존 대비 agent task reward의 상한이 1.0에서 1.2로 바뀌는 것은 명시적으로 기록한다.

## 8. prerequisite reward multiplication 제거

새 모드에서는 다음과 같은 식이 어떤 형태로도 남으면 안 된다.

```python
activation * state
activation * progress
pre * edge_reward
(1 - term) * edge_reward
(1 - term) * raw_reward + term * maximum
```

`relation_gate`로 sigmoid를 계산한 뒤 곱하는 경로도 새 reward에서는 사용하지 않는다. 단순히 activation을 1로 덮어서 오래된 gating path를 위장하기보다는 새 reward kernel에서 prerequisite와 reward를 분리한다.

Context 계산에 prerequisite 정보는 유지한다. 제거 대상은 **보상에 곱하는 execution gate**다. 유효 edge/padding mask, entity/assignment mask, 관계의 실제 상태 정의, 안전 penalty까지 삭제하라는 뜻이 아니다.

`q_pre`는 context이므로 0.1이어도, 아직 포화되지 않은 edge의 raw state reward는 `0.2*phi`, raw progress reward는 `0.2*P`다.

## 9. Policy context: 1번 방식만 구현

### 9.1 Scalar context 정의

이전 논의대로 두 숫자만 사용한다.

```text
q_pre(i,t) = min(phi_j(t) for j in Pre(i))
             prerequisite가 없으면 1.0

q_term(i,t) = phi_term(i)(t)
             TERM이 없으면 0.0

ctx_i(t) = [q_pre(i,t), q_term(i,t)]
```

`q_pre`에는 sigmoid gate가 아닌 raw state phi의 min을 사용한다. 곱으로 aggregate하지 않는다. `q_term`도 이번 기본형에서는 참조 edge의 raw phi다. 추가적인 beta, level, edge index embedding, satisfied/achieved/owner_done channel은 context MLP에 넣지 않는다.

**주의:** `q_term`은 연속적인 정책 입력이고, reward를 포화하는 `T_i`는 TERM edge의 엄격한 own_success다. 같은 변수가 아니다. 예를 들어 AT phi=0.95라도 Z 오차가 2 mm라면 HOLDING의 q_term은 0.95지만 AT own_success와 HOLDING의 term_success는 false다. `q_term >= 0.9`만으로 reward를 포화하지 않는다.

Policy context는 언제나 실제 phi로 계산한다. Paid state=1이나 effective success reward를 phi로 되먹이지 않는다.

### 9.2 기본 two-agent carry 예

| Edge | Prerequisite | TERM | q_pre | q_term |
|---|---|---|---|---|
| H_A HOLDING O_A | 없음 | O_A AT G_A | 1 | phi_AT_A |
| O_A AT G_A | H_A HOLDING O_A | 없음 | phi_H_A | 0 |
| H_B HOLDING O_B | 없음 | O_B AT G_B | 1 | phi_AT_B |
| O_B AT G_B | H_B HOLDING O_B | 없음 | phi_H_B | 0 |

A/B 사이에 지정하지 않은 dependency를 생성하지 않는다.

### 9.3 Semantic과 context를 따로 만들고 결합

기존 A2 semantic 정의는 유지한다.

```text
sem_i = SemanticEncoder(source_type_i, relation_type_i, target_type_i)
        -> 64-D
ctx_i = ContextEncoder([q_pre_i, q_term_i])
        -> 64-D
z_i = FusionMLP(concat(sem_i, ctx_i))
        -> 64-D
```

구체적인 기본 구성:

```text
SemanticEncoder: 현재 source16 + relation32 + target16, 기존 shared MLP 유지
ContextEncoder:  Linear(2,32) -> ReLU -> Linear(32,64)
FusionMLP:       Linear(128,64) -> ReLU -> Linear(64,64)
```

이 encoder/mixer의 weight는 모든 task edge에서 공유한다. Actor와 critic은 기존처럼 서로 별도의 encoder weight를 가진다. 같은 source/relation/target type이라도 context가 다르면 fused representation이 달라질 수 있어야 한다.

`z = pre*(1-term)*sem` 같은 고정 multiplicative policy gate는 추가하지 않는다. Context에 따른 활용 방식은 fusion과 이후 policy가 학습한다. MLP가 반드시 단조 on/off를 보장한다고 가정하지 않는다.

### 9.4 Entity attention bias에 연결

현재 구조를 유지하여 fused z를 layer/head scalar bias로 투영한다.

```text
sem                [E,64]
ctx                [N,E,2]
z_ctx              [N,E,64]
z_fused            [N,E,64]
edge_bias          [layers,N,heads,E]
entity_pair_bias   [layers,N,heads,L,L]
```

각 edge i의 bias를 `(edge_src[i], edge_dst[i])`에 배치한다. Attention은 기존처럼 entity QK logits에 이 bias를 더한다. GTA와 entity tokenizer, human readout은 유지한다.

현재 `build_dynamic_relation_bias`의 `[phi,gate,satisfied,achieved,done]` 경로를 새 mode에서 실행하지 않는다. 기존 task semantic bias를 별도 더해서 fused semantic을 중복 반영하지 않는다. SELF/NONE 등 배경 static bias를 유지한다면 task edge 위치의 중복 가산을 명시적으로 제거한다.

같은 entity pair에 여러 task edge가 있을 때 `index_copy`로 마지막 edge가 앞선 edge를 덮어쓰지 않게 한다. 새 경로의 기본 aggregation은 `scatter_add`로 명시한다. 향후 정규화 실험은 별도 변경으로 남기고 이번에 reward scale과 함께 임의로 바꾸지 않는다. 동일 pair의 반복 relation은 정보 압축 한계가 있을 수 있으므로 네트워크가 모든 task sequence를 구별한다고 주장하지 않는다.

E×E PRE/TERM matrix attention, edge-token Transformer, GNN, relation-specific PRE/TERM attention head는 이번 범위에 추가하지 않는다.

## 10. Observation / PPO / normalization

새 실험의 policy suffix는 edge별 두 context 값이다.

```text
[기존 clean-scene node/pose observation | flatten(edge_context[N,E,2])]
suffix_width = 2 * E
```

원래 `4E + M = 9M` suffix와 구분한다. `[phi,gate,satisfied,achieved] + subgoal_done`은 새 policy 입력에서 제거하고 필요한 기록은 diagnostics-only로 남긴다.

`get_relation_suffix_size`, network parser, env info, rollout buffer, actor/critic input width, player, normalizer의 passthrough 크기를 함께 갱신한다. 공통 schema helper로 width를 계산하여 파일마다 다른 공식을 두지 않는다.

현재 상태 phi(t)에서 q_pre(t), q_term(t)를 계산하여 obs_t에 저장한다. `a_t` 뒤의 물리 상태로 reward를 계산하고, 동일한 다음 상태에서 obs_(t+1)의 context를 계산한다. Old implementation의 phi_prev gate를 실수로 next-state observation에 섞지 않는다.

**PPO minibatch에서는 저장된 rollout context를 읽어야 한다.** 학습 forward 중 현재 simulator runtime의 최신 phi를 조회하여 과거 obs의 context를 바꾸면 안 된다. Actor/critic 모두 동일한 시점의 context를 사용한다.

Context 두 값과 GTA pose는 RMS/일반 observation clipping을 우회한다. Human/Object의 기존 타입별 RMS는 유지한다. Raw phi, q_pre, q_term을 reward saturation 때문에 1로 덮어쓰지 않는다.

이번 구현은 한 rollout에서 공유되는 고정 task graph로 시작해도 된다. Graph metadata는 env와 policy에 동일하게 전달한다. 추후 environment별 graph randomization을 지원한다면 해당 graph binding/mask도 rollout에 함께 저장해야 한다. 현재처럼 global graph 하나만 쓰면서 sample별 graph가 다른 척하지 않는다.

## 11. Graph를 E=2M에서 분리

현재 `compile_carry_subgoal`은 default task를 생성하는 편의 template로 남겨도 된다. 그러나 그 결과를 쓰는 generic reward/context/network 경로는 임의 E를 받아야 한다.

최소 구성은 explicit edge list를 compile하는 공통 함수와, 그 함수를 호출하는 default independent-carry template이다. 예:

```yaml
edges:
  - id: hold_a
    owner: 0
    src: H_0
    relation: HOLDING
    dst: O_0
    pre: []
    term: at_a
    required_goal: false

  - id: at_a
    owner: 0
    src: O_0
    relation: AT
    dst: G_0
    pre: [hold_a]
    term: null
    required_goal: true
```

이것은 새 graph input schema 예시이며 현재 저장소에 이미 있는 기능이라고 가정하지 않는다. Task graph instance는 reward 정의와 분리된 `env.relationGraph` 같은 위치에 두어, 평가에서 graph 크기를 바꾸는 것과 reward 식을 바꾸는 것을 구별한다.

Edge ID는 참조 해석용이다. Learned edge-ID embedding으로 사용하지 않는다. Compile 시 이름을 tensor index로 해석하고 permutation 시 pre/term/src/dst/owner를 모두 함께 remap한다.

새 경로에서는 다음 고정 가정을 제거한다.

```text
E = 2 * num_agents
suffix_width = 9 * num_agents
phi[:, 0::2]  -> 무조건 HOLDING
phi[:, 1::2]  -> 무조건 AT
target = 2 * agent + 1
reshape(N, M, 2)
owner.repeat_interleave(2)  -> generic 경로에서 사용
```

현재 `_evaluate_relations`처럼 assigned object/goal pair만 계산하여 stack하지 말고, edge type별 evaluator에 `edge_src`, `edge_dst`로 gather한 실제 binding을 전달한다. Physical object reset assignment와 logical object slot mapping도 유지해야 한다.

`edge_owner`는 reward 수령자이며 source entity type에서 임의 추론하지 않는다. Human→Object가 아닌 Object→Goal edge도 올바른 owner에게 지급한다. Goal success 집계는 explicit required_goal mask/set으로 수행하고 '마지막 edge'를 찾지 않는다.

초기화 시 observation buffer width를 계산하기 전에 E와 graph layout을 알 수 있어야 한다. `_init_relation_runtime`이 기존처럼 base environment 초기화 뒤에 실행된다는 이유로 observation width가 다시 2M로 돌아가지 않게 초기화 순서를 정리한다.

새 checkpoint로 다른 E를 사용할 때 graph와 relation-index buffer는 재생성 가능한 non-persistent metadata로 둔다. Parameter shape가 E/L/M/O에 묶이면 안 된다. Flatten된 전체 edge context를 하나의 `Linear(2E, ...)`에 넣지 않는다.

Graph compiler는 잘못된 source/target type, 범위 밖 index, 미존재 pre/term 참조, 자기 자신을 term으로 참조하는 오류, 비어 있는 유효 graph 등을 검사한다. Padding이나 무효 edge를 참조해서 포화 보상을 만들지 않는다.

## 12. Config, scripts, checkpoint 계약

새 YAML의 핵심 값은 다음과 같다. 아직 없는 key는 validator/runtime와 함께 구현한다.

```yaml
relationReward:
  mode: state_relation_v1
  schema_version: 2
  state_reward_weight: 0.2
  progress_reward_weight: 0.2
  success_reward_weight: 0.2
  satisfaction_threshold: 0.9
  holding:
    hand_distance_scale: 10.0
  at:
    state_definition: box_near
    near_distance_scale: 10.0
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
  diagnostics:
    enabled: true
    sample_envs: 2
    log_interval: 30
```

상위 env 설정과 기본 학습 scene, reset 비율, physics 설정은 `approach_distance_success`에서 복사한다. 별도 승인 없이 RSI curriculum, AMP 비중, PPO hyperparameter를 변경하지 않는다. 학습 env는 2048을 유지한다.

제거/비활성화할 기존 새-mode key는 `subgoal_success_bonus`, agent용 `current_success_reward`, `soft_gate`, history bonus 관련 once/reset suppression 옵션, 기존 agent-wide saturation flag다. Legacy mode에는 필요한 기존 key와 기능을 남긴다.

새 experiment scripts:

```text
tokenhsi/scripts/multi_agent/approach_distance_edge_context_success_train.sh
tokenhsi/scripts/multi_agent/approach_distance_edge_context_success_test.sh
tokenhsi/scripts/multi_agent/approach_distance_edge_context_success_vnc.sh
output/approach_distance_edge_context_success/
```

Checkpoint에는 새 relation schema, observation fields/dim-per-edge, context fusion 방식, reward config를 기록한다. 기존 checkpoint를 새 구조로 일반 resume/evaluate하지 않는다. `strict=False`로 shape/key mismatch를 숨기지 않는다. 이번 기본 검증은 새 학습이다.

다만 새 schema로 학습한 checkpoint를 추론에서 다른 E/M/O/task graph로 평가하는 것은 허용해야 한다. Semantic relation vocabulary, per-edge feature width, reward definition, model architecture compatibility와 **task instance graph 크기**를 분리하여 검사한다. Graph 크기까지 reward config 동일성 비교에 묶어서 edge 확장을 차단하지 않는다. Training resume에서 task 자체를 바꾸는 일은 평가 override와 별도로 취급한다.

## 13. Logging과 evaluation

최소한 다음을 실제 값으로 구분하여 기록한다.

```text
edge/phi_raw
edge/progress_raw
edge/q_pre
edge/q_term
edge/own_success
edge/term_success
edge/reward_saturated
edge/state_component       # weighted: 0~0.2
edge/progress_component    # weighted: 0~0.2
edge/success_component     # weighted: 0 또는 0.2
edge/total                # weighted: 0~0.6
agent/task_total
penalty/power, collision, box_speed
```

Own success rate를 `F_i`로 보고하지 않는다. HOLDING이 실제로는 깨졌지만 AT TERM 때문에 reward만 포화된 상태를 구별할 수 있어야 한다.

Goal/task success는 explicit required_goal edge들의 실제 현재 own_success로 집계한다. HOLDING을 release해야 하는 carry에서 모든 HOLDING까지 실제 성공이어야 episode 성공이라고 요구하지 않는다. Ever-achieved success와 current-maintained success는 지표 이름을 구분한다.

평가 종료 규칙은 기존 fall/timeout 유지이며 성공 순간 즉시 종료로 바꾸지 않는다. 현재 episodeLength=600 등 기존 값도 그대로 둔다.

기존 `REWARD_TERM_NAMES`, `RELATION_TB_GROUPS`, CSV/timeline/plot 소비 코드와 shape도 함께 갱신한다. `0::2`를 쓰는 통계 때문에 새 graph가 깨지지 않게 relation/owner/group mask 기반으로 집계한다.

## 14. 필수 CPU 테스트와 기대값

새 mode 전용 테스트를 추가하고 baseline 테스트는 유지한다. 아래는 실행 결과가 아니라 구현 후 확인해야 할 acceptance criteria다.

### 14.1 Reward 수치

| 조건 | 기대 결과 |
|---|---|
| phi=0.8, P=0.6, own=false, term=false | edge R=0.28 |
| own=true, P=0 | state/progress/success 각 0.2, R=0.6 |
| own=false, TERM edge own=true | success까지 포함하여 R=0.6 |
| own=true, TERM edge own=true | 여전히 R=0.6, 중복 가산 없음 |
| TERM 없는 edge | 다른 edge의 성공으로 포화하지 않음 |
| valid=false | 세 paid component와 edge reward 모두 0 |
| 같은 agent의 edge 2개 포화 | agent task reward=1.2, 별도 +0.2 없음 |
| 같은 agent의 edge 5개 포화 | agent task reward=3.0 |

Prerequisite phi를 0↔1로 바꾸되 해당 edge의 phi/P/S/TERM success를 고정하면 reward는 같아야 한다. q_pre만 바뀌어야 한다.

### 14.2 성공 경계와 유지/해제

HOLDING phi=0.9는 성공, 0.8999는 실패다. AT phi=0.9와 |Z|=0.001은 성공, |Z|=0.00101 또는 phi=0.8999는 실패다. Signed Z input을 다루는 helper라면 ±Z에 대칭이어야 한다.

HOLDING 성공 후 놓쳐 phi<0.9가 되면 AT 성공 전에는 포화가 풀린다. AT가 성공 중이면 HOLDING phi=0이어도 HOLDING/AT 각각 0.6이다. AT 성공도 사라지고 HOLDING도 실패면 다음 step에서 즉시 raw reward로 돌아간다. 과거 done/achieved가 true여도 포화가 유지되면 안 된다.

성공 상태가 5 step 유지되면 매 step 지급한다. Reset 성공도 현재 규칙으로 지급하며 first-success suppression을 적용하지 않는다.

TERM edge의 F는 true지만 own_success는 false인 fixture를 만들어, 그 edge를 TERM으로 참조하는 앞 edge가 잘못 재귀 포화되지 않는지 확인한다.

### 14.3 Context와 물리 state 분리

```text
Pre(i) 없음        -> q_pre=1
Pre(i)=[j,k], phi=.8,.3 -> q_pre=.3
TERM 없음          -> q_term=0
TERM=j             -> q_term=phi_j
```

AT phi=.95, Z error=.002인 경우 q_term=.95와 term_success=false가 동시에 성립해야 한다. Reward를 포화한 뒤에도 phi/q_pre/q_term이 보상값으로 바뀌지 않아야 한다.

HOLDING k=10의 analytic 값을 테스트한다. State evaluator와 reset, step, observation의 k가 동일해야 한다. Distance progress와 좌표 변환 테스트도 baseline 식과 같아야 한다.

### 14.4 E/순서/owner 독립성

E=2,4,5,8,10의 CPU graph/reward/context fixture를 사용한다. M을 고정한 채 E만 바꾸는 케이스를 반드시 포함하여 `E=2M` 잔존 가정을 검출한다. 이 fixture가 모두 물리적으로 실행 가능한 task라는 의미는 아니다.

Edge를 shuffle하고 pre/term/index/binding을 일관되게 remap하면 edge별 출력도 같은 permutation으로 바뀌고, owner별 reward와 동일 entity layout의 policy 출력은 수치 tolerance 안에서 같아야 한다.

A의 AT 성공이 지정되지 않은 B edge를 포화시키지 않아야 한다. 명시적으로 설정된 cross-agent TERM은 해당 edge에만 작동해야 한다. Owner가 0,1,0,1,0처럼 interleave되어도 scatter 합이 맞아야 한다.

같은 entity pair의 여러 edge bias가 overwrite되지 않고 합산되는지 테스트한다. Invalid/padding edge는 bias나 reward에 기여하지 않아야 한다.

### 14.5 Network / PPO / checkpoint

동일한 새 checkpoint의 learned parameter를 바꾸지 않고 서로 다른 E로 forward한다. `Linear` input size가 E에 묶이지 않아야 한다. Graph buffer는 strict state_dict weight load를 깨지 않아야 한다.

Context-only change가 fused embedding에 반영되는지, actor/critic output이 finite인지 확인한다. Context 및 fusion MLP gradient도 확인한다. 현재 bias projection은 zero-init이므로 첫 backward에서 upstream gradient가 0일 수 있다. Projection이 한 번 업데이트된 뒤 다음 step에서 sem/ctx/fusion 경로로 유한한 gradient가 흐르는지 검사한다. 테스트를 맞추려고 initialization을 임의 변경하지 않는다.

Normalizer는 edge context와 GTA pose를 그대로 보존해야 한다. 저장된 과거 observation으로 forward할 때 simulator의 현재 phi를 바꿔도 출력이 바뀌면 안 된다.

새 checkpoint save/reload, 새 schema의 다른 E 평가, 기존 schema의 잘못된 resume 거부를 각각 테스트한다.

Power/collision/box-speed penalty가 성공 상태에서도 사라지지 않는지 테스트한다.

## 15. 시뮬레이션 검증과 작업 마무리

먼저 simulator import가 필요 없는 테스트를 실행한다. Reward/runtime integration까지 통과한 뒤 실제 실행 환경이 준비되어 있으면 새 config로 짧은 rollout/PPO smoke를 수행한다. Isaac Gym import 순서와 `runtime_env.sh`를 따른다.

GPU는 `nvidia-smi`로 확인하고 현재 프로세스를 임의 종료·재시작하지 않는다. Training env 수는 2048을 유지하고 iteration만 줄인다. 별도 `OUTPUT_PATH=output/approach_distance_edge_context_success_check`를 쓴다. GPU 번호나 장기 본학습 시작을 임의로 정하지 않는다.

검증할 단계는 접근 → HOLDING 성공 → 이동 → AT 성공 → 손 release다. 이 구간의 phi/Z/own_success/term_success/F/세 component/agent total/penalty를 기록하여 포화 규칙과 실제 물리 전환을 분리해 확인한다.

작업 완료 시 변경 파일, 적용 수식, config/script 사용법, CPU 테스트 결과, 실제 수행한 smoke 범위, 미실행 검증과 이유, checkpoint 호환 범위를 보고한다. 실행하지 않은 실험의 성공·수렴을 주장하지 않는다.

`changelog.md`, `markdowns/config.md`, 필요하면 `markdowns/structure.md`를 실제 변경과 검증 결과에 맞춰 갱신한다.

## 16. 해석상 주의: 보장하지 않는 것

이번 설계는 prerequisite/term 곱셈에 의한 reward 감쇠를 제거하고 성공 시 상한으로 전환한다. 그렇다고 시간에 따라 보상이 절대 감소하지 않는다는 수학적 보장은 아니다. 성공 판정이 false로 돌아오거나 HOLDING이 먼저 깨지고 AT/Z 성공이 늦게 발생하면 낮은 구간이 있을 수 있다. 위 시뮬레이션 로그로 실제 전환을 확인하고, 임의 interpolation/latch를 몰래 추가하지 않는다.

같은 이유로 q_pre가 observation에 있다는 사실만으로 엄격한 task 순서 준수가 보장되지 않는다. 이번 범위는 context 기반 policy 학습이며, 별도 sequence-validity hard gate를 reward에 다시 넣지 않는다.

TERM이 physical own_success만 참조하는 live 규칙에서는, 더 긴 chain의 중간 relation이 나중에 깨지면 이전 edge 포화도 풀릴 수 있다. 이 경우 recursive completion/latch가 필요하다고 임의 변경하지 말고 별도 설계 이슈로 보고한다. Edge 수를 입력으로 처리하는 능력과 더 긴 물리 task의 성공은 다른 검증 항목이다.

HOLDING phi≥0.9는 현재 손 중점 거리 기준이다. k=10에서 대략 10.3 cm 이내에 해당하며 contact/lift 자체를 보장하지 않는다. AT의 1 mm 조건 역시 목표 중심 Z 오차 조건이지 release/support detector가 아니다. 이번에는 합의한 조건을 그대로 구현하고, 악용이나 실패가 있으면 raw 물리 진단과 영상으로 보고한다.

## 17. 최종 구현 요약

```text
실제 현재 상태
  -> edge phi + distance progress
  -> relation별 own_success
  -> term_index로 참조한 다른 edge의 own_success
  -> F = own_success OR term_success
  -> F면 state/progress/success 모두 1
  -> 각 항 0.2, edge당 최대 0.6
  -> owner별 sum, 기존 penalty는 별도 유지

별도 policy 경로
  -> 실제 phi에서 q_pre=min(prerequisite phi), q_term=term phi
  -> ctx=[q_pre,q_term]
  -> shared semantic encoder + shared context encoder
  -> concat + shared fusion MLP
  -> layer/head별 entity attention bias
  -> 기존 GTA Transformer / agent readout
```

**성공 보상도 TERM 포화에 포함한다. Reward에 prerequisite를 다시 곱하지 않는다. Context에 dependency matrix를 추가하지 않는다. 포화된 reward를 실제 state/success로 되먹이지 않는다. E=2M를 generic 경로에 남기지 않는다.**

---

### 확인한 원본 소스

기준: `KSH3880/CVPR2027@34a8cb2e4f2482ace7183f71bd40bf32b74be56b` (`approach_clean`). 위의 기존 동작 설명은 다음 파일을 읽고 작성했다.

- `AGENTS.md`
- `markdowns/config.md`, `markdowns/structure.md`, `markdowns/ma_clean_scene_gta.md`
- `tokenhsi/data/cfg/multi_agent/approach_distance_success.yaml`
- `tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation.yaml`
- `tokenhsi/utils/relation_task_spec.py`
- `tokenhsi/env/tasks/multi_agent/relation_reward.py`
- `tokenhsi/env/tasks/multi_agent/relation_task.py`
- `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py` (초기화/크기 계산 부분)
- `tokenhsi/learning/multi_agent/amp_network_builder_ma.py` (semantic/dynamic edge encoder와 forward/초기화 부분)
- `tokenhsi/learning/multi_agent/scene_normalizer.py`
- `tokenhsi/tests/test_relation_distance.py` (성공·gate·포화 관련 테스트 부분)

이 문서는 수정 지시서다. 문서 작성 단계에서는 저장소 수정, 기존 프로세스 제어, repository test 실행, 실제 학습을 수행하지 않았다.
