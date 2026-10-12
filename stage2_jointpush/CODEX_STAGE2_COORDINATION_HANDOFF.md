# Codex 구현 지시서 — Stage 2 Relational Coordination

**작성일: 2026-09-25**  
**상태: 설계 및 구현 요구사항. 이 문서를 작성하면서 실제 저장소 코드를 수정하거나 학습을 실행한 것은 아니다.**

> Stage 1은 이미 구현되어 있다. 기존 Stage 1 코드·config·checkpoint 동작을 보존하고, 아래 Stage 2를 opt-in 기능으로 추가한다. 우선 2-agent cooperative scenario에서 경향성을 확인하고, 같은 checkpoint로 더 많은 agent/edge를 처리할 수 있는 구조를 만든다.
>
> **최종 채택안:** Frozen Stage-1 actor encoder → Human-to-Grounded-Edge softmax cross-attention → coordination context `c` → `[h,c]` → Stage-1-initialized, fully trainable action head.

---

## 0. 먼저 읽을 핵심 결정

| 항목 | 이번 구현의 결정 |
|---|---|
| Stage 1 | 이미 구현됨. 다시 만들거나 기존 스킬·보상 정의를 임의로 개편하지 않는다. |
| Actor backbone | 기존 entity tokenizer, type embedding, semantic edge encoder, relation/GTA Transformer를 freeze한다. |
| Coordination 입력 | 현재 **하나의 phase에 속한 모든 유효한 task edge**와 그 endpoint의 post-Transformer token. |
| Grounded edge | `edge64 + src_token64 + tgt_token64 → shared projection → 64D`. |
| Coordination 연산 | Human token = Q, grounded edge set = K/V인 **softmax multi-head cross-attention**. |
| 출력 | 각 human마다 64D `c_i`. Network parameter는 모든 human이 공유한다. |
| Action head | `[h_i(64), c_i(64)] → 1024 → 512 → 32`. 전체 trainable. |
| 초기화 | 첫 weight는 `[W_stage1, 0]`, 첫 bias와 뒤 layer들은 Stage 1에서 복사. |
| 이번에 하지 않을 것 | Frozen action head 앞에 `h+Δh`만 넣는 방식, layer-wise adapter, sigmoid gating, explicit dependency DAG. |
| Stage 2 학습 | 우선 `2 humans / 3 objects / 2 goals`, 한 agent당 **하나의 skill bundle**, 최대 2 edges. |
| Phase 내부 | 모든 agent의 edge를 동시에 제공. agent별로 순차 지급하지 않는다. |
| 공유 물체 | 허용한다. 단, 모순된 목표와 물리적으로 불가능한 조합은 제외한다. |
| RSI | 기존 motion/RSI를 재사용하되, **공유 물체에 대한 joint-consistent initialization**을 적용한다. |
| LR | 처음부터 10배 낮추지 않는다. 첫 toy 기본값은 새 module·action head 모두 `2e-5`. |
| 확장 | 분리된 pair 여러 개뿐 아니라 3명 이상이 공유 물체를 쓰는 구조도 입력 가능하게 만든다. 성공은 실험으로 확인. |
| Long-horizon | 필수 Stage 3를 만들지 않는다. Stage 2 checkpoint로 reset 없는 phase 전환을 먼저 평가하고, 필요할 때만 별도 fine-tuning. |

### 확정 사항과 구현 기본값의 구분

위 architecture, freeze 범위, head 초기화, 단일 phase 학습, explicit dependency 입력 미사용은 대화에서 정한 방향이다.

아래에 제시하는 projection hidden width, mixture 비율, RSI 확률, 새 config key/파일명 등은 **Codex가 바로 구현할 수 있게 둔 첫 toy 기본값**이다. 기존 코드의 실제 schema에 맞춰 배치하되, 변경한 기본값은 구현 보고서에 기록한다. 실험 결과로 검증된 최적값이라고 취급하지 않는다.

---

## 1. 기준 코드와 변경 원칙

### 1.1 확인한 원격 기준

- Repository: `KSH3880/CVPR2027`
- Branch: `approach_clean_scenario_2`
- 확인한 commit: `1c119b41b47bdbe49ad0a0fea7802da3cafca5ef`
- 기존 학습 config: `tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation.yaml`
- 당시 원격 scenario config는 정리되어 Git 이력에서 확인한다. 현재 Stage 1 실행 설정은 `markdowns/config.md`에 있다.

**사용자의 로컬 Stage 1 구현이 이 원격 snapshot보다 최신일 수 있다.** 시작할 때 현재 branch·working tree·실제 Stage 1 실행 config를 확인한다. 이 문서에 맞추려고 checkout/reset하거나 사용자 변경사항을 덮어쓰지 않는다. Stage 1 checkpoint 경로는 사용자가 지정하는 인자로 받으며, 임의의 checkpoint를 고르지 않는다.

### 1.2 확인된 구조

`amp_ma_carry_relation.yaml`과 `amp_network_builder_ma.py` 기준:

```text
Actor:
  H/O/G별 shared tokenizer
  token width = 64
  Relation/GTA Transformer = 4 layers, 2 heads, FFN width 512
  Human output → shared action head
  action head = 64 → 1024 → 512 → 32

Critic:
  actor와 별도의 encoder 및 value head
```

현재 `RelationEncoder.forward()`는 모든 entity를 Transformer로 처리한 뒤, clean_scene에서 `x[:, :num_agents]`만 반환한다. Stage 2에서는 **같은 한 번의 forward에서 전체 post-Transformer token과 human token을 함께 얻을 수 있게** 인터페이스를 확장한다. [R1, R3, R8]

### 1.3 기존 코드에서 특히 주의할 사항

1. `utils/edge_scenario_spec.py`의 기존 sampler/validator에는 `2 humans, 3 objects, edge_capacity=4` 고정 조건이 있다. Network만 가변 길이로 만들어서는 추론 확장이 완료되지 않는다. [R4]
2. `Stage1ContextRuntime.step()`의 기존 sharing은 `local.flip(-1)`를 사용한다. 이는 2-agent 상대를 선택하는 방식이지, N-agent의 실제 관련 teammate를 찾는 방식이 아니다. Stage 2에서 그대로 재사용하지 않는다. [R5]
3. 기존 semantic edge encoder는 **source entity type + relation type + target entity type**을 인코딩한다. 실제 `Ha/Oa`의 고유 identity나 연속적인 위치가 semantic 64D에 자동으로 들어 있는 것이 아니다. 동일한 typed relation의 embedding은 같을 수 있다. 따라서 src/dst index로 정확한 token을 gather하는 것이 필수다. [R2]
4. 기존 RSI는 같은 object를 사용하는 두 agent의 non-loco 초기화 충돌을 재추출하는 경로가 있다. Stage 2에서는 단순 retry만으로 해결하지 말고, 아래 공유 물체 초기화 원칙을 적용한다. [R6]
5. 현재 환경은 goal 수를 human 수에 묶고 `numObjects >= numAgents`를 요구하는 부분이 있다. 첫 확장 평가는 이 제약을 지키면서 진행해도 된다. Network의 variable token 처리와 환경의 임의 H/O/G 개수 지원을 혼동하지 않는다. [R1, R6]

---

## 2. Task와 phase의 의미 — 이 부분을 바꾸지 말 것

### 2.1 Stage 2는 한 phase 안의 cooperation을 학습한다

```text
예시 1:
  A: HOLDING(Ha,Oa), AT(Oa,Ga)
  B: CLIMB(Hb,Oa)

이 phase의 입력:
  { HOLDING(Ha,Oa), AT(Oa,Ga), CLIMB(Hb,Oa) }
```

**세 edge를 처음부터 동시에 제공한다.** A의 edge만 먼저 주고, 완료된 다음 B의 CLIMB을 주는 방식이 아니다. A/B의 접근·유지·대기·비켜주기 등의 실제 제어가 학습 대상이다.

다음 cooperative scenario로 넘어갈 때만 phase 단위로 edge set을 교체한다. 순서 없는 edge set만으로 사용자의 임의의 전체 시간 순서를 추론한다고 주장하지 않는다.

### 2.2 Agent당 하나의 skill bundle

| Sampler의 bundle | Policy에 주는 실제 edge |
|---|---|
| HOLDING | `HOLDING(H,O)` |
| SIT | `SIT(H,O)` |
| CLIMB | `CLIMB(H,O)` |
| CARRY | `HOLDING(H,O)` + `AT(O,G)` |
| STACK | `HOLDING(H,O_src)` + `ON_TOP(O_src,O_base)` |

`CARRY/STACK`은 sampler를 설명하는 이름이다. 새로운 relation type이나 scenario ID token을 추가하라는 뜻이 아니다.

한 agent의 `SIT + CLIMB`, `SIT + CARRY` 등은 첫 Stage 2에서 제외한다. 반면 `HOLDING + AT/ON_TOP`은 하나의 placement skill을 구성하는 두 edge라서 허용한다.

### 2.3 입력하지 않는 정보

- 별도의 `AT → CLIMB` dependency arrow / DAG
- upstream/downstream ID embedding
- phase 내부 agent별 active/pending 스케줄
- scenario ID, agent별 learnable ID, edge slot positional embedding
- attention 정답 label

`src, dst, relation, owner, valid` 등 기존 graph binding은 그대로 필요하다. 물체 공유 정보는 그 graph에서 읽을 수 있다. Reward/RSI sampler가 공유 여부를 이용하는 것과 actor에게 별도의 dependency 정답을 주는 것은 구분한다.

---

## 3. Actor architecture

### 3.1 Tensor contract

표기: `B` = env/minibatch의 scene 수, `N` = human 수, `O` = object 수, `G` = goal 수, `L=N+O+G`, `E` = 해당 실행의 edge capacity.

```text
post_tf_nodes: [B, L, 64]
human_tokens: [B, N, 64]
edge_semantic: [B, E, 64]
edge_src/dst/relation/owner: [B, E]
edge_valid: [B, E]   # bool
coordination_context: [B, N, 64]
action_mean: [B, N, 32]
```

기존 clean_scene의 **scene당 actor encoder 한 번 호출** 구조를 유지한다. Human별로 scene을 복사하거나 Transformer를 N번 호출하지 않는다. PPO loss에 필요한 action/value flatten은 기존 위치에서만 수행한다.

### 3.2 Frozen encoder 출력 재사용

기존 반환 형식을 깨지 않는 선택적 API를 추가한다. 예를 들어:

```python
# 이름은 예시. legacy 호출 동작은 유지할 것.
features = actor_encoder.forward_features(obs)
z = features["all_tokens"]       # [B,L,64]
h = features["human_tokens"]     # [B,N,64]
```

동일한 actor forward에서 모든 token을 얻는다. Critic token을 actor coordination 입력에 섞지 않는다.

중요: freeze는 **파라미터가 고정**이라는 뜻이다. `h,z`는 현재 observation과 edge set이 바뀌면 매 step 달라진다. Episode 시작 feature를 고정해 재사용하면 안 된다.

### 3.3 실제 task edge embedding 추출

기존 frozen actor의 semantic encoder로 현재 packet의 유효한 task edge만 인코딩한다.

```text
e_j = frozen_edge_encoder(
    source_type(entity_types[src_j]),
    relation_j,
    target_type(entity_types[dst_j])
)
```

`build_edge_embeddings()`가 기본 background relation matrix를 읽는 경로라면 그것만 호출해서는 안 된다. **각 PPO sample에 저장된 graph packet의 src/dst/relation을 사용**한다. SELF/NONE 및 L×L background 전체를 coordination task edge로 넣지 않는다.

Binding은 `e_j` 안에 완전히 저장되었다고 가정하지 말고, 별도의 src/dst index를 계속 보존한다.

### 3.4 Grounded edge — Toy Version A

```text
z_src_j = post_tf_nodes[src_j]
z_dst_j = post_tf_nodes[dst_j]

concat(e_j, z_src_j, z_dst_j): 192D
             ↓ shared phi
bar_e_j: 64D
```

수식:

\[
\bar e_j=\phi_\theta([e_j,z_{src(j)},z_{dst(j)}]).
\]

첫 구현 기본값:

```text
phi = Linear(192,128) → ReLU → Linear(128,64)
```

모든 edge에 같은 `phi`를 적용한다. Src/tgt 순서는 구분하지만 edge 목록의 나열 순서는 의미를 가지지 않는다.

Gather는 batch별 실제 graph binding을 따른다. Invalid slot의 src/dst가 -1 등이면 안전한 index로 치환한 후 mask한다. Invalid slot에 NaN/Inf가 섞여 projection까지 전파되지 않게 입력을 유한한 값으로 만든다.

### 3.5 Human-to-edge cross-attention

```text
Q input: h             [B,N,64]
K input: bar_E         [B,E,64]
V input: bar_E         [B,E,64]
```

K/V는 같은 rich feature를 입력받지만 **각각 다른 projection weight**를 사용한다.

\[
Q=HW_Q,\quad K=\bar E W_K,\quad V=\bar E W_V,
\]
\[
C=\operatorname{MHA}(H,\bar E,\bar E).
\]

첫 구현 기본값:

```text
embed_dim=64
num_heads=2
dropout=0.0
batch_first=True
softmax attention
no causal mask
padding mask = ~edge_valid
output context dimension = 64
```

`c_i`는 **human i가 현재 task relation들과 그 endpoint 상태에서 읽은 coordination context**다. `c_i`를 반드시 새 human token이라고 부를 필요는 없다. 이번 구조는 `h_i`와 `c_i`를 분리해 head에서 concat한다.

모든 human query를 한 batch 연산으로 계산한다. 사람별 module, pair별 module, 고정 2명 전용 output layer를 만들지 않는다. Src/tgt를 공유하지 않는 edge도 기본적으로 hard-mask하지 않는다. 첫 toy는 valid mask만 쓰고 관련성은 학습하게 한다.

### 3.6 Empty edge / padding 처리

- 추가 padding slot만 늘려도 action이 변하지 않아야 한다.
- 모든 edge가 invalid인 scene은 softmax가 NaN이 되지 않게 안전 처리한다.
- 이 경우 최종 `c_i=0`을 명시적으로 보장한다. Projection/output bias 때문에 nonzero가 남지 않게 한다.
- `E=0` 자체를 지원하거나, 최소 한 개의 안전한 dummy slot을 넣은 뒤 결과를 zeroing한다.
- Attention dropout은 toy에서 끈다. Debug/identity test도 deterministic하게 수행한다.

### 3.7 이번 구현에서 제외하는 대안

- Semantic-only key인 Version B는 이후 ablation 후보. 기본 모델은 A만 구현해도 된다.
- Sigmoid/independent gating은 이번 기본 구조에 넣지 않는다.
- 별도 edge-to-edge Transformer, explicit dependency matrix, edge-pair action factor, FiLM, layer-wise adapter를 추가하지 않는다.
- `h+Δh → 완전히 frozen action head`로 설계를 되돌리지 않는다.
- Raw world 좌표나 oracle future state를 슬쩍 추가하지 않는다. Post-TF token이 정보 보존에 실패한다는 증거가 있을 때 별도 변경안으로 다룬다.

---

## 4. Action head — Stage 1 초기화 후 전체 fine-tuning

### 4.1 구조

```text
Stage 1: h(64)       → 1024 → 512 → 32
Stage 2: [h(64),c(64)] → 1024 → 512 → 32
```

기존 hidden activation과 output activation은 그대로 유지한다. 별도 fusion MLP는 넣지 않는다. 첫 linear가 곧 `[h,c]`를 섞는 layer다.

### 4.2 초기화

첫 layer는:

\[
W_{first}^{S2}=[W_{first}^{S1}\;\;0],\qquad b_{first}^{S2}=b_{first}^{S1}.
\]

```text
Stage2 first_linear.weight[:, :64] = Stage1 first_linear.weight
Stage2 first_linear.weight[:, 64:] = 0
Stage2 first_linear.bias           = Stage1 first_linear.bias
뒤의 모든 action-head weight/bias   = Stage1에서 복사
```

이후 **첫 layer의 양쪽 weight와 뒤 layer 전부를 trainable**로 둔다. Stage 1 값은 초기화로만 사용하는 것이며, action head는 freeze하지 않는다.

동일한 observation/graph, normalization 통계, inference mode에서 학습 시작 전 action mean은 Stage 1과 같아야 한다. 이는 입력·관계 구성이 달라진 새 시나리오의 성공을 보장한다는 뜻이 아니라, **같은 입력에 대한 초기 함수 보존**이다.

### 4.3 Zero initialization 관련 주의

- 새 grounding/cross-attention까지 모두 zero-init하지 않는다. 일반 초기화를 사용한다.
- Zero-init하는 곳은 새 first-layer의 `c` 입력 weight block이다.
- 최초 backward에서는 `W_c=0`이라 coordination branch에 actor-loss gradient가 0일 수 있다.
- 첫 update로 `W_c`가 바뀐 후 coordination branch로 gradient가 전달된다. 이 첫-step 현상을 오류로 판정하지 않는다.
- 모든 새 layer를 0으로 만들어 `c=0`과 `W_c=0`을 동시에 강제하면 학습이 막힐 수 있다.

### 4.4 코드 크기 기준

현재 `64→1024→512→32` head는 bias 포함 **607,776 parameters**다. `128→1024→512→32`로 바꾸면 **673,312 parameters**, 증가분은 **65,536**이다. 이는 확인한 layer dimension으로 계산한 수치이며 tokenizer/Transformer/critic/coordination module은 포함하지 않는다.

Parameter 수만으로 학습 시간이나 Stage 1 정보의 양을 판단하지 않는다. 실제 속도는 아래 profiling으로 측정한다.

---

## 5. Freeze, optimizer, checkpoint

### 5.1 Freeze/train 범위

| 구성 요소 | Stage 2 처리 |
|---|---|
| Actor entity tokenizers/type embeddings | Stage 1에서 load, freeze |
| Actor semantic edge encoder/context path | Stage 1에서 load, freeze |
| Actor Relation/GTA Transformer | Stage 1에서 load, freeze |
| Actor observation RMS | Stage 1 통계 load, 첫 toy에서 update 정지 |
| Grounded-edge projection `phi` | 새로 초기화, train |
| Human-to-edge cross-attention | 새로 초기화, train |
| 128D-input action head | Stage 1 보존 초기화, **전체 train** |
| Actor sigma | 기존 fixed/learn_sigma 설정 유지 |
| Critic encoder/value head | 독립 경로 유지, Stage 2 return에 맞춰 train |
| AMP discriminator와 AMP 학습 경로 | 기존 학습 방식을 보존. 이번에 임의로 freeze/제거하지 않는다. |

Actor encoder의 고정 연산은 `no_grad()` 또는 detach 경계를 명확히 둬서 backward graph를 만들지 않는다. **Coordination module까지 같은 no_grad 블록에 넣지 않는다.** `model.train()` 호출 후에도 frozen module/actor RMS의 상태가 의도대로 유지되는지 검사한다.

Critic은 actor와 별도 encoder이므로 함께 freeze할 이유가 없다. Stage 1의 호환되는 critic weight로 초기화하고 학습한다. Value RMS는 새 reward 분포에 적응하게 하며, 이전 통계를 복사할지 초기화할지는 실제 restore 로직에 맞춰 명시적으로 기록한다. Actor obs RMS와 value/AMP RMS를 혼동하지 않는다.

### 5.2 LR — 이번에는 과도하게 낮추지 않기

현재 확인된 학습 config LR은 `2e-5`이다. 첫 toy의 목적은 빠른 경향성 확인이다.

```text
coordination modules: 2e-5
entire action head:  2e-5
critic / AMP:        기존 기본값 유지
```

`head_lr_scale=1.0`을 기본값으로 두고, 안정성 문제가 실제로 생기면 `0.5` 등을 비교할 수 있게 한다. 처음부터 copied head를 `2e-6`처럼 10배 낮추지 않는다. 필요하면 새 module만 `5e-5`로 올리는 실험을 별도로 제공한다.

**첫 layer 안의 h/c column block은 하나의 Parameter다.** 일반 optimizer param group만으로 두 block에 별도 LR을 줄 수 있다고 구현하지 않는다. 이번 toy는 전체 head 동일 LR이면 충분하다. 향후 block별 LR이 필요하면 두 Linear branch로 분리하는 등 실제 구현을 바꿔야 한다.

기존 LR schedule/update 함수가 param group의 설정을 덮어쓰지 않는지 확인하고, 적용된 effective LR을 로그에 기록한다.

### 5.3 Stage 1 import와 Stage 2 resume를 분리

**Stage 1 → 새 Stage 2 실행:**

1. Stage 2 model 생성.
2. 호환 Stage 1 backbone/critic/AMP weight를 명시적으로 복사.
3. Action head 첫 layer만 위 규칙으로 확장하고 뒤 layer 복사.
4. Actor input normalization 통계 load.
5. Freeze 적용.
6. Stage 2 optimizer를 새로 구성. Stage 1 optimizer/epoch를 무비판적으로 resume하지 않는다.
7. 모든 복사·shape mismatch·새 parameter·freeze 상태를 `stage2_transfer_report.json`에 기록.

**Stage 2 → 이어 학습/평가:**

- 기존 Stage 2 head와 coordination weight를 그대로 load한다.
- 매 resume마다 `W_c`를 0으로 만들거나 Stage 1 head를 다시 덮어쓰지 않는다.
- Optimizer/LR/학습 step 복원은 Stage 2 resume에서만 수행한다.

`strict=False`로 실패를 숨기지 않는다. 허용된 새 key/첫-layer shape 변화만 처리하고, 그 밖의 누락·불일치는 보고하거나 fail-fast한다. Load 이후 builder의 전체 weight initializer가 복사/zero-padding을 덮어쓰지 않도록 순서를 검증한다.

Checkpoint metadata에 stage, coordination variant, dimensions, source checkpoint, source config/commit, freeze policy, RSI mode를 기록한다.

---

## 6. Stage 2 scenario sampler

### 6.1 첫 학습 범위

```text
num_agents = 2
num_objects = 3
num_goals = 2
max_edges_per_agent = 2
edge_capacity = 4
one skill bundle per agent per phase
```

위 수치는 **학습 분포 제한**이다. Coordination network와 explicit evaluation graph parser를 2-agent 전용으로 만들라는 뜻이 아니다.

### 6.2 우선 구현할 cooperative family

```text
place_climb:
  A: HOLDING(Ha,Oa), AT(Oa,Ga)
  B: CLIMB(Hb,Oa)

place_sit:
  A: HOLDING(Ha,Oa), AT(Oa,Ga)
  B: SIT(Hb,Oa)

place_stack:
  A: HOLDING(Ha,Oa), AT(Oa,Ga)
  B: HOLDING(Hb,Ob), ON_TOP(Ob,Oa)
```

각 family 안에서 agent role swap, object/goal binding, 초기 위치·방향·거리 등을 randomize한다. User의 현재 Stage 1이 지원하는 geometry/skill 범위를 유지한다.

독립된 2-agent bundle 조합도 control/retention sample로 지원한다. 첫 mixture의 구현 기본값은 cooperative 0.9, independent 0.1이며, cooperative 3 family는 균등이다. 이 비율은 검증된 값이 아니므로 config로 쉽게 변경 가능하게 한다.

고정 `place_climb` 등은 작은 smoke/debug 실행용 preset으로 지원하고, 실제 training은 mixture를 쓸 수 있게 한다. 인덱스 A/B/Oa에 역할을 고정하지 않는다.

### 6.3 Invalid 판정

**제외할 것:**

- 한 agent에게 서로 다른 bundle 2개를 한 phase에 부여: `SIT+CLIMB`, `SIT+CARRY` 등.
- Placement에 필요한 HOLDING이 없거나, HOLDING target과 AT/ON_TOP source가 다름.
- `ON_TOP(O,O)` 또는 물리적으로 모순되는 support cycle.
- 같은 object를 서로 다른 agent가 동시에 상충하는 위치/상태로 조작하도록 요구.
- 첫 toy에서 두 agent가 같은 object를 동시에 HOLDING하는 공동 운반. 별도 bonus이므로 자동 활성화하지 않는다.
- 서로 배타적인 support footprint/contact 목표를 동시에 요구하는 배치.
- 도달 불가능한 높이/크기/목표, reset 시 penetration, 불안정한 초기 지지 상태.

**자동으로 invalid로 처리하면 안 되는 것:**

- A가 운반할 Oa를 B가 SIT/CLIMB 대상으로 쓰는 것.
- A의 운반 대상 Oa를 B의 ON_TOP target support로 쓰는 것.
- Explicit evaluation에서 3명 이상이 하나의 물체에 연결된다는 이유만으로 reject하는 것.

Graph semantics의 모순과 physical feasibility를 구분한다. 새 Stage 2 sampler를 별도 구현하거나 mode-specific validation을 두고, Stage 1 validator를 광범위하게 완화하지 않는다.

### 6.4 중요: edge set은 sequence가 아니다

Phase 동안 current task edge set 전체가 backbone relation bias와 coordination branch에 함께 제공된다. 어느 agent를 먼저 움직이게 하는 ground-truth phase gate는 추가하지 않는다.

여러 cooperative scenario를 시간적으로 연결하는 것은 별도 long-horizon 평가에서만 수행한다.

---

## 7. RSI — 필수 구현 항목

### 7.1 기본 원칙

Stage 2에서 한 agent는 한 bundle만 수행하므로 RSI 종류는 그 bundle로 결정한다. `HOLDING+AT/ON_TOP`의 두 edge를 별도 행동으로 보고 RSI를 각각 뽑지 않는다.

기존 Stage 1의 motion library, reference-state sampling, retargeting, AMP history 초기화를 재사용한다. 새로운 STAND_UP skill, 임의 idle animation, RSI 전용 새 행동 label을 만들지 않는다.

다만 **두 agent가 같은 object를 사용하는 scene에서는 agent별 RSI를 독립적으로 완성한 뒤 합치면 안 된다.** Shared object의 pose/velocity는 하나여야 한다.

### 7.2 첫 toy의 구체적인 RSI policy

아래 확률은 출발점으로 둔 구현 기본값이다. 실제 motion 이름과 Stage 1 지원 목록에 맞춰 검증하고, config에서 변경 가능하게 한다.

| 현재 bundle / 상황 | 첫 toy RSI |
|---|---|
| HOLDING | `loco 0.5 / pickUp 0.5` |
| HOLDING+AT | `loco 0.5 / pickUp 0.1 / carryWith 0.4` |
| HOLDING+ON_TOP | `loco 0.5 / pickUp 0.1 / carryWith 0.4` |
| SIT on a teammate-manipulated shared object | `loco 1.0` — 아직 그 물체에 앉아 있지 않은 상태 |
| CLIMB on a teammate-manipulated shared object | `loco 1.0` — 아직 그 물체 위에 올라가지 않은 상태 |
| Independent SIT/CLIMB | 기존 Stage 1 template RSI 유지 가능 |

설명:

- `loco`는 기존 locomotion reference initialization이며, 별도의 “일어서기 단계”를 넣는 것이 아니다.
- Carry 중인 초기화는 일부 유지한다. 모든 episode를 ground pickup부터만 시작시키면 dependency가 아니라 pickup 재탐색에 학습이 소모될 수 있다.
- 첫 cooperative toy에서는 `putDown`/완료된 support interaction RSI를 기본적으로 빼고 dependency가 실제로 필요한 시작 상태를 만든다.
- 정확한 pre-sit/pre-climb clip 또는 frame filter가 이미 있다면 그것을 사용할 수 있지만, 없는 모션을 가정해 참조하지 않는다.
- Motion category만으로 초기 성공 여부가 완전히 통제되지 않는다. 최종 물리 state를 만든 뒤 actual edge success를 검사한다.

### 7.3 Shared-object initialization authority

예:

```text
A: HOLDING(Oa)+AT(Oa,Ga)
B: CLIMB(Oa)
```

이때:

```text
Oa: scene object로 한 번만 초기화
A가 carry RSI라면: A의 reference와 호환되도록 Oa와 A를 함께 설정
B: 동일한 실제 Oa를 기준으로 loco/pre-contact 상태에 배치
```

B의 climb reference가 다시 Oa 위치를 덮어쓰면 안 된다.

`place_stack`에서는:

```text
A의 RSI ↔ Oa
B의 RSI ↔ Ob
B의 ON_TOP target = 현재 존재하는 동일한 Oa
```

B의 reference가 support Oa를 별도로 생성·이동하거나 다른 pose로 덮어쓰지 않는다.

여기서 “authority”는 reset 시 충돌을 막기 위한 코드 규칙일 뿐 actor에게 주는 upstream/downstream input이 아니다. 고정 agent ID가 아니라 현재 graph의 HOLDING/placement 관계로 결정한다.

### 7.4 Reset 처리 순서

1. Current graph/bundles/bindings를 sample.
2. Graph validity와 shared-object 관계 확인.
3. Joint-compatible RSI 조합을 선택.
4. Manipulated object별 canonical pose/velocity를 한 번 결정하고 해당 human reference를 함께 배치.
5. 나머지 human 및 unassigned object 배치.
6. Goal/support geometry 설정.
7. Penetration, hand/object alignment, feet/support, reachability 및 실제 relation success 검사.
8. 실패한 env만 재시도. 재시도 원인과 횟수를 기록. 조용히 다른 graph로 바꾸지 않는다.
9. Reset tensor를 기존 IsaacGym 제약에 맞춰 commit하고 scene state refresh.
10. Progress history, relation success/history, episode metrics, AMP observation history를 **최종 수락 state와 일치하게** 초기화.

현재 코드의 accepted reset metadata 누적 및 tensor setter 처리 방식을 보존한다. Motion pose만 바꾸고 AMP history를 이전 reference에 남겨두면 안 된다. [R6, R7]

### 7.5 “RSI로 이미 풀어 놓은 task” 방지

Main cooperative training/evaluation에서는 다음을 구분해 기록한다.

```text
start mode: loco / pickup / held
initial upstream placement success
initial downstream SIT/CLIMB/ON_TOP success
initial full-scenario success
```

기본 cooperative 초기화는 downstream 최종 interaction이 이미 성공한 상태와 full-scenario 성공 상태를 제외한다. 초기 HOLDING 성공 자체는 허용한다. 이는 carrying warm-start를 위한 것이다.

완료 근처 RSI는 추후 유지/안정성 보강용 별도 curriculum으로 허용할 수 있지만, 처음부터 전부 금지하거나 전부 허용하는 일반 원칙으로 확대하지 않는다. Main 평가에서는 non-completed initial state를 사용한다.

### 7.6 Joint RSI 예시

```text
place_climb:
  A = carryWith 또는 loco/pickUp
  Oa = A 초기화와 정합
  B = Oa에 아직 올라가지 않은 loco 상태
  Ga = Oa가 이미 배치 성공인 초기 상태를 피하도록 sample

place_sit:
  A/Oa는 위와 동일
  B = Oa에 아직 앉지 않은 loco 상태

place_stack:
  A/Oa와 B/Ob를 각각 정합되게 초기화
  Oa는 공유 support 하나뿐
  Ob는 아직 Oa 위에 최종 배치되지 않음
```

---

## 8. Reward와 성공 판정

### 8.1 기존 relation reward를 기반으로 한다

Stage 1의 state/progress/success 함수와 기본 weight, placement 이후 HOLDING saturation/release 처리, AMP reward, power/collision penalty를 우선 유지한다. Stage 2 기능을 켠다고 Stage 1 reward 자체를 변경하지 않는다.

참조 snapshot은 component weight가 각각 0.2이고, paired placement의 **현재 success**로 HOLDING 보상을 포화시키는 구조다. 사용자의 최신 Stage 1과 다르면 최신 구현을 기준으로 기록한다. [R5]

### 8.2 공유 물체로 연결된 task끼리만 제한적 reward sharing

Policy에는 dependency arrow를 주지 않되, 학습 reward에서는 관련 teammate의 결과를 반영한다.

- Agent i가 사용하는 object set은 그 agent 소유 edge의 **object-type src/dst 전체**로 계산한다.
- Agent j와 그 object set이 겹치면 sharing 후보.
- Self 연결은 제외하고, 독립된 task끼리는 reward를 섞지 않는다.
- Goals나 human index가 우연히 같다고 shared object로 판정하지 않는다.
- `AT(Oa,Ga)`는 owner metadata를 이용해 수행자에게 귀속시킨다.

첫 toy 기본값은 기존 own task reward scale을 유지하는 다음 형태다.

\[
R_i^{task}=\begin{cases}
(1-\lambda)r_i^{local}+\lambda\,\operatorname{mean}_{j\in\mathcal N_i}r_j^{local}, & |\mathcal N_i|>0\\
r_i^{local}, & |\mathcal N_i|=0
\end{cases}
\]

```text
lambda = 0.1   # configurable
r_local = 기존 owner별 edge task reward 합; 이미 mixed된 reward가 아님
```

이후 기존 개인 power/collision 등의 penalty와 AMP 경로를 적용한다. Peer 수가 증가해도 단순 합으로 reward가 비례 증폭되지 않게 peer 평균을 사용한다.

**현재 `local.flip(-1)` sharing과 새 sharing을 중복 적용하지 않는다.** Stage 2에서는 raw `local_task_reward`부터 새 mixing을 한 번만 수행한다. [R5]

주의: 1-edge/2-edge bundle의 own reward 최대값 차이는 기존 scale을 유지하면 남아 있다. 첫 toy에서 보상 설계까지 동시에 바꾸지 말고 raw sum과 edge 평균을 모두 로그에 남긴다. Edge-count normalization을 비교하려면 별도 명시적 ablation/config로 수행한다.

### 8.3 성공 처리

- A만 성공했다고 episode를 종료하거나 A control을 끄지 않는다. B가 행동할 동안 A도 매 step action을 출력한다.
- Success/maintenance/release 의미는 기존 relation 함수와 required-goal 구분을 따른다.
- `A가 과거에 AT 성공한 적 있음` + `B가 다른 시점에 CLIMB 성공한 적 있음`만으로 joint success라고 하지 않는다.
- Full scenario success는 해당 phase가 요구하는 결과들이 **같은 시점에 유효**한지 판정한다. 유지 duration을 쓰면 config에 명시하고 모든 method에 동일하게 적용한다.
- `Place+Climb`의 보상/성공 판정이 원하는 물리적 협력을 실제로 요구하는지 확인한다. 단지 두 edge가 같이 있다는 이유로 특정 행동 순서가 보장되는 것은 아니다.

---

## 9. PPO observation / normalization / replay

모든 Stage 2 actor forward는 그 sample의 observation에 저장된 graph packet으로 src/dst/relation/valid를 복원한다.

**금지:** PPO minibatch를 학습할 때 현재 live environment의 graph를 참조하기. Reset/phase 전환 이후 live graph는 rollout 당시 graph와 다를 수 있다.

- 기존 `semantic_graph_packet`, `parse_semantic_packet`, packet size 계산 함수를 재사용/확장한다.
- Packet width를 임의로 `5×4` 등의 상수로 계산하지 않는다.
- Edge capacity가 바뀌면 observation suffix/normalizer passthrough/buffer shape도 함께 갱신한다.
- Graph index/type/valid와 GTA pose 등 기존 passthrough는 normalize하지 않는다.
- Actor feature를 rollout마다 재계산해도 frozen 결과는 같아야 한다. 캐시는 첫 구현에 필수 아니다.
- Actor backbone freeze와 critic/AMP의 gradient 경계를 정확히 분리한다.

---

## 10. 확장 평가를 가능하게 만드는 구현

### 10.1 두 수준의 평가

**A. 안전한 확장 — 독립 pair 여러 개**

```text
A: HOLDING(Oa)+AT(Oa,Ga)
B: CLIMB(Oa)

C: HOLDING(Oc)+AT(Oc,Gc)
D: SIT(Oc)
```

한 phase에 두 scenario의 전체 6 edges를 동시에 입력한다. 다른 2-edge bundle 조합이면 4명에서 최대 8 edges가 된다.

**B. 목표 확장 — 3명 이상이 하나의 object에 연결**

```text
A: HOLDING(Ha,Oa), AT(Oa,Ga)
B: CLIMB(Hb,Oa)
C: HOLDING(Hc,Oc), ON_TOP(Oc,Oa)
```

총 3 humans / 5 edges를 **하나의 phase에서 동시에** 입력한다. 이를 `{A,B}` 실행 후 `{C}` 실행으로 바꾸면 다른 실험이므로 금지한다.

4명 평가라면 D는 독립 task일 수도 있고, 물리적으로 가능한 추가 shared role일 수도 있다. 무조건 4명 모두 같은 작은 상자를 쓰게 만들지는 않는다.

### 10.2 Network 및 parser 요구사항

- 같은 checkpoint의 같은 parameter로 N/E 증가를 처리.
- MHA query length=N, key/value length=E를 runtime shape로 사용.
- `E`개의 vector를 flatten하여 고정 길이 MLP 입력으로 만들지 않는다.
- 기본 Stage 2 training sampler는 2명 전용이어도 되지만 **explicit evaluation graph compiler는 N-agent를 지원**해야 한다.
- `edge_capacity`는 실행/config 단위의 padding capacity이며 학습 parameter의 shape가 아니다.
- 목표 첫 테스트: `(N,E_capacity)=(2,4),(3,6),(4,8)` 및 실제 valid edge가 3/4/5/6개인 경우.
- 현 환경 제약을 유지하는 첫 eval은 `numGoals=N`, `numObjects>=N`로 구성 가능하다. 예를 들어 4명에서 4~5개 object를 두고 일부는 unused여도 된다.
- Module checkpoint와 env/parser 제한을 구분하고, 추가 index/count hard-code는 찾아서 명시적으로 처리한다.

### 10.3 물리적으로 불가능한 테스트를 성공시키려 하지 말 것

현재 CLIMB와 ON_TOP reward가 둘 다 작은 support의 중심을 강하게 요구하면, `CLIMB+ON_TOP` 동시 요구가 물리적으로 충돌할 수 있다. Cross-attention이 이 모순을 해결해주지는 않는다.

첫 구현에서는 기존 Stage 1 reward/geometry를 보존한다. Higher-order test는 실행 가능한 support 면적과 목표 조건을 확인한 경우만 진행한다. 필요하면 이후 별도 실험으로 feasible-support-region reward를 설계하되, 이번 Stage 2 코드 변경에 몰래 섞지 않는다.

2-agent 학습에서 3명 이상의 공유-role composition이 성공하는지는 **검증할 연구 가설**이다. 입력 shape 지원을 성능 일반화의 증거로 보고하지 않는다.

---

## 11. Long-horizon은 별도 평가/선택적 확장

필수 Stage 3 학습을 이번 구현에 넣지 않는다.

우선 Stage 2 checkpoint를 그대로 사용해:

```text
Phase 1: {A CARRY, B CLIMB, ...}
        ↓ 공동 phase 완료
Phase 2: {A SIT, B CARRY, ...}
        ↓
Phase 3: ...
```

를 평가할 수 있는 구조를 고려한다.

- Phase 내부의 전체 cooperative edge set을 함께 준다.
- Phase 사이에는 simulator reset, human teleport, object 재배치, RSI 재추출을 하지 않는다.
- RSI는 episode 시작에만 수행한다.
- 새 phase graph에 맞춰 goal/reference binding과 progress baseline 등을 갱신하되 물리 상태는 그대로 둔다. Success를 억지로 모두 false로 만들지 말고 새 관계를 실제 state에서 평가한다.
- 처음부터 STAND_UP 명령이나 자동 서기 animation을 끼우지 않는다.
- 이전 단계에서 유지해야 할 목표는 다음 phase의 task specification에도 반영해야 한다. 필요한 유지 관계까지 지워버리고 유지 성능을 기대하지 않는다.
- Transition에서 실패하면 별도 long-horizon adaptation을 논의한다. 추가 학습한 결과를 zero-shot으로 부르지 않는다.

**구현 우선순위는 Stage 2 single-phase와 variable-N/E 평가 지원이다.** Phase runner 전체와 transition training을 먼저 만들다가 본 작업을 지연시키지 않는다.

---

## 12. 검증 항목 — 구현 완료 조건

### 12.1 Network / checkpoint unit tests

1. **Stage 1 regression:** Stage 2 feature flag가 꺼져 있으면 기존 forward·checkpoint loading·config 동작이 유지됨.
2. **초기 함수 보존:** 같은 obs/graph/RMS에서 Stage 1과 새 Stage 2의 action mean 및 fixed sigma가 수치 허용오차 내에서 일치. GPU matmul shape 차이에 따른 작은 부동소수점 오차는 허용하되 오차값을 보고.
3. **Freeze 확인:** 복사한 actor encoder parameter에 grad가 없고 optimizer update 후 값도 불변. Actor RMS update도 정지.
4. **Trainable 확인:** head의 h/c weight와 뒤 layer 전체, grounding, MHA가 optimizer에 포함됨. `W_c=0`인 첫 backward 특성을 고려하고, 두 번 이상의 nondegenerate update에서 coordination gradient 경로를 확인.
5. **Endpoint binding:** 동일 typed edge라도 src/dst가 다른 경우 올바른 각 token을 gather함. Semantic vector만으로 binding을 대신하지 않음.
6. **Edge permutation:** edge 순서와 해당 packet field를 같이 섞어도 action이 일치.
7. **Entity permutation:** 우선 같은 type 내부 permutation과 graph index/owner/action 대응을 함께 바꾼 경우 결과가 대응되어야 함. 기존 type-block layout을 무시한 임의 token shuffle과 혼동하지 않음.
8. **Padding:** valid edge는 동일하고 E_capacity만 늘려도 결과가 일치. Empty edge 입력도 NaN 없이 c=0.
9. **Variable N/E:** 위 2/3/4-agent shape 테스트에서 새 parameter/head 추가 없이 forward와 checkpoint load 성공.
10. **PPO packet consistency:** 서로 다른 graph의 rollout sample을 섞어도 각 sample의 packet으로 feature를 재구성함. Live env graph 변경이 이전 sample 결과를 바꾸지 않음.
11. **Resume:** Stage 2 재시작 시 학습된 W_c와 coordination weights가 보존되고 zero-padding transfer가 재적용되지 않음.

### 12.2 RSI / scene tests

- Shared object별 pose/velocity 초기화 source가 하나임.
- A carry RSI + B CLIMB/SIT-loco 조합에서 hand/object 정합성과 비관통 초기 상태 확인.
- place_stack에서 A/Oa와 B/Ob는 각각 정합하고 shared support Oa가 덮어써지지 않음.
- 초기 downstream/full success 비율, RSI rejection 원인, retry 횟수 기록.
- Reset metadata와 AMP history가 수락한 최종 state와 일치.
- Invalid graph를 retry로 무한히 숨기지 않고 명시적 오류/원인 제공.

### 12.3 학습 smoke test

작은 env 수와 짧은 update로 다음만 먼저 확인한다.

```text
place_climb → finite loss/grad/reward, 실제 head update
place_stack → 동일 + shared support 초기화
place_sit   → 동일 + downstream 초기 성공 차단
```

짧은 smoke를 cooperative 성능 검증으로 보고하지 않는다. IsaacGym/GPU/motion/checkpoint가 없어서 실행 못 한 항목은 미실행으로 명시한다. 긴 학습은 실행 명령을 제공하고 사용자 실험과 구분한다.

### 12.4 Performance profiling

같은 env/minibatch/N/E, warm-up, device 조건에서 측정한다.

```text
actor encoder forward
coordination forward/backward
action head forward/backward
critic update
AMP discriminator update
rollout / simulator step
peak allocated CUDA memory
trainable / frozen parameter counts
```

GPU 비동기 실행을 고려해 CUDA events 또는 동기화된 profiler를 사용한다. 파라미터 수나 MACs만으로 실제 wall-clock speedup을 주장하지 않는다.

---

## 13. 연구 평가 및 로그

### 13.1 기본 비교

| 실험 | 무엇을 확인하는가 |
|---|---|
| Stage 1 기본 skill | 이미 만들어진 skill의 출발 성능 및 retention |
| Stage 1 vs Stage 2, 같은 cooperative scene | B/downstream 및 full scenario success가 개선되는가 |
| Same A task + different B role | A가 상대 role에 맞춰 유용하게 행동을 바꾸는가 |
| 2-agent train → 여러 independent pairs | 학습한 local cooperation이 더 많은 동시 agent로 확장되는가 |
| 2-agent train → 3+ agent shared object | 새로운 higher-order role composition이 가능한가 |
| Zero-shot multi-phase | Reset 없이 다음 cooperative scenario로 전환 가능한가 |

Stage 1과 Stage 2의 environment/task goal/timeout/evaluation 성공 정의를 맞춘다. Stage 2만 외부 scheduler로 유리하게 제어하면 안 된다.

### 13.2 필수 로그

```text
scenario family / role swap / bindings
valid edge count / agent count
per-edge phi, progress, current success
per-agent local reward / mixed reward / partner count
upstream & downstream success / full current scenario success
collision / fall / completion time
initial RSI skill / initial success / reset rejection reason
coordination context norm
first-layer W_c norm / copied-head parameter drift
parameter-group effective LR / gradient norm
attention weights (소수 env, 낮은 logging frequency)
```

`attention가 크다 → 그 edge가 행동의 원인이다`로 단정하지 않는다. Role masking/shuffling/context ablation과 실제 성공 변화로 보조 검증한다.

학습 후 `c=0`으로 만드는 것은 coordination-context ablation이지 원래 Stage 1 복원이 아니다. **Action head를 fine-tune했기 때문에** Stage 1 baseline은 원래 checkpoint를 따로 평가해야 한다.

### 13.3 나중에 추가할 ablation

- Stage 2 action-head fine-tuning only, coordination branch 없음.
- Full Stage 2에서 teammate reward sharing 제거.
- Teammate role/edge 정보의 효과.
- Version A rich K/V vs Version B semantic K / rich V.
- 필요하면 TokenHSI-style frozen head + layer-wise adapters.

첫 구현에 모든 ablation architecture를 동시에 만들 필요는 없다.

---

## 14. 권장 변경 파일과 산출물

아래 새 파일명은 제안이다. 실제 로컬 구조에 맞게 최소 변경으로 배치한다.

### 14.1 Model / learner

```text
수정:
  tokenhsi/learning/multi_agent/amp_network_builder_ma.py
    - 선택적으로 전체 post-TF token 반환
    - Stage2 action path / 128D head
    - 기존 Stage1 disabled 경로 보존

신규 후보:
  tokenhsi/learning/multi_agent/coordination_head.py
    - grounded edge projection
    - human-to-edge MHA
    - valid/empty masking

신규 후보:
  tokenhsi/learning/multi_agent/stage2_transfer.py
    - 명시적 Stage1 load / head expansion / report

수정:
  tokenhsi/learning/multi_agent/ma_agent.py
  tokenhsi/learning/multi_agent/ma_players.py
  tokenhsi/learning/multi_agent/scene_normalizer.py
    - Stage2 transfer/resume/freeze/RMS/evaluation 경로
    - 필요 시에만 변경
```

### 14.2 Task / RSI / graph

```text
신규 후보:
  tokenhsi/utils/edge_stage2_spec.py
    - cooperative sampler / explicit N-agent graph validation

신규 또는 mode-specific 수정:
  tokenhsi/env/tasks/multi_agent/edge_stage2_reward.py
    - 기존 local reward 재사용 + sharing을 한 번 적용

관련 기존 경로:
  tokenhsi/utils/edge_scenario_spec.py
  tokenhsi/utils/edge_stage1_spec.py
  tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py
  tokenhsi/env/tasks/multi_agent/edge_ontop_task.py
    - joint RSI / count assumptions / explicit graph support
```

`graph_packet`, state reward, motion/AMP 코드를 불필요하게 복제하지 않는다. 반대로 Stage 2를 Stage 1 schema_version=9로 위장해서 legacy validator의 의미를 바꾸지도 않는다. Stage 2 mode/version과 checkpoint metadata를 명확하게 추가한다.

### 14.3 Config / scripts / tests

```text
신규 config 후보:
  tokenhsi/data/cfg/train/rlg/amp_ma_stage2_coordination.yaml
  tokenhsi/data/cfg/multi_agent/approach_stage2_coordination.yaml

신규 script 후보:
  tokenhsi/scripts/multi_agent/approach_stage2_coordination_train.sh
  tokenhsi/scripts/multi_agent/approach_stage2_coordination_test.sh
  tokenhsi/scripts/multi_agent/approach_stage2_coordination_smoke.sh

신규 tests:
  model transfer / freeze / variable-count / masking / graph packet
  cooperative sampler / invalid cases / shared-object RSI
```

Stage1 import checkpoint와 Stage2 resume checkpoint를 인자/환경변수에서 구분한다. 임의 경로를 hard-code하지 않는다.

---

## 15. Config 설계안 — 실제 schema에 맞춰 구현

아래는 신규 설정의 의미를 보여주는 설계안이다. 기존 코드가 이미 인식하는 완성 config라는 뜻은 아니다.

```yaml
# network 아래 신규 coordination 설정 예시
coordination:
  enabled: true
  variant: rich_kv                 # Version A
  d_model: 64
  edge_semantic_dim: 64
  grounding_hidden: 128
  grounding_dim: 64
  num_heads: 2
  dropout: 0.0
  attention: softmax
  use_valid_task_edges_only: true
  explicit_dependency_input: false
  context_dim: 64
  freeze_actor_encoder: true
  freeze_actor_obs_rms: true
  head_mode: concat_context_finetune
  init_head_context_columns: zero
  stage1_checkpoint: null          # 실행 시 명시. 미지정이면 명확한 오류를 낸다

# optimizer / learner 설정 예시
stage2_optimization:
  learning_rate: 2.0e-5
  coordination_lr_scale: 1.0
  action_head_lr_scale: 1.0
  reset_optimizer_on_stage1_import: true

# sampler 설정 예시
stage2_scenarios:
  train_num_agents: 2
  train_num_objects: 3
  train_num_goals: 2
  max_edges_per_agent: 2
  edge_capacity: 4
  one_bundle_per_agent: true
  cooperative_probability: 0.9
  independent_probability: 0.1
  cooperative_families: [place_climb, place_sit, place_stack]
  random_role_swap: true
  random_object_goal_binding: true
  shuffle_edge_order: true
  allow_shared_objects: true
  allow_joint_holding: false
  allow_higher_order_explicit_eval: true

stage2_reward_sharing:
  enabled: true
  partner_rule: shared_object_in_owned_edges
  lambda: 0.1
  peer_reduce: mean
  source: raw_local_task_reward
  apply_once: true

stage2_rsi:
  mode: joint_consistent
  shared_support_user_init: loco
  template_probabilities:
    HOLDING: {loco: 0.5, pickUp: 0.5}
    HOLDING_AT: {loco: 0.5, pickUp: 0.1, carryWith: 0.4}
    HOLDING_ON_TOP: {loco: 0.5, pickUp: 0.1, carryWith: 0.4}
  independent_sit_climb: inherit_stage1
  reject_initial_downstream_success: true
  reject_initial_full_success: true
  allow_initial_holding_success: true
  initialize_shared_object_once: true

long_horizon:
  required_training_stage: false
  zero_shot_eval_first: true
  reset_between_phases: false
  rsi_between_phases: false
```

Config key 명칭과 계층은 기존 schema 검증과 충돌하지 않게 적용하고, 실제 최종 YAML 및 실행 명령을 산출한다.

---

## 16. Codex 실행 순서와 완료 보고

1. 로컬 Stage 1 코드/config/checkpoint contract를 확인하고 원격 참조와 차이를 기록.
2. Feature flag와 Stage 2 model path, Stage1-preserving head transfer부터 구현.
3. Pure PyTorch unit tests: shape, identity, freeze, gradients, masking, permutation, resume.
4. Cooperative sampler와 joint RSI, shared-object reward mixing 연결.
5. 가능한 환경에서 small simulator smoke test.
6. Explicit N-agent evaluation graph와 checkpoint 재사용 shape test.
7. 실행 가능한 train/test 명령, config, 테스트 결과/미실행 항목을 정리.

완료 보고에는 다음을 포함한다.

```text
변경 파일 목록 / Stage1 regression 결과
실제 Stage1 checkpoint와 config
모델 shape 및 trainable/frozen parameter 수
stage2_transfer_report.json
RSI 기본값 / shared-object 초기화 규칙 / failure diagnostics
실제 optimizer LR
실행한 테스트 / 실패 또는 미실행 테스트
Stage2 시작 전 action equivalence 오차
2-agent 학습 및 3/4-agent 평가 명령
남은 물리적/환경적 제한
```

**피해야 할 설계 변경:**

- Backbone 전체 fine-tuning으로 몰래 전환.
- Action head를 random initialization.
- Action head를 freeze하고 h만 residual 수정하는 이전 안으로 회귀.
- Actor마다 독립 head/coordination network 생성.
- Edge를 4개까지만 받을 수 있게 flatten.
- Shared-object task를 agent별 phase로 강제 분리.
- 현재 scene의 graph를 PPO replay sample에 대신 사용.
- 공유 물체를 agent별 RSI에서 여러 번 overwrite.
- Stage 3 / platform / PUSH/PULL / new stand-up skill을 필수로 추가.
- Higher-order 성공이나 학습 속도 향상을 실행 없이 확정적으로 보고.

---

## 17. 한 장 요약

```text
현재 phase의 scene observation + 전체 task edge packet
                         |
                         v
           Frozen Stage-1 actor encoder
      (entity tokenizers + edge semantics + GTA Transformer)
                         |
              전체 post-TF entity tokens
                 /                   \
            Human h_i             z_src, z_tgt
                |                      |
                |                 edge64와 concat
                |                      |
                |              trainable phi: 192→64
                |                      |
                Q                     K,V
                 \                    /
                  Shared softmax cross-attention
                              |
                        c_i : 64D
                              |
                        [h_i | c_i]
                              |
               Trainable action head: 128→1024→512→32
                 init: first W=[W_stage1,0]
                 init: all other weights=Stage1
                              |
                       각 human의 action
```

**학습:** 2-agent, 단일 cooperative phase, object sharing 허용, joint-consistent RSI.  
**검증:** 협력 성능 → 상대 role에 따른 행동 변화 → 더 많은 agent/edge → unseen shared-object composition.  
**추후:** 여러 phase의 reset 없는 연결을 먼저 zero-shot 평가하고, 필요 시 별도 adaptation.

---

## 참고한 실제 코드

아래 링크는 확인한 commit에 고정되어 있다. 문서의 새 module/config는 제안이며, 아래 파일에 이미 구현되어 있다는 뜻이 아니다.

- **[R1] Model / encoder / head**  
  `tokenhsi/learning/multi_agent/amp_network_builder_ma.py`  
  https://github.com/KSH3880/CVPR2027/blob/1c119b41b47bdbe49ad0a0fea7802da3cafca5ef/tokenhsi/learning/multi_agent/amp_network_builder_ma.py
- **[R2] Semantic edge encoding / PPO packet**  
  `tokenhsi/learning/multi_agent/edge_context_encoder.py`  
  https://github.com/KSH3880/CVPR2027/blob/1c119b41b47bdbe49ad0a0fea7802da3cafca5ef/tokenhsi/learning/multi_agent/edge_context_encoder.py
- **[R3] Actual training dimensions / LR**  
  `tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation.yaml`  
  https://github.com/KSH3880/CVPR2027/blob/1c119b41b47bdbe49ad0a0fea7802da3cafca5ef/tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation.yaml
- **[R4] Scenario sampler / fixed-count validation**  
  `tokenhsi/utils/edge_scenario_spec.py`  
  https://github.com/KSH3880/CVPR2027/blob/1c119b41b47bdbe49ad0a0fea7802da3cafca5ef/tokenhsi/utils/edge_scenario_spec.py
- **[R5] Existing local reward / two-agent sharing**  
  `tokenhsi/env/tasks/multi_agent/edge_stage1_reward.py`  
  https://github.com/KSH3880/CVPR2027/blob/1c119b41b47bdbe49ad0a0fea7802da3cafca5ef/tokenhsi/env/tasks/multi_agent/edge_stage1_reward.py
- **[R6] Environment / reference-state initialization**  
  `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py`  
  https://github.com/KSH3880/CVPR2027/blob/1c119b41b47bdbe49ad0a0fea7802da3cafca5ef/tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py
- **[R7] Joint scene reset / AMP reset metadata integration**  
  `tokenhsi/env/tasks/multi_agent/edge_ontop_task.py`  
  https://github.com/KSH3880/CVPR2027/blob/1c119b41b47bdbe49ad0a0fea7802da3cafca5ef/tokenhsi/env/tasks/multi_agent/edge_ontop_task.py
- **[R8] Actual character action dimension**  
  `tokenhsi/env/tasks/humanoid.py`  
  https://github.com/KSH3880/CVPR2027/blob/1c119b41b47bdbe49ad0a0fea7802da3cafca5ef/tokenhsi/env/tasks/humanoid.py
- **[R9] PPO scene batching / transfer / normalization**  
  `tokenhsi/learning/multi_agent/ma_agent.py`  
  https://github.com/KSH3880/CVPR2027/blob/1c119b41b47bdbe49ad0a0fea7802da3cafca5ef/tokenhsi/learning/multi_agent/ma_agent.py
