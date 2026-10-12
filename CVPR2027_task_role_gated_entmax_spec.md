# CVPR2027 — Task/Role-Gated GTA Attention + Entmax 1.5

> **Codex 구현 명세서** | 기준 브랜치: `stage1_task_emb`  
> 목표: 기존 실험을 **보존**하고, Stage 1에 신규 독립 실험을 추가한다. 이 문서는 **설계안**이며, 코드 구현·장기 학습·성능 향상이 검증되었다는 의미가 아니다.

## 0. 요청과 핵심 원칙

기존 모델은 Human/Object/Goal entity token으로 장면 전체를 표현하고, task가 attention의 **scalar bias**에 주로 반영된다. 토큰 수가 늘어날 때 무관한 entity의 attention mass가 증가할 수 있고, 서로 다른 task가 같은 대상에 집중하면 attention map만으로는 행동별 feature를 충분히 다르게 만들기 어렵다는 가설이 있다.

**이번 변경은 한 가지 attention 설계로 통합한다.**

- **Task/Role embedding → 하나의 Gate → Key와 Value 동시 변조**: 각 관계에서 어떤 feature를 비교하고 전달할지 바꾼다.
- **Entmax 1.5**: 전체 token을 attention 후보로 두되, 점수가 낮은 token에는 정확히 0 weight를 줄 수 있도록 한다.
- **기존 GTA 유지**: 현재 코드가 사용하는 4D direct-sum representation과 호환되도록 Gate를 **4D 블록당 하나의 스칼라**로 만든다.
- **기존 additive relation bias 제거**: QK score에 학습된 task/role scalar를 더하는 별도 경로를 사용하지 않는다.
- **단일 공유 정책 유지**: task별/agent별 전용 Transformer, action head, 마스킹된 task-only head, 새 Task token, 별도 cross-attention을 추가하지 않는다.
- **모든 non-task token도 계속 읽을 수 있어야 한다**: 충돌 회피를 위해 hard mask나 task edge만 남기는 neighbor restriction은 사용하지 않는다.

> 성공 기준은 Gate 값이 서로 달라지는 것이 아니라, **(a) 동일한 상태에서 task/role 교체 시 행동이 올바르게 변화**, **(b) 관련 없는 entity를 추가해도 action이 안정**, **(c) 가까운 비-task 장애물에는 반응**하는지다.

## 1. 현재 저장소 계약 (구현 전 반드시 확인)

- Repository: `KSH3880/CVPR2027`, branch: `stage1_task_emb`.
- 환경 YAML: `tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding.yaml`
- 학습 YAML: `tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_task_embedding.yaml`
- 실행 문서: `markdowns/config.md`; 코드 위치 문서: `markdowns/structure.md`; 변경 이력: `changelog.md`.
- 기존 핵심 코드:
  - `tokenhsi/learning/multi_agent/amp_network_builder_ma.py` — `RelationTransformerLayer`, GTA Q/K/V, attention 및 actor/critic encoder.
  - `tokenhsi/learning/multi_agent/task_role_encoder.py` — 현재 task embedding / bias 및 task role 실험.
  - `tokenhsi/utils/task_role_spec.py` — `task_graph_packet`, `task_edges`와 방향성 role.
- 현재 모델: `d_model=64`, `num_layers=4`, `num_heads=2`, `head_dim=32`, FFN `64→512→64`, `gta.enable=true`, `representation=se3_direct_sum`.
- 현재 Stage 1: 2 humans / 4 objects / 2 goals → **N=8 tokens**, env batch 2048. Policy task는 `sit`, `climb`, `carry_at`, `carry_ontop` (ID 0–3). 단독 HOLDING은 샘플하지 않음.
- 입력 suffix: agent당 `[valid, task, actor, payload, target]` 5개. **Primitive reward graph와 정책 task graph를 혼동하지 말 것.**
- 기존 `task_type_embedding` 모드는 6개(`4 tasks + NONE + SELF`) embedding 및 `physical type pair`별 scalar bias projection을 사용. **이 모드/체크포인트는 변경하지 않고 신규 variant로 분리**한다.
- 현재 `RelationEncoder` 설정 검사는 `relation_bias`와 `relation_bias_mode`를 묶어 검증한다. 새 모드를 추가할 때 additive bias를 실수로 계산하지 않도록 **검증 로직을 명확히 분기**해야 한다.
- Actor/Critic encoder는 분리되어 있다. 신규 Gate도 각 branch에 독립적으로 둔다. **Stage 2 연동은 이번 변경 범위 밖**이다.

## 2. Task/Role 입력 — Task embedding이 메인

### 2.1 방향성 관계 (기존 task_edges 사용)

관계 `i → j`는 **query/receiver token `i`가 key/value token `j`를 읽는 방향**으로 정의한다.

| Task | Directed edge | Role ID | One-hot |
|---|---|---:|---|
| `sit` | H→O(target) | 1 | `[0,1,0]` |
| `climb` | H→O(target) | 1 | `[0,1,0]` |
| `carry_at` | H→O(payload) | 0 | `[1,0,0]` |
| `carry_at` | H→G(target) | 1 | `[0,1,0]` |
| `carry_at` | O(payload)→G(target) | 2 | `[0,0,1]` |
| `carry_ontop` | H→O(payload) | 0 | `[1,0,0]` |
| `carry_ontop` | H→O(support/target) | 1 | `[0,1,0]` |
| `carry_ontop` | O(payload)→O(support) | 2 | `[0,0,1]` |

역할 정의: `0=actor→payload`, `1=actor→target`, `2=payload→target`. **동일 task에서 H→payload와 H→target을 구별**하는 용도다. Stage 1에서는 해당 3가지 directed role 조합만 사용한다. 역방향 edge는 현재 packet에 명시되지 않으므로 task gate를 주지 않는다.

### 2.2 Relation input과 Gate

- 학습 가능한 task embedding: `E_T = nn.Embedding(4, 64)`; 모듈에 4개 task ID를 그대로 노출할 것.
- Role: 별도 학습 embedding 없이 fixed **3D one-hot**.
- 물리 type(H/O/G)이나 agent/object ID를 relation 입력에 재차 concatenate하지 않는다. entity type은 이미 tokenizer와 type embedding, Q/K/V에 반영된다.
- `[task64 || role_onehot3]` → **67D**.
- Layer별 공유 Gate MLP: `Linear(67,64) → SiLU → Linear(64,16)`.
- 출력 16D는 **2 heads × 8 GTA blocks/head**. 16개의 값을 각각 4회 반복해 **64D K/V Gate** 생성.
- Task embedding과 Gate MLP는 모든 agent/object/task instance에 재사용. Actor와 Critic의 파라미터는 각기 독립. Layer별 Gate MLP는 별도, task embedding은 encoder의 4 layers에서 공유하는 것으로 정의한다.
- 마지막 Linear `weight=bias=0` 초기화. 따라서 시작 Gate는 1이고 처음에는 task gate가 항등 변환이다(첫 optimizer step에서는 마지막 projection만 주로 학습됨).

$$
\begin{aligned}
 e_{ij}&=[\,E_T(t_{ij})\;\Vert\; \operatorname{OneHot}_3(r_{ij})\,]\in\mathbb R^{67},\\
 a_{ij}^{\ell}&=1+\tanh\!\left(W_2^{\ell}\operatorname{SiLU}(W_1^{\ell}e_{ij}+b_1^{\ell})+b_2^{\ell}\right)\in(0,2)^{16},\\
 \bar a_{ij}^{\ell,h}&=\operatorname{repeat\_interleave}\big(a_{ij}^{\ell,h}\in\mathbb R^8,\;4\big)\in\mathbb R^{32}.
\end{aligned}
$$

`e_ij` 자체에 현재 물리 상태를 추가로 붙이지 않는다. **상태 의존성은 Q/K/V에 이미 있으며**, Gate가 상태 기반 QK 비교 및 V 전달을 변조한다. Gate가 고정 task/role 조건이라는 사실을 실험 및 논문에서 명확히 기술한다.

Non-task/self/미연결 edge는 **Gate = all ones**로 두고 그에 대한 learned relation gate 계산은 생략한다.

## 3. GTA와 K/V Gate — 정확한 수식 및 변환 순서

**중요: 현재 코드의 실제 convention을 유지한다.** `build_gta_transforms`는 `g_i=T_i^{-1}`, `ginv_i=T_i`를 반환한다. 현재 Q/K/V 적용은 다음과 같다.

$$
Q_i^G=\rho(g_i^\top)Q_i,\quad
K_j^G=\rho(g_j^{-1})K_j,\quad
V_j^G=\rho(g_j^{-1})V_j.
$$

- 여기서 `rho`는 **4×4 블록 변환을 head의 8개 블록에 각각 적용**하는 코드의 `apply_gta_transform`이다.
- 각 layer 입력: `X [B,N,64]` → Q/K/V `[B,2,N,32]` → GTA Q/K/V `[B,2,N,32]`.
- 각 task edge `(i,j)`에서 16D Gate를 두 head의 8개 블록으로 나눈다.
- **GTA 적용 후** 각 `(i,j)`의 `K_j^G`, `V_j^G`에 block-wise Gate를 적용한다. 아래처럼 sparse correction을 사용하면 dense pairwise K/V `[B,2,N,N,32]`를 만들지 않아도 된다.

**Key / Attention logit** (head `h`, dimension `d_h=32`):

$$
\begin{aligned}
K_{ij}^{*,\ell,h}&=\bar a_{ij}^{\ell,h}\odot K_j^{G,\ell,h},\\
s_{ij}^{\ell,h}&=\frac{(Q_i^{G,\ell,h})^\top K_{ij}^{*,\ell,h}}{\sqrt{32}}.
\end{aligned}
$$

일반 연결에는 `K* = K^G`; **기존 learned additive relation bias는 사용하지 않는다.** 구현상 `Q^G @ (K^G)^T`를 먼저 계산한 뒤 유효 edge에만

$$
\Delta s_{ij}^{\ell,h}=\frac{(Q_i^{G,\ell,h})^\top\big[(\bar a_{ij}^{\ell,h}-1)\odot K_j^{G,\ell,h}\big]}{\sqrt{32}}
$$

를 `scatter_add`로 보정하면 수학적으로 동일하다.

**Value / aggregation**:

$$
\begin{aligned}
V_{ij}^{*,\ell,h}&=\bar a_{ij}^{\ell,h}\odot V_j^{G,\ell,h},\\
Z_i^{G,\ell,h}&=\sum_j\alpha_{ij}^{\ell,h}V_{ij}^{*,\ell,h},\\
O_i^{\ell,h}&=\rho(g_i)Z_i^{G,\ell,h}.
\end{aligned}
$$

효율적인 구현은 기존 `alpha @ V^G`, 이어서 **world/aligned GTA representation에서** task edge마다

$$
\Delta Z_i^{G,\ell,h}
=\sum_{(i,j)\in \mathcal E_{\rm task}}\alpha_{ij}^{\ell,h}
\big[(\bar a_{ij}^{\ell,h}-1)\odot V_j^{G,\ell,h}\big]
$$

를 scatter한 다음 **마지막에 한 번만** `apply_gta_transform(g_i, ...)`을 수행하는 방식이다. 이렇게 하면 기존 GTA의 빠른 토큰별 변환을 유지한다.

### Block-wise Gate가 GTA와 왜 호환되는가?

Head 32D = 8개의 4D 블록. Gate를 각 블록의 4개 성분에 동일하게 적용한다.

$$
D(a)=\mathrm{diag}(a_1 I_4,\ldots,a_8 I_4),\qquad
\rho(T)=\mathrm{diag}(T,\ldots,T).
$$

따라서 임의의 GTA 4×4 변환 `T`에 대하여

$$
D(a)\rho(T)=\rho(T)D(a).
$$

Gate는 Task/Role에서만 생성하므로 `T`에 따라 Gate 값이 바뀌지 않는다. **이 결과는 추가 Gate가 GTA의 4D 블록 변환과 교환 가능함을 보장하지만, tokenizer/QKV/FFN을 포함한 전체 정책의 엄밀한 SE(3)-equivariance 증명은 아니다.** 4D 블록 내 독립 4개 Gate를 사용하는 변경은 이 교환 성질을 일반적으로 깨뜨리므로 이번 실험에서 제외한다.

## 4. Softmax → Entmax 1.5

전체 토큰 `j=1...N`은 마스킹 없이 score 후보로 둔다. Task relation은 K/V Gate에만 사용하며, non-task 장애물도 QK score가 높으면 선택될 수 있어야 한다.

$$
\alpha_{ij}^{\ell,h}=\operatorname{entmax}_{1.5,j}(s_{ij}^{\ell,h}),\quad
\alpha_{ij}\ge 0,\quad\sum_j\alpha_{ij}=1.
$$

1.5-entmax의 해를 다음처럼 표현할 수 있다.

$$
\alpha_{ij}=\Big[\max\{0,\tfrac12s_{ij}-\tau_i\}\Big]^2,
\qquad\sum_j\alpha_{ij}=1.
$$

`tau_i`는 query/head/sample마다 따로 정해지는 threshold다. Softmax와 달리 낮은 점수에는 정확히 0 weight를 줄 수 있지만, **어떤 토큰이 필요한지 자체를 Entmax가 이해하는 것은 아니다.** 충돌 회피 학습과 task 학습이 해당 점수를 형성해야 한다.

- 초기 실험에서는 **fixed entmax 1.5**를 사용하고, learnable alpha/temperature, top-k, task-only mask, 별도 scene head 등은 넣지 않는다.
- numerically stable하고 gradient를 지원하는 정확한 entmax 1.5 구현을 사용한다(환경에 라이브러리가 없으면 잘 검증된 PyTorch 구현 추가). 정규화, 유한 gradient, zero-support case가 없는지 테스트한다.
- 최대 2048 env × 긴 PPO rollout을 고려해 dense pairwise *feature* tensor는 만들지 않되, 현재와 동일한 dense `[B,2,N,N]` score/attention은 허용한다. **Entmax로 계산 복잡도가 자동으로 O(N)으로 줄었다고 주장하지 말 것.**
- Task edge가 Entmax에서 0이 되는 비율도 로깅한다. 필요한 token이 지워져 성능이 나빠질 가능성은 실험으로 확인해야 한다.

## 5. 한 Layer의 정확한 Tensor Shape 및 구현 의사코드

| Tensor | Shape (일반) | Stage 1 기본 B=2048,N=8,E≤6 |
|---|---|---|
| `X` | `[B,N,64]` | `[2048,8,64]` |
| `Q,K,V`, GTA 변환 후 | `[B,2,N,32]` | `[2048,2,8,32]` |
| `src,dst,task,role,valid` | `[B,E]` | `[2048,6]` 최대 슬롯 |
| task embedding | `[B,E,64]` | `[2048,6,64]` |
| role one-hot | `[B,E,3]` | `[2048,6,3]` |
| gate MLP input | `[B,E,67]` | `[2048,6,67]` |
| gate 4D block factors | `[B,2,E,8]` | `[2048,2,6,8]` |
| expanded gate | `[B,2,E,32]` | `[2048,2,6,32]` |
| task edge score delta | `[B,2,E]` | `[2048,2,6]` |
| full score, entmax alpha | `[B,2,N,N]` | `[2048,2,8,8]` |
| task edge V delta | `[B,2,E,32]` | `[2048,2,6,32]` |
| per-token GTA output | `[B,2,N,32]` | `[2048,2,8,32]` |
| output after concat heads | `[B,N,64]` | `[2048,8,64]` |

**주의:** `E≤6`은 학습 batch마다 유효한 연결 수가 다르다는 뜻. 구현에서는 padding 포함 `E_slot=3*num_agents`를 사용하고 `valid`로 무효 연결의 모든 score/value correction을 0 처리한다. 기존 `task_edges`에는 역할 pair가 포함돼 있다.

```python
# PSEUDOCODE: relation attention 한 layer. 실제 코드의 layout/GTA API에 맞춰 변환할 것.
# q,k,v: [B,2,N,32], src/dst/task/role/valid: [B,E]
qG = gta_query(q, poses)              # existing
kG = gta_key(k, poses)                # existing
vG = gta_value(v, poses)              # existing

# Task/role gate only on VALID directed task edges
onehot = F.one_hot(role, 3).to(q.dtype)          # [B,E,3]
e = torch.cat([task_embed(task), onehot], -1)   # [B,E,67]
g16 = 1 + torch.tanh(gate_mlp_layer(e))          # [B,E,16]
g = g16.view(B,E,2,8).repeat_interleave(4, -1)
g = g.permute(0,2,1,3)                          # [B,2,E,32]
dg = (g - 1) * valid[:,None,:,None]

Q_edge = gather_queries(qG, src)                 # [B,2,E,32]
K_edge = gather_values(kG, dst)                  # [B,2,E,32]
V_edge = gather_values(vG, dst)                  # [B,2,E,32]

scores = qG @ kG.transpose(-1,-2) / sqrt(32)    # [B,2,N,N]
delta_score = (Q_edge * dg * K_edge).sum(-1) / sqrt(32)
scores = scatter_add_edge_scores(scores, src, dst, delta_score, valid)

alpha = entmax15(scores, dim=-1)                 # [B,2,N,N]
base_world = alpha @ vG                         # [B,2,N,32]
alpha_edge = gather_edge_attention(alpha, src, dst)
delta_world = scatter_add_to_receivers(
    alpha_edge[...,None] * dg * V_edge, src, valid, N)
out = gta_return(base_world + delta_world, poses)

# existing output projection, residuals, norms and FFN
x = original_transformer_update(x, out)
```

- `scatter_add` 시 invalid/padding이 `src=0,dst=0`으로 매핑되어 있어도 `valid=0`이면 **0을 더하게 만들 것**. 중복 유효 연결이 생기면 score delta를 합산한다. 향후 동일 directed pair에 서로 다른 task가 동시에 등장하는 경우 gate 합성 규칙은 별도 논의 필요(Stage 1 범위 밖).
- Role one-hot은 기존 `task_edges`의 `(src_role,dst_role)` `(0,1)/(0,2)/(1,2)`를 각 0/1/2로 결정. `src/dst`는 실제 **entity token index**이므로 shuffle 후 다시 재매핑해야 한다.
- Task/Role 조합은 제한적이므로 **4 tasks × 3 roles = 12조합의 Gate를 layer마다 한 번 계산한 뒤 gather**하는 최적화도 가능하다. 이 경우 학습 gradient, task permutation, endpoint permutation을 동일하게 보장할 것.

## 6. 확장성 의미와 Stage 2 연결

- **Parameter scalability**: learned task embedding + layer별 Gate MLP + QKV + action head는 `num_agents`, `num_objects`와 무관하다.
- **Set/permutation handling**: human/object/goal 토큰 순서 및 task edge 레코드 순서를 섞어도 최종 action/value와 gradient가 동일해야 한다(동일 연산의 floating-point tolerance 내).
- **Attention candidate는 전체 scene**: task edge가 아닌 주변 Human/Object도 QK로 attention 후보에 남는다. collision awareness를 구조적으로 차단하지 않는다.
- **Selective aggregation**: Entmax가 score가 낮은 무관한 토큰을 정확히 0으로 만들 수 있어, token 수가 증가할 때 희석을 줄일 가능성이 있다. 하지만 task/collision 관련 토큰을 잘못 버릴 위험도 있다.
- **Compute scalability**: dense QK/Entmax는 여전히 N²를 필요로 한다. '계산량 선형화'를 주장하지 않는다.
- **Generalization은 미검증**: 학습한 Task/Role 조합을 새로운 수의 agent/object와 새로운 역할 binding에 적용할 수 있도록 구조를 설계하는 것이며, zero-shot 성공은 실험으로 입증해야 한다.
- **Stage 2 재사용 인터페이스**: `task_embed.weight[task_id]`는 **단독 64D embedding**으로 얻을 수 있게 유지한다. Stage 2의 dependency/coordination에서는 role one-hot 없이 task embedding만 사용하려는 연구 계획. 현재 `GroundedEdgeCoordination` 구현이 이 Task embedding을 직접 사용하지 않으므로 **이번 실험에서 Stage 2를 변경하거나 호환성을 주장하지 말 것**.
- 신규 Task 종류 확장은 task embedding table과 학습/검증이 추가로 필요하다. Task 종류까지 자동으로 무제한 일반화된다는 뜻이 아니다.

## 7. 구현 요청: 기존 실험 보존 및 파일 계약

1. **작업 전** 저장소 안내대로 `git status --short`, `changelog.md`, `markdowns/structure.md`, `markdowns/config.md`의 관련 섹션을 확인. 기존 실행/변경은 그대로 보존.
2. `stage1_task_emb` 실험을 덮어쓰지 않고 **새로운 Stage 1 variant / env YAML / train YAML**을 생성한다. 제안 suffix: `task_role_gate_entmax` (리포지토리 기존 naming 규칙과 충돌하면 일관된 다른 suffix로 조정).
3. 신규 mode에 대해 설정 검증을 업데이트한다. **기존 learned additive task bias는 해당 신규 mode에서 비활성화**해야 한다. 동시에 `relation_message`도 false로 유지한다. 기존 mode의 동작/체크포인트 계약을 변경하지 않는다.
4. `Task/Role Gate` 모듈을 attention layer에 통합한다. Gate 16D → 4x 반복 적용을 GTA의 Q/K/V 처리 순서와 맞춘다. 각 layer가 별도 Gate MLP를 소유, encoder의 task embedding은 layer 간 공유.
5. 기존 actor/critic 분리, AMP, reward, RSI, entity tokenizer, scene observation, goal binding, task packet, collision penalty를 유지한다. **이번 실험은 attention만 변경**한다.
6. 신규 train/test/VNC wrapper와 독립 output, checkpoint `variant/fusion/mode` 검증을 추가한다. 기존 checkpoint와 직접 resume/eval되지 않도록 한다.
7. `markdowns/config.md` 실험 목록 및 실행법, 필요 시 `markdowns/structure.md`, `changelog.md`에 이유/변경/검증 결과를 업데이트한다.
8. 이미 실행 중인 학습 프로세스 중단/재시작 금지. GPU 지정/점유 확인 필요. 별도 실험 scratch. 기존 공유 데이터/RSI cache 원본 수정 금지.

### 추천 train-config 핵심 (설계 예시, 정확한 필드명은 구현과 함께 결정)

```yaml
transformer:
  num_features: 64
  num_layers: 4
  layer_num_heads: 2
  relation_bias: false                   # no additive learned bias
  relation_message:
    enable: false
  relation_attention_mode: task_role_gate_entmax
  task_role_gate:
    enable: true
    task_embedding_dim: 64
    role_onehot_dim: 3
    hidden_dim: 64
    gta_block_size: 4
    shared_gate_for_key_value: true
    zero_init_output: true
  attention_normalizer: entmax15
  gta:
    enable: true
    representation: se3_direct_sum
    translation_scale: 1.0
```

이 YAML은 현재 코드에서 바로 동작하는 형태가 아니다. **새 `relation_attention_mode`와 설정 검증을 먼저 구현해야 한다.** `relation_bias=false`가 기존 Stage 1 guard를 위반한다면, 신규 mode만 허용하는 분기를 명시하고 additive bias 코드를 실행하지 않도록 한다.

## 8. 검증 / Acceptance Criteria

### A. 수학적 단위 테스트

- Gate 초기화: 4 task × 3 role에서 처음 `g16 == 1` (tolerance), `K*/V*` = 기존 GTA K/V. 단, **normalizer를 Entmax로 바꾸므로 전체 모델 출력은 기존 Softmax와 동일하지 않음**.
- Gate block 구조: 16D→64D, 각 4개 연속 차원 값 동일, 2 heads × 8 blocks 정확.
- **GTA commutation**: 임의의 유효 rigid transform `T`와 Task gate `G`에 대해 `G ρ(T)v ≈ ρ(T) Gv`; float tolerance 검증. element-wise arbitrary gate는 일반적으로 이를 만족하지 않음을 음성 테스트로 확인. 전체 policy equivariance를 주장하는 테스트로 오해하지 말 것.
- Sparse score correction을 실제 pairwise gated key의 dense reference와 비교해 전 출력/gradient가 일치.
- Sparse value correction을 실제 pairwise gated value의 dense reference와 비교해 전 출력/gradient가 일치.
- Role 구분: `carry_ontop` H→payload와 H→support가 같은 H→O 타입이어도 서로 다른 role 입력을 받는지 확인. SIT과 CLIMB은 동일 role(1)이지만 다른 task ID 사용.
- Non-task 및 reverse direction: Gate exactly ones; 모든 token은 attention 후보. invalid/padded edge correction=0.
- Entmax: 각 attention row 합=1, weight ≥0, finite forward/backward, 낮은 score에 정확히 zero 가능, reference 구현과 비교; task edge가 zero가 되는 진단 가능.
- Entity/task-edge/token permutation 시 actor/critic 출력 및 gradient invariant/equivariant, 기존 goal/object binding 유지.

### B. 학습 경로 / 평가

- 기존 Stage 1의 reward, episode success, RSI/AMP family, object size distribution, observation width/task packet 유지.
- CPU 단위/회귀 테스트 통과. **GPU 학습 짧은 확인 시 env 2048 유지**, `MAX_ITERATIONS=1` 또는 문서에 맞는 최소 iteration, 별도 `_check` output 사용. 기존 본학습과 output 섞지 않기.
- Actor/Critic의 task embedding, Gate projection에 finite/nonzero gradient와 optimizer 업데이트 확인. 마지막 layer 0-init 시 초기 단계의 gradient 경로가 막혀도 이후 열리는지 확인.
- 테스트용 checkpoint 로드 및 SIT/CLIMB/carry_at/carry_ontop headless evaluation 경로 검증. 장기 성공률·VNC 수행 여부는 실제 결과만 보고.
- 비교 실험은 반드시 **동일한 물리/보상/RSI/AMP와 총 frame budget**에서 수행:
  1. 기존 task embedding + learned bias + Softmax (baseline)
  2. Entmax only (additive bias 없음, relation gate 항등)
  3. Task/Role Gate + Softmax
  4. **Task/Role Gate + Entmax** (본 실험)
- Stage 1 이후 확장 평가: 2H/4O → 4H/8O, 4H/16O 등 **입력 생성기/sampler가 허용하는 범위로 평가 코드 확장 필요**. 모델 파라미터 크기와 실제 rollout success를 별도 기록.
- **필수 diagnostics**: task 대상 Attention mass, 유효 task edge의 Entmax zero-rate, 전체 nonzero support 수, distractor 추가 시 최종 human feature/action 변화, 가까운 non-task 장애물에 대한 attention/충돌 발생률, task별 success 및 gradient norm.

## 9. 구현 범위 밖 / 의도적으로 추가하지 않는 것

- 새로운 Task token, Task-to-Human cross-attention, Task-specific action head/experts, Task별 Transformer, 별도 Task/Scene head 분리.
- K gate와 V gate의 별도 독립 학습, src/dst role embedding 테이블, physical-type pair projection, learnable temperature/alpha, learned scalar relation bias.
- Hard top-k, task-only neighborhood mask, 손으로 정한 collision-distance attention mask.
- Stage 2 dependency 방식 변경, reward 변경, 물리 및 RSI/AMP 데이터 수정.

---

### Codex에게 요청하는 작업 요약

**`stage1_task_emb`를 기반으로 새로운 `task_role_gate_entmax` Stage 1 실험을 독립 구현해줘.** 기존 learned relation bias와 static relation message는 제거하고, `task64 + role one-hot3 → shared-per-layer MLP → 16D GTA block gate` 하나로 **K와 V를 함께 변조**한다. GTA와 교환 가능한 4D 블록 Gate를 사용하고, 전체 scene에 대해 `Entmax 1.5`로 attention을 정규화한다. Sparse task edge gather/scatter를 사용해 dense pairwise `[B,2,N,N,32]` feature 텐서는 생성하지 않는다. 기존 task/reward/RSI/AMP/actor-critic/GTA/공유 action head와 타 실험을 보존한다. 전용 config·스크립트·테스트·문서를 만들고 변경 사항과 **실제로 검증한 결과만** 보고해줘.
