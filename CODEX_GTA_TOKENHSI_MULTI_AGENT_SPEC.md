# CVPR2027 — TokenHSI Multi-Agent GTA Integration Spec for Codex

> **Purpose**  
> Implement a new **one-scene / one-X multi-agent policy** that replaces the current learned pairwise geometry branch with **GTA (Geometric Transform Attention)**, while preserving TokenHSI's humanoid-control observation philosophy and the existing **A2 semantic edge bias**.
>
> This file is intended to be given directly to Codex as the implementation specification.  
> **Read the existing repository before editing. Do not blindly rewrite files.**

---

## 0. External references — read these first

### GTA paper
- **Title:** GTA: A Geometry-Aware Attention Mechanism for Multi-View Transformers
- **Authors:** Takeru Miyato, Bernhard Jaeger, Max Welling, Andreas Geiger
- **Venue:** ICLR 2024
- **arXiv:** https://arxiv.org/abs/2310.10375
- **ICLR proceedings:** https://proceedings.iclr.cc/paper_files/paper/2024/hash/20e6b4dd2b1f82bc599c593882f67f75-Abstract-Conference.html

### Official GTA code
- https://github.com/autonomousvision/gta

The most important GTA equations for this implementation are Eq. (4)–(6) in the paper.

For token features `X`, with
- `Q = X W_Q`
- `K = X W_K`
- `V = X W_V`
- token geometry/group element `g_i`
- representation `rho(g_i)`

GTA defines

```text
O_i =
sum_j softmax_j[
    Q_i^T rho(g_i g_j^-1) K_j
] rho(g_i g_j^-1) V_j
```

and uses the group-representation factorization

```text
rho(g_i g_j^-1) = rho(g_i) rho(g_j)^-1
```

to compute the efficient form

```text
Q'_i = rho(g_i)^T Q_i
K'_i = rho(g_i)^-1 K_i
V'_i = rho(g_i)^-1 V_i

A = softmax(Q' K'^T + semantic_relation_bias)

Ohat_i = sum_j A_ij V'_j
O_i = rho(g_i) Ohat_i
```

This factorized implementation is critical.  
**Do NOT materialize pairwise `[B,H,L,L,d_head]` K/V tensors.**

---

# 1. Repository context

Target repository:

```text
https://github.com/KSH3880/CVPR2027
```

The existing `edge_geo`/clean-scene implementation already contains useful infrastructure:

```text
tokenhsi/env/tasks/multi_agent/humanoid_ma.py
tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py
tokenhsi/learning/multi_agent/amp_network_builder_ma.py
tokenhsi/learning/multi_agent/ma_agent.py
tokenhsi/data/cfg/train/rlg/amp_ma_carry.yaml
tokenhsi/tests/test_ma_scene_policy.py
```

The current clean-scene policy already does the important one-X work:

```text
one simulator environment
    ->
one shared scene observation
    ->
H/O/T tokens
    ->
one Transformer forward
    ->
gather all Human outputs
    ->
shared action head
```

Preserve that high-level architecture.

---

# 2. Goal of this change

The previous approaches have the following trade-off.

### Legacy multi-row / agent-centric replication

For each agent:

```text
agent A frame -> entire scene -> Transformer
agent B frame -> entire scene -> Transformer
agent C frame -> entire scene -> Transformer
...
```

Relative coordinates are explicit, but the Transformer is effectively repeated for each agent.

### Naive one-pass global

```text
one global scene -> one Transformer
```

Compute is efficient, but each agent loses the strong agent-centric geometric representation that original TokenHSI uses.

### Desired GTA design

```text
ONE shared entity set X
+
one geometric frame g_i per token
+
GTA inside attention
```

The desired decomposition is:

```text
x_i = WHAT the entity is / its intrinsic local state
g_i = WHERE its local coordinate frame is
A2  = WHAT typed task/semantic relation exists between two entities
```

GTA should resolve geometric frame relationships inside attention without duplicating the scene.

---

# 3. Non-negotiable design decisions

Implement the following as the **default GTA v0**.

```text
Human raw node   = 223D
Object raw node  = 30D
Target raw node  = 1D constant

token dimension  = 64
num heads        = 2
head dimension   = 32

g_H = Human root position + Human heading rotation
g_O = Object position + Object FULL physical rotation
g_T = Target position + owner's Human heading rotation

rho(g) = block_diag(g, g, ..., g)
         8 copies of 4x4 g
         -> 32 x 32 per head

physical geometry injection = GTA ONLY
semantic task relation      = existing A2 typed semantic edge MLP bias
```

Do **not** combine GTA with:
- old 15D pairwise Geo MLP,
- Geo score,
- Geo message,
- learned geometry bias,
- explicit pairwise relative-velocity branch,
- A1 lookup in place of the selected A2 semantic edge MLP.

Keep old branches available only if needed for historical ablations, but the new GTA configuration must not execute them.

---

# 4. Token layout

For:
- `M` Humans,
- `O` Objects,
- `M` Targets,

use

```text
L = 2*M + O
```

and token order

```text
[ H_0 ... H_(M-1) | O_0 ... O_(O-1) | T_0 ... T_(M-1) ]
```

The assigned objects should retain the existing logical owner ordering:
- logical object slot `a` corresponds to the object assigned to Human `a`,
- extra/unassigned objects follow afterward.

No agent-ID embedding should be required.

---

# 5. Human node H — use TokenHSI-style 223D self state

## 5.1 Raw Human node

Use the existing TokenHSI-style intrinsic Human observation:

```text
H_i raw: 223D
```

For the current 15-body humanoid, conceptually:

```text
root height                          1
relative body positions             14 * 3 = 42
body rotations, tangent/normal      15 * 6 = 90
body linear velocities              15 * 3 = 45
body angular velocities             15 * 3 = 45
------------------------------------------------
total                                           223
```

Use the existing function/path that produces the clean self-state in the Human's **own heading frame**.

Important:
- retain root height;
- do not append global/env-local XY;
- do not append `cos(yaw), sin(yaw)`;
- do not append observer-relative position;
- do not make different H features for different observing agents.

For the GTA clean path, construct this state with the equivalent of:

```text
cleanSceneLocalRootObsPolicy = true
cleanSceneRootHeightObsPolicy = true
```

Do not inherit `false` legacy defaults for these two flags. Otherwise the first
feature becomes a constant zero and the root-rotation block retains absolute yaw, which
contradicts this section.

Each Human has exactly one self-state.

```text
H_A -> one 223D vector
H_B -> one 223D vector
...
```

Then use the existing type-specific tokenizer:

```text
223 -> tokenizer MLP -> 64D
```

and Human type embedding.

## 5.2 Why

This stays close to original TokenHSI:
- body state remains heading-centric,
- local locomotion/control information is preserved,
- absolute yaw is removed from node content,
- the policy does not need to relearn heading invariance from global coordinates.

The Human root height appears both implicitly in the entity position used for `g_H` and explicitly in the 223D TokenHSI state.  
**Keep this deliberate redundancy in v0** because root height is an original control-state feature and we want minimal deviation from TokenHSI.

---

# 6. Object node O — 30D object-local intrinsic state

## 6.1 Raw Object node

Use:

```text
object-local linear velocity       3
object-local angular velocity      3
object-local bbox corners          8 * 3 = 24
------------------------------------------------
total                                        30
```

Mathematically:

```text
v_O_local = R_O^T v_O_world
w_O_local = R_O^T w_O_world
bbox corners = object-local box corner coordinates
```

Then:

```text
30 -> Object tokenizer -> 64D
```

plus Object type embedding.

Do not put into the Object node:
- global position,
- env-local position,
- world orientation,
- owner-relative position,
- owner-relative orientation,
- owner-frame bbox,
- relative target vector.

Those geometric quantities belong to `g_O` and GTA.

## 6.2 Use FULL Object orientation for its frame

The Object content is defined in the **full physical Object local frame**.

Therefore its geometric frame must be:

```text
g_O uses full object orientation
```

not only object heading/yaw.

This is important for consistency:

```text
x_O frame == g_O frame
```

The previous edge-geometry experiment used source heading for some `dp/dv/dw` calculations.  
That was a later edge design choice, not an original TokenHSI requirement. Do not carry it into GTA v0.

---

# 7. Target node T — 1D constant is intentional

## 7.1 Raw Target node

Use

```text
T_i raw = [1.0]
```

Then:

```text
1D -> Target tokenizer -> 64D
```

plus Target type embedding.

The value `1.0` is only a non-empty tokenizer placeholder.  
It does **not** encode target position.

The target's actual geometric information is in `g_T`.

Task ownership is supplied by the existing semantic relation matrix:
- `REL_OWN_GOAL`,
- `REL_OBJECT_GOAL`,
etc.

Therefore Target information is decomposed as:

```text
Target type/entity content  -> Target tokenizer/type embedding
Target position/frame       -> g_T
Target ownership            -> A2 semantic edge bias
```

## 7.2 Do not worry that 1D is "too small"

Transformer input is not 1D after tokenization.

```text
[1] -> Target tokenizer -> 64D token
```

For the current carry task a target is only a **point target**:
- no desired physical orientation,
- no target velocity,
- no target shape.

So there is no additional intrinsic target content to encode.

If future tasks introduce target orientation or other goal attributes, the Target raw node can later be expanded. Do not preemptively add them now.

---

# 8. Geometry g_i — one SE(3) element per token

This is the most important GTA-specific part.

## 8.1 Coordinate convention

The GTA paper uses camera extrinsic matrices mapping world coordinates into a camera/local coordinate system.

Follow the same convention.

Let an entity local frame have:
- env-local position `p_i`,
- local-to-env rotation `R_i`.

Define

```text
T_i = local -> env
```

as

```math
T_i =
[ R_i   p_i ]
[  0     1  ]
```

and define GTA's token group element as

```math
g_i = T_i^-1
```

therefore

```math
g_i =
[ R_i^T   -R_i^T p_i ]
[   0           1     ]
```

So `g_i` maps:

```text
env/shared coordinates -> token i local coordinates
```

Then the relative transformation used by GTA is

```math
g_i g_j^-1
```

which becomes

```math
g_i g_j^-1 =
[ R_i^T R_j      R_i^T (p_j - p_i) ]
[     0                    1          ]
```

This is exactly the desired transform:

```text
token j local frame -> token i local frame
```

That convention must be unit-tested.

---

# 9. Position convention

Isaac Gym actor positions contain simulator-environment offsets.

Before constructing `g_i`, use:

```text
p_env = p_world - env_origin
```

Do not use the absolute simulator grid coordinates.

### Scaling

For GTA v0, prefer:

```text
translation_scale = 1.0
```

i.e. positions in metres, because original TokenHSI task-relative coordinates are also expressed in physical units.

Expose a config option:

```yaml
gta:
  translation_scale: 1.0
```

If numerical ranges later require scaling, allow a **single uniform scalar**:

```text
p_scaled = translation_scale * p_env
```

Do not independently scale X/Y and Z inside an SE(3) transform.

This is a fixed geometry unit conversion, not RunningMeanStd normalization. Start with
`1.0`; if rollout diagnostics show inflated transformed Q/K/V norms or saturated
attention logits, lower it with one uniform scalar (for example `0.2`) for all XYZ.

---

# 10. Entity-specific g definitions

## 10.1 Human

The Human raw 223D content is expressed in its own **heading frame**.

Therefore:

```text
p_H = Human root position - env_origin
R_H = Human heading rotation only
```

not full root pitch/roll rotation.

So:

```math
g_H =
[ R_heading^T   -R_heading^T p_H ]
[      0                 1          ]
```

This aligns `g_H` with the frame used to construct the 223D Human node.

---

## 10.2 Object

The Object 30D intrinsic node is expressed in the Object's **full physical local frame**.

Therefore:

```text
p_O = Object root position - env_origin
R_O = FULL object physical rotation
```

and:

```math
g_O =
[ R_O^T   -R_O^T p_O ]
[   0          1      ]
```

Do not reduce `R_O` to heading only.

---

## 10.3 Target

A point Target has position but no physical orientation.

However GTA needs a frame/group element.

For Target `T_a` owned by Human `H_a`, define a **virtual Target frame**:

```text
origin      = target position
orientation = owner Human heading
```

Thus:

```text
p_Ta = target position - env_origin
R_Ta = heading rotation of Human a
```

and:

```math
g_Ta =
[ R_Ha_heading^T   -R_Ha_heading^T p_Ta ]
[        0                       1         ]
```

This does **not** claim the target physically has an orientation.

It is only a coordinate-frame convention.

### Why owner heading?

For the own Human/Target pair:

```math
g_Ha g_Ta^-1
```

has relative rotation `I`, and its translation is:

```math
R_Ha^T (p_Ta - p_Ha)
```

which matches the form of the original TokenHSI target observation:

```text
target position expressed in the Human heading frame
```

This is why owner-heading is preferred over world-axis identity for Target v0.

---

# 11. What happens to the old 15D geometry?

The previous experimental geometry was:

```text
dp   3
dR   6
dv   3
dw   3
-------
     15
```

Do not feed this 15D vector into GTA.

GTA v0 uses a genuine SE(3) group element:

```text
position + orientation
```

only.

The correspondence is:

```text
old relative position -> represented by g_i g_j^-1
old relative rotation -> represented by g_i g_j^-1

old relative dv       -> removed from geometry branch
old relative dw       -> removed from geometry branch
```

### This does not discard velocity state

Human node already contains Human body linear/angular velocities.

Object node already contains Object linear/angular velocities.

Therefore velocity information remains in node content.

Also note an important original-TokenHSI detail:

Original carry observation rotates the Object's world linear/angular velocity into the Human heading frame; it does **not** explicitly subtract the Human velocity from the Object velocity.

So the old edge experiment's explicit `(v_j - v_i)` and `(w_j - w_i)` terms are not a requirement inherited from original TokenHSI.

For GTA-only v0, remove them.

---

# 12. Compact geometry storage

The old clean-scene path may append per-entity 13D kinematics:

```text
position 3
quaternion 4
linear velocity 3
angular velocity 3
------------------
13
```

For GTA-only geometry, only pose is required:

```text
position 3
quaternion 4
------------------
7
```

Recommended scene observation:

```text
Human nodes   M * 223
Object nodes  O * 30
Target nodes  M * 1
Pose records  L * 7
```

where:

```text
Human pose record  = env-local root position + heading quaternion
Object pose record = env-local object position + full object quaternion
Target pose record = env-local target position + owner heading quaternion
```

The appended pose records are **not ordinary node features** and must bypass running-statistic normalization.

Do not normalize quaternions.

---

# 13. Tokenization

Preserve type-specific tokenizers:

```text
Human tokenizer: 223 -> ... -> 64
Object tokenizer: 30  -> ... -> 64
Target tokenizer: 1   -> ... -> 64
```

and type embeddings.

The exact internal tokenizer hidden sizes can remain current config values unless code inspection reveals a reason not to.

Result:

```text
X: [B, L, 64]
```

There is only one token per physical entity.

---

# 14. rho(g) — GTA v0 representation

Current Transformer:

```text
d_model   = 64
num_heads = 2
d_head    = 32
```

Use the simplest direct-sum SE(3) representation:

```math
rho(g) = g ⊕ g ⊕ ... ⊕ g
```

with 8 copies of the 4x4 matrix.

```text
4 * 8 = 32
```

Therefore:

```text
rho(g_i): [32, 32] per token per head
```

Conceptually:

```text
head feature 32D
=
[ block0(4) | block1(4) | ... | block7(4) ]

each block transformed by the same 4x4 g_i
```

### General implementation rule

Do not hard-code `8` without an assertion.

```python
assert head_dim % 4 == 0
num_rho_blocks = head_dim // 4
```

For current config:

```text
num_rho_blocks = 8
```

### rho is NOT learned

No MLP is used to construct `rho`.

No learned geometry parameters are added.

`rho(g)` is determined algebraically by `g`.

This follows the GTA paper's valid representation construction:
- a 4x4 rigid transformation is a representation of SE(3),
- block concatenation/direct sum of representations is also a representation.

---

# 15. Do not materialize a 32x32 rho if unnecessary

For efficiency, reshape each head:

```text
[B, H, L, 32]
    ->
[B, H, L, 8, 4]
```

and apply the same 4x4 matrix to each of the eight 4D blocks.

Similarly do not build pairwise relative matrices for all `(i,j)` during normal forward.

You need token-wise:
- `g_i`,
- `g_i^-1`.

The factorized GTA equation handles the rest.

---

# 16. Construct g and g^-1 analytically

Given local-to-env pose:

```math
T_i =
[ R_i  p_i ]
[  0    1  ]
```

use:

```text
g_i^-1 = T_i
g_i    = T_i^-1
```

So avoid a generic matrix inverse where possible.

Construct directly:

```math
g_i =
[ R_i^T  -R_i^T p_i ]
[   0          1      ]
```

and:

```math
g_i^-1 =
[ R_i  p_i ]
[  0    1  ]
```

This is cheaper and numerically cleaner.

Quaternion convention in this repository is `xyzw`.  
Normalize/check quaternions before converting to matrices if needed, but do not alter simulator state silently.

---

# 17. GTA attention implementation

Current QKV:

```text
q, k, v: [B, num_heads, L, head_dim]
        = [B, 2, L, 32]
```

Paper equations treat feature vectors as mathematical column vectors.

Conceptually implement:

```math
Q'_i = rho(g_i)^T Q_i
K'_i = rho(g_i)^-1 K_i
V'_i = rho(g_i)^-1 V_i
```

Then:

```math
score_ij =
(Q'_i)^T K'_j / sqrt(d_head)
```

Add A2 semantic relation bias:

```math
score_ij += b_rel[i,j,head]
```

Then:

```math
A = softmax(score, dim=j)
```

and:

```math
Ohat_i = sum_j A_ij V'_j
O_i    = rho(g_i) Ohat_i
```

Only after `O_i` has been transformed back into token `i`'s local frame should it enter:

```text
output projection
residual connection
LayerNorm
FFN
...
```

This is important: the residual `x_i` is associated with token `i`'s local frame, so GTA output must be mapped back to that frame before residual addition.

---

# 18. Tensor implementation warning: row-vs-column convention

PyTorch tensors store vectors as the last dimension, but the paper writes vectors as columns.

Do not mechanically copy transposes without testing.

Recommended approach:

```text
q4, k4, v4:
[B,H,L,num_blocks,4]

g:
[B,L,4,4]

ginv:
[B,L,4,4]
```

For a mathematical transform `M @ vec_column`, implement with an einsum equivalent to:

```python
torch.einsum("blij,bhlrj->bhlri", M, vec)
```

with the correct `M`:
- Q uses `g.transpose(-1,-2)`,
- K uses `ginv`,
- V uses `ginv`,
- output uses `g`.

Write a dedicated helper and test it against a slow explicit reference implementation.

---

# 19. A2 semantic relation bias — retain it

Keep the current semantic relation taxonomy:

```text
REL_NONE
REL_SELF
REL_TEAMMATE
REL_OWN_OBJECT
REL_OWN_GOAL
REL_OBJECT_GOAL
```

Use the existing A2 typed semantic edge encoder. Its input for every directed pair is:

```text
source entity type embedding 16
relation type embedding      32
target entity type embedding 16
--------------------------------
concatenate                   64 -> shared MLP -> semantic edge 64
```

Project the resulting semantic edge to one scalar per layer/head and add it to the
attention logit. Keep the final layer/head bias projection zero-initialized, matching
the current A2 implementation.

Conceptual attention score:

```math
score_ij =
(Q'_i)^T K'_j / sqrt(d_head)
+
A2_bias(source_type_i, relation_ij, target_type_j)
```

Why keep A2?

GTA answers:

```text
"What is the geometric transform between these tokens?"
```

A2 answers:

```text
"What task/ownership relation connects these entities?"
```

These are orthogonal.

A2 and GTA remain independent: A2 sees only discrete entity/relation types, while GTA
sees only token poses. Keep A1 lookup available for historical ablations, but configure
the GTA experiment with A2.

---

# 20. All H/O/T tokens remain full self-attention participants

Do not turn this into Human-only cross-attention.

Every layer updates:

```text
H <- H/O/T
O <- H/O/T
T <- H/O/T
```

i.e. full `L x L` self-attention remains.

GTA merely changes how Q/K/V are geometrically aligned.

After all Transformer layers:

```text
x: [B,L,64]
```

gather only the Human tokens:

```text
x_h = x[:, :M]          # [B,M,64]
```

Then apply the shared action head:

```text
action_i = shared_action_head(x_h[i])
```

The action head parameters are shared across Humans.

Objects and Targets have no action head.

---

# 21. Multi-agent / entity-count scalability

Do not create parameters that depend on:
- `M`,
- `O`,
- token index,
- agent ID.

Changing the number of Humans/Objects should change only:
- token count `L`,
- relation matrix size,
- pose-record count.

Model weights should stay shape-compatible where possible.

The existing `set_entity_counts()` behavior should remain functional.

---

# 22. Proposed configuration

Recommended clean GTA config:

```yaml
transformer:
  num_features: 64
  tokenizer_units: [256, 128]
  num_layers: 4
  layer_num_heads: 2
  layer_dim_feedforward: 512
  extra_mlp_units: [1024, 512]

  relation_bias: true
  relation_bias_mode: edge_mlp # A2

  gta:
    enable: true
    translation_scale: 1.0
    representation: se3_direct_sum

  # old geometry must be disabled for GTA-only experiment
  geometry:
    enable: false
```

If the current config architecture does not have `gta`, introduce it cleanly.

Do not let `geometry.enable=true` and `gta.enable=true` silently run together.  
Prefer an assertion that prevents accidental simultaneous physical-geometry branches.

---

# 23. File-by-file implementation plan

## 23.1 `tokenhsi/env/tasks/multi_agent/humanoid_ma.py`

Ensure clean Human state is:

```text
223D TokenHSI self state
```

Expected:
- own heading frame,
- root height retained,
- no arena XY/yaw suffix.

Likely keep/use:

```text
_compute_clean_humanoid_nodes(...)
get_clean_humanoid_obs_size()
```

Make sure `get_clean_humanoid_obs_size()` returns the original `_num_obs` = 223 for this asset/config.

---

## 23.2 `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py`

Change clean scene features to:

```text
Human   223
Object   30
Target    1
```

Object node:
- local linear velocity 3,
- local angular velocity 3,
- local bbox 24.

Target node:
- constant ones `[B,M,1]`.

Replace old 13D geometry records with 7D GTA pose records:

```text
Human:
[root_pos-env_origin, human_heading_quat]

Object:
[object_pos-env_origin, full_object_quat]

Target:
[target_pos-env_origin, owner_human_heading_quat]
```

Token order must remain:

```text
[H | O | T]
```

The active GTA clean path uses 7D records. Historical 13D storage belongs only to the
old pairwise-Geo ablation and must not be read by GTA.

---

## 23.3 `tokenhsi/learning/multi_agent/ma_agent.py`

Update scene normalization to:
- normalize the complete raw Human 223D block with one Human type-wise RMS;
- normalize the complete raw Object 30D block with one Object type-wise RMS;
- pass the Target constant 1D through unchanged; do not create/update Target RMS;
- bypass appended GTA pose records;
- never normalize quaternions;
- preserve one-copy `(T,N,scene_obs)` rollout storage.

If pose record changes 13 -> 7, update `kinematic_size`/pose-size plumbing.

---

## 23.4 `tokenhsi/learning/multi_agent/amp_network_builder_ma.py`

This is the primary model change.

### Keep
- tokenizers,
- type embeddings,
- relation matrix,
- A2 typed semantic edge encoder,
- shared Transformer stack,
- shared Human readout,
- separate actor/critic encoders,
- one-X forward.

### Remove from GTA path
- `build_pairwise_geometry`,
- 15D geometry encoder,
- geometry score,
- geometry message,
- pairwise Geo MLP forward.

Do not necessarily delete historical code if old ablations need it, but GTA mode must bypass it completely.

### Add
Helpers roughly equivalent to:

```text
quat -> R
pose -> g and g^-1
apply_rho_g(...)
apply_rho_gT(...)
apply_rho_ginv(...)
```

`RelationTransformerLayer.forward()` should accept tokenwise `g`/`ginv` (or precomputed pose matrices) and apply GTA around normal attention.

Do not create `[B,H,L,L,32]` pairwise K/V.

---

## 23.5 `tokenhsi/data/cfg/train/rlg/amp_ma_carry.yaml`

Add GTA config and set old geometry off.

For first experiment use:
- A2 `edge_mlp`,
- GTA on,
- old Geo off.

---

## 23.6 Tests

Update/add tests in:

```text
tokenhsi/tests/test_ma_scene_policy.py
```

or a new focused GTA test module.

---

# 24. Required unit tests

Implement all of the following.

## Test A — node dimensions

Verify:

```text
H = 223
O = 30
T = 1
pose = 7
```

and overall scene width is correct for multiple `(M,O)` combinations.

---

## Test B — g convention

Construct simple poses manually.

Example:

```text
H at (0,0,0), heading 0
O at (2,0,0), rotation identity
```

Verify:

```text
g_H g_O^-1
```

contains translation `(2,0,0)` in H frame.

Rotate H heading by +90 degrees and verify relative translation changes to the expected local coordinates.

---

## Test C — Target own-frame geometry

For Target `T_a` using owner heading:

```text
R_Ta = R_Ha_heading
```

verify:

```text
relative rotation H_a <- T_a = identity
relative translation = R_Ha^T (p_Ta - p_Ha)
```

---

## Test D — group representation / factorization

For random valid SE(3) poses:

```text
rho(g_i g_j^-1)
```

must numerically equal:

```text
rho(g_i) @ rho(g_j)^-1
```

within tolerance.

Because rho is a direct sum of identical 4x4 blocks, this should hold exactly up to floating point.

---

## Test E — slow pairwise GTA vs factorized GTA

This is the most important correctness test.

Implement a slow reference for a small tensor:

```math
K_ij = rho(g_i g_j^-1) K_j
V_ij = rho(g_i g_j^-1) V_j
```

and compute attention directly.

Compare against factorized:

```text
rho(g)^T Q
rho(g)^-1 K
rho(g)^-1 V
...
rho(g) output
```

Outputs must match within numerical tolerance.

This test catches row/column and transpose/inverse mistakes.

---

## Test F — identity geometry

If every token has:

```text
g_i = I
```

then GTA attention with relation bias should equal the existing A2 semantic-only attention implementation.

This is an excellent regression test.

---

## Test G — gradients

Verify gradients propagate through:
- QKV projection,
- Transformer projection,
- tokenizers,
- action head.

For the zero-initialized A2 final projection, verify the expected staged behavior:
- initially the semantic bias is exactly zero;
- on the first backward pass the final projection receives a finite nonzero gradient;
- after that projection has updated, gradients reach the A2 embeddings and shared edge MLP.

`g`/rho themselves are algebraic and have no learnable parameters.

---

## Test H — no pairwise GTA feature tensor

Check memory/shape instrumentation to ensure normal GTA forward does not materialize:

```text
[B,H,L,L,head_dim]
```

K/V geometry tensors.

The attention matrix `[B,H,L,L]` is expected.

---

## Test I — entity-count forward

Run at least:

```text
(M,O) = (1,1)
(M,O) = (2,2)
(M,O) = (2,3)
(M,O) = (3,4)
```

Ensure parameter count stays constant and Human outputs have shape:

```text
[B,M,64]
```

---

# 25. Isaac Gym smoke test

After CPU/unit tests:

1. Run a tiny clean-scene GTA training smoke test.
2. Verify one PPO iteration completes.
3. Verify actions are `[N*M, action_dim]`.
4. Verify critic values are `[N*M,1]`.
5. Save/reload checkpoint.
6. Run evaluation forward.
7. Confirm no NaN/Inf in:
   - g,
   - g inverse,
   - transformed Q/K/V,
   - attention logits,
   - policy mean.

Log quaternion norms during initial testing.

---

# 26. Performance benchmark

Compare:

```text
legacy_multirow A1
clean one-X + old pairwise Geo
clean one-X + GTA
```

Use the existing RTX PRO 6000 microbenchmark convention if available.

At minimum benchmark:
- actor forward ms,
- peak GPU memory,
- `(M,O)` = `(1,1), (2,3), (3,4), (4,5)`.

Expected structural behavior:
- GTA adds token-wise transform overhead;
- it should avoid old pairwise `[L,L,64]` Geo encoder/message overhead;
- it should remain one-scene rather than `M` Transformer rows.

Do not assert a speedup before measuring.

---

# 27. Why this stays close to original TokenHSI

The intention is not to redesign the low-level controller observation.

### Human

Original TokenHSI heading-centric Human self state is preserved almost directly:

```text
H = 223D
```

### Object

Original carry observation contains:
- object velocity,
- object angular velocity,
- object relative position/orientation,
- bbox,
- target position,

all expressed from the Human viewpoint.

The GTA version factorizes these into:

```text
object intrinsic state in object frame
+
object pose g_O
+
human pose g_H
+
target pose g_T
```

The underlying physical information is retained while observer-dependent coordinates are no longer precomputed once per agent.

### Original Object velocity detail

Original TokenHSI rotates Object world velocity/angular velocity into the Human heading frame but does not explicitly compute:

```text
v_object - v_human
w_object - w_human
```

Therefore removing the edge experiment's explicit `dv/dw` pairwise differences is not a major departure from original TokenHSI.

### Target

Original Target observation:

```text
R_H^T (p_T - p_H)
```

is encoded geometrically by choosing:
- `g_H` = Human heading frame,
- `g_T` = Target origin + owner Human heading virtual frame.

Thus own Human-to-Target geometry follows the same geometric convention.

---

# 28. Why this follows GTA rather than merely borrowing the name

The implementation must satisfy all of these:

1. Each token has a geometric attribute `g_i` that is a genuine group element.
2. Use `SE(3)` homogeneous transforms.
3. Use a valid representation `rho`.
4. `rho(g)` is a direct sum of 4x4 SE(3) representations.
5. Use the GTA relative transform `rho(g_i g_j^-1)`.
6. Use homomorphism factorization to avoid pairwise transformed K/V.
7. Transform:
   - Q with `rho(g)^T`,
   - K/V with `rho(g)^-1`,
   - attention output back with `rho(g)`.
8. Do not replace GTA with an MLP embedding of pose.
9. Do not simply add pose bias to attention scores/messages.

This is a real GTA adaptation to H/O/T entity tokens.

---

# 29. Important caveats

## 29.1 Do not claim the entire policy is rigorously SE(3)-equivariant

This project has:
- gravity,
- ground,
- Human root height,
- heading-specific observation design,
- PPO/control heads,
- semantic relation bias,

so the paper should describe this as a **GTA-based geometry-aware attention mechanism**, not automatically as a fully SE(3)-equivariant policy.

## 29.2 Target frame is virtual

Target owner-heading orientation is a convention, not a physical target orientation.

Document this explicitly in code comments.

## 29.3 Tokenizer is nonlinear

The raw state factorization demonstrates that entity-local content + pose contains the required physical geometry, but a nonlinear tokenizer means GTA is not literally reconstructing every original raw TokenHSI observer-relative input component one-by-one.

Do not claim exact functional equivalence to the replicated baseline.

---

# 30. Things Codex must NOT do

Do not:

```text
- create one scene row per Human
- reintroduce ego-first scene replication
- create pair-specific K/V with shape [B,H,L,L,d_head]
- concatenate 15D Geo to every pair and run an MLP
- run GTA and old Geo score/message simultaneously
- put global XYZ/yaw back inside H/O/T raw nodes
- use full Human root rotation for g_H
- use heading-only rotation for g_O
- give Target a fake physical orientation unrelated to its owner
- normalize quaternion pose records with RunningMeanStd
- add agent-ID embeddings
- make action heads agent-specific
- change PPO reward/AMP semantics unless required for a bug fix
```

---

# 31. Suggested implementation sequence

Do the work in this order:

### Phase 1 — observation cleanup

Get exact scene layout working:

```text
H 223
O 30
T 1
pose 7
```

Add shape tests.

### Phase 2 — g construction

Construct and unit-test:

```text
g_H
g_O
g_T
g^-1
relative g_i g_j^-1
```

No Transformer change yet.

### Phase 3 — GTA helper

Implement `rho(g)` application using `[... , num_blocks, 4]` reshaping.

Test direct-sum factorization.

### Phase 4 — replace attention

Modify `RelationTransformerLayer` to GTA and validate:
- identity geometry == semantic-only A2 attention,
- slow pairwise == factorized GTA.

### Phase 5 — end-to-end

Run actor/critic unit tests, then Isaac Gym smoke test.

### Phase 6 — benchmark

Measure GPU time/memory.

Do not proceed to performance tuning until correctness tests pass.

---

# 32. Expected shape flow for an example

Example:

```text
M = 3 Humans
O = 4 Objects
Targets = 3
L = 10
B = number of scene samples
```

Raw:

```text
Human nodes  [B,3,223]
Object nodes [B,4,30]
Target nodes [B,3,1]
Pose records [B,10,7]
```

After tokenizer:

```text
X [B,10,64]
```

Per Transformer layer:

```text
Q/K/V        [B,2,10,32]
reshape GTA  [B,2,10,8,4]
g            [B,10,4,4]
g^-1         [B,10,4,4]

Q'           [B,2,10,32]
K'           [B,2,10,32]
V'           [B,2,10,32]

attention    [B,2,10,10]
output       [B,10,64]
```

After final layer:

```text
Human output [B,3,64]
action       [B,3,action_dim]
flatten      [B*3,action_dim]
```

No `B x 3 x 10 x 64` replicated scene tensor should exist in the GTA clean path.

---

# 33. Deliverables

When implementation is complete, provide:

1. list of modified files,
2. concise architecture summary,
3. exact H/O/T/pose dimensions,
4. exact GTA equations used,
5. tests added and their results,
6. smoke-test result,
7. benchmark table,
8. any deviation from this spec and why.

Also update/add a markdown document in the repository describing the GTA clean-scene path so future work does not confuse it with the old pairwise Geo implementation.

---

# 34. Final architecture in one block

```text
Simulator scene
    |
    |-- Human intrinsic self state, heading-local: 223D
    |-- Object intrinsic state, object-local:       30D
    |-- Target intrinsic placeholder:                1D
    |
    |-- one pose per entity:
    |     H: root position + Human heading
    |     O: object position + full rotation
    |     T: target position + owner Human heading
    |
    v
type-specific tokenizers
    |
    v
X = one shared [H | O | T] token set, 64D each
    |
    v
Q/K/V
    |
    v
GTA:
    Q' = rho(g)^T Q
    K' = rho(g)^-1 K
    V' = rho(g)^-1 V

    attention = softmax(Q'K'^T / sqrt(d_head) + A2 semantic edge bias)

    output = rho(g) [attention @ V']
    |
    v
residual + FFN
    |
    v
repeat Transformer layers
    |
    v
select Human tokens only
    |
    v
shared action head
    |
    v
all M Human actions
```

The physical geometry mechanism in this path is **GTA only**.
