# Clean-scene GTA multi-agent policy

This is the active `policyObsMode: clean_scene` architecture. The historical 15-D
pairwise Geo encoder remains in code for ablations but is disabled in the GTA config.

## Active configuration

```yaml
transformer:
  relation_bias: true
  relation_bias_mode: edge_mlp
  gta:
    enable: true
    translation_scale: 1.0
    representation: se3_direct_sum
    diagnostics_first_forward: true
  geometry:
    enable: false
```

GTA and `geometry.enable` are mutually exclusive. `legacy_multirow` remains the old A1
lookup baseline and does not execute GTA.

## Observation layout

For `M` Humans, `O` Objects, and `L = 2M + O` tokens:

```text
[M * Human223 | O * Object30 | M * Target1 | L * pose7]
```

The total stored width is:

```text
M*223 + O*30 + M*1 + (2M+O)*7 = 238M + 37O
```

| M | O | L | observation width |
|---:|---:|---:|---:|
| 1 | 1 | 3 | 275 |
| 2 | 2 | 6 | 550 |
| 2 | 3 | 7 | 587 |
| 3 | 4 | 10 | 862 |

Token order is always `[H_0..H_(M-1) | O_0..O_(O-1) | T_0..T_(M-1)]`.
Assigned object slot `a` and Target slot `a` belong to Human `a`; additional Objects
are unassigned distractors.

## Intrinsic nodes and normalization

### Human: 223-D

The complete TokenHSI maximum-coordinate self state is constructed in the Human's own
heading frame:

```text
root height                         1
relative body positions            42
heading-local body rotations       90
heading-local linear velocities    45
heading-local angular velocities   45
total                              223
```

Absolute XY and yaw are not appended. `cleanSceneLocalRootObsPolicy` and
`cleanSceneRootHeightObsPolicy` are both enabled for this path without changing the
legacy flags. One Human RMS(223) is shared across all Human slots.

### Object: 30-D

```text
full-object-local linear velocity    3
full-object-local angular velocity   3
object-local bbox corners           24
total                               30
```

One Object RMS(30) is shared across Object slots. Position and orientation are absent
from the node because they are carried by the GTA pose.

### Target: 1-D

```text
Target = [1.0]
```

The constant is a tokenizer placeholder and bypasses RMS. Target type and ownership
come from the type embedding and A2 semantic edge.

### Normalization boundary

```text
Human223 -> type-wise RMS
Object30 -> type-wise RMS
Target1  -> passthrough
pose7    -> passthrough
```

Pose position and quaternion must not pass through RunningMeanStd. Observation clipping
must remain unset/infinite so it cannot corrupt pose records before GTA construction.

## Pose records

Each token stores an env-local position and local-to-env quaternion in `xyzw` order:

```text
pose7 = [p_world - env_origin, q_local_to_env]
```

| Entity | position | orientation |
|---|---|---|
| Human | root env-local XYZ | Human heading only |
| Object | root env-local XYZ | full physical rotation |
| Target `a` | target env-local XYZ | owner Human `a` heading |

The Target orientation is a virtual coordinate convention, not a physical target
orientation. Quaternions are converted to unit quaternions immediately before matrix
construction; no statistical quaternion normalization is used.

## GTA transform convention

For local-to-env pose `T_i`:

```text
T_i = g_i^-1 = [R_i, p_i; 0, 1]
g_i           = [R_i^T, -R_i^T p_i; 0, 1]
```

Thus `g_i g_j^-1` maps token `j` local coordinates into token `i` local coordinates.
Before constructing the matrices, one uniform scale is applied:

```text
p_i <- translation_scale * p_i
```

The default `translation_scale=1.0` uses metres. It is fixed scaling, not RMS. If
rollout diagnostics show saturated attention, tune one scalar for XYZ together; do not
use different XY and Z scales inside the SE(3) matrix.

With `d_model=64`, two heads, and `d_head=32`:

```text
rho(g) = g direct-summed 8 times
```

The implementation reshapes each head to `[8,4]` and applies the 4x4 matrix without
materializing a 32x32 matrix.

## Attention with A2

For every Transformer layer:

```text
Q' = rho(g)^T Q
K' = rho(g)^-1 K
V' = rho(g)^-1 V

A = softmax(Q'K'^T / sqrt(d_head) + A2_semantic_bias)
O = rho(g) (A @ V')
```

A2 receives only `(source entity type, relation type, target entity type)` and projects
its 64-D semantic edge to one scalar per layer/head. GTA receives only pose. There is
no old Geo score, Geo message, pairwise velocity, or pairwise `[L,L,d_head]` K/V tensor.

The A2 final projection starts at zero. Its projection learns on the first optimizer
step; gradients reach the A2 embeddings/MLP after that projection has moved.

## Shape flow

```text
nodes -> type tokenizers + type embeddings -> X [B,L,64]
pose7 -> g/g^-1                         -> [B,L,4,4]
Q/K/V                                  -> [B,2,L,32]
GTA block view                         -> [B,2,L,8,4]
attention                              -> [B,2,L,L]
updated tokens                         -> [B,L,64]
Human readout                          -> [B,M,64]
actor / critic                         -> [B*M,action_dim] / [B*M,1]
```

Actor and critic encoders have separate weights. Parameters do not depend on `M` or
`O`; changing entity counts changes only token/relation/pose tensor shapes.

## Validation requirements

- Analytic `g` and `g^-1` multiply to identity.
- `g_i g_j^-1` matches manually constructed local-frame examples.
- Factorized GTA matches a slow explicit pairwise reference.
- Identity poses match semantic-only A2 attention.
- GTA and old Geo cannot be enabled simultaneously.
- A2, QKV, tokenizer, actor, and critic gradients are finite.
- One PPO iteration, checkpoint save/reload, and evaluation forward complete without
  NaN/Inf.
- Log quaternion norms, transformed Q/K/V norms, attention-logit p95/p99, and attention
  entropy during the first real rollout.

With `diagnostics_first_forward: true`, the actor and critic each emit these statistics
once, on their first forward. The flag has no effect after that call and does not add
parameters or checkpoint state.

The revised node/tokenizer/RMS shapes and GTA semantics require a new clean-scene
checkpoint.

## Verified smoke and microbenchmark

The clean GTA path completed one headless Isaac Gym PPO iteration with `N=4`, `M=2`,
and `O=3` on an RTX PRO 6000. Runtime construction reported 587 observations, seven
tokens, A2 `edge_mlp`, GTA direct-sum, and no old Geo branch. Actor/critic quaternion
norms were exactly 1.0; attention entropy stayed between 1.67 and 1.95 (the uniform
seven-token maximum is about 1.946). The iteration completed forward/backward, saved a
checkpoint, and strict model/RMS reload plus evaluation forward produced only finite
outputs.

The following actor-forward benchmark uses 1,024 scenes, 50 warm-up iterations, and
200 timed iterations on the same GPU. Peak memory is incremental CUDA allocation above
the post-warm-up baseline.

| M | O | legacy A1 ms | clean old-Geo ms | clean GTA ms | legacy MiB | old-Geo MiB | GTA MiB |
|---:|---:|-------------:|-----------------:|-------------:|-----------:|------------:|--------:|
| 1 | 1 | 0.926 | 1.585 | 1.361 | 17.3 | 22.4 | 19.9 |
| 2 | 3 | 1.784 | 1.942 | 1.855 | 81.3 | 69.1 | 46.8 |
| 3 | 4 | 3.218 | 2.419 | 2.253 | 174.8 | 118.6 | 68.0 |
| 4 | 5 | 5.502 | 3.095 | 2.794 | 305.3 | 172.4 | 88.0 |

GTA is not faster than legacy at `M=1`, where there is no duplicated per-Human scene
work to remove. It becomes faster and substantially smaller as entity count grows; it
also measured faster and smaller than the retained old pairwise-Geo ablation at every
tested size. Re-run `tokenhsi/tests/benchmark_ma_scene_policy.py` after major CUDA,
PyTorch, model-width, or hardware changes rather than treating these numbers as fixed.
