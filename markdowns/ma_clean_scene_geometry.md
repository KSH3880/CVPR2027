# Clean scene multi-agent policy

> Historical edge-geometry ablation document. The active clean-scene configuration now
> uses GTA and is documented in [`ma_clean_scene_gta.md`](ma_clean_scene_gta.md). The
> 227/39/3 nodes and pairwise 15-D Geo below are not executed when GTA is enabled.

## Modes

- `policyObsMode: clean_scene` builds and stores one scene observation per simulator
  environment. The actor and critic each encode that scene once and read all humanoid
  tokens.
- `policyObsMode: legacy_multirow` preserves the original ego-first A1 implementation
  and its checkpoint/input layout.

The clean ablations live under `network.transformer`:

```yaml
relation_bias: true
relation_bias_mode: edge_mlp  # A2 default; lookup restores clean A1
geometry:
  enable: true
  use_score: true
  use_message: true
  mode: full15  # position3 | pose9 | full15; raw width always remains 15
```

`legacy_multirow` is always forced to the exact old A1 baseline. `clean_scene` defaults
to A2 and can select A1 with `relation_bias_mode: lookup`. Disabling geometry produces a
semantic-relation-only clean-node ablation, not an exact reproduction of the old
baseline, because owner-relative features have intentionally moved out of the nodes.

## Semantic relation edge (A2)

The clean default replaces only A1's scalar semantic lookup with the `edge_a2` typed
edge encoder:

```text
source entity type embedding 16 ─┐
relation type embedding      32 ─┼─ concat 64 -> shared MLP -> edge embedding 64
target entity type embedding 16 ─┘

edge embedding 64 -> layer/head projection -> scalar semantic bias
```

The relation taxonomy (`NONE`, `SELF`, `TEAMMATE`, `OWN_OBJECT`, `OWN_GOAL`, and
`OBJECT_GOAL`) is unchanged. A2 can distinguish the same relation ID across directed
H/O/T type combinations, while A1 has only one scalar per `(layer, head, relation ID)`.
The A2 edge has no batch dimension and its parameters are independent of M/O. Its final
projection is zero-initialized, so the first forward has zero semantic bias; the first
optimizer step trains that projection and later steps propagate gradients into the
edge embeddings and MLP. With four layers and two heads, A2 adds 9,072 parameters per
encoder (18,144 across the separate actor/critic) relative to A1, independent of M/O.

Semantic and physical edges remain independent:

```text
attention score = QK + A2 semantic bias + Geo score
attention value = V_j + Geo message_ij
```

A2 receives discrete types/relations and needs no RMS or physical scaling. Geo receives
continuous 15-D relative state and retains its fixed scaling below. `relation_bias:
false` disables the semantic branch without disabling Geo.

## Feature split

The legacy policy used 230-D humanoid, 39-D object, and 6-D target tokens. In
`clean_scene` the node/edge split is "own state + own shared scene pose" versus
"pairwise relative geometry":

- Human, 227-D: 222-D TokenHSI body state in the human's heading frame (the original
  223-D state without its root-height scalar), followed by env-local
  `(x/arena_scale, y/arena_scale, z, cos(yaw), sin(yaw))`. For 15 bodies the 222-D
  prefix is relative body positions `14*3`, rotations `15*6`, linear velocities
  `15*3`, and angular velocities `15*3`.
- Object, 39-D: object-frame linear velocity `3`, object-frame angular velocity `3`,
  eight object-local bbox corners `24`, fixed-scaled env-local position `3`, and the
  full world/env orientation in continuous tangent/normal representation `6`.
- Target, 3-D: fixed-scaled env-local XYZ position. Carry targets are point targets and
  have no orientation node feature. Both legacy owner-relative target vectors remain
  absent from the node.

For every entity, fixed-scaled position means subtracting the simulator environment
origin, dividing X/Y by `arena_scale`, and retaining Z in env-local metres. Local/self
prefixes are normalized with type-specific running statistics (Human `222`, Object
`30`). Shared scene-pose suffixes and the entire Target node bypass running-statistic
normalization, preserving one numerical coordinate convention across entity types.

The rollout appends one compact 13-D kinematic record `(position, quaternion, linear
velocity, angular velocity)` per entity. This is not tokenized as node content. It is
used algebraically on forward to construct geometry, avoiding storage of an `L x L`
tensor at every rollout step.

The 13-D geometry-construction records remain env-local but unscaled and bypass node
normalization. Humans and objects store their full physical quaternion and world-frame
linear/angular velocity. A point Target stores identity quaternion and zero velocities;
it never borrows its owner's heading.

## Pairwise geometry

For H/O sources, relative position and velocity use only the source heading:

```text
dp_ij = heading(q_i)^-1 * (p_j - p_i)
dv_ij = heading(q_i)^-1 * (v_j - v_i)
dw_ij = heading(q_i)^-1 * (w_j - w_i)
dR_ij = q_i^-1 * q_j -> tangent/normal 6-D
```

Using heading rather than full orientation for `dp/dv/dw` prevents a tilted object from
tilting forward/left/height coordinates. `dR` deliberately uses full orientations.
Targets have no heading, so T-source positions are plain differences in shared env-local
axes. Any pair containing a Target keeps only `dp`; its remaining 12 components are
exact zeros. Every self edge is all-zero so `REL_SELF`, rather than a rotation constant,
defines self semantics.

| Directed pair | position 3 | rotation 6 | linear velocity 3 | angular velocity 3 |
|---|---|---|---|---|
| H -> H | actual (source heading) | actual (full) | actual | actual |
| H -> O | actual (source heading) | actual (full) | actual | actual |
| H -> T | actual (source heading) | zero | zero | zero |
| O -> H | actual (source heading) | actual (full) | actual | actual |
| O -> O | actual (source heading) | actual (full) | actual | actual |
| O -> T | actual (source heading) | zero | zero | zero |
| T -> H | actual (scene axes) | zero | zero | zero |
| T -> O | actual (scene axes) | zero | zero | zero |
| T -> T | actual (scene axes) | zero | zero | zero |

Diagonal entries in every row of the table are the all-zero self exception. Geometry is
dense and independent of the semantic relation matrix: a `REL_NONE` pair still receives
its physical geometry, while A2 separately embeds its entity types and `NONE` relation.

### Geometry scaling and ablation

No running-statistic normalization is applied to raw kinematics or Geo. Fixed physical
scales preserve zero padding and remain independent of the H/O/T pair mixture:

```text
dp: [1/arena_scale, 1/arena_scale, 1]  # scaled XY, Z in metres
dR: 1                                  # already bounded
dv: 0.25
dw: 0.25
```

`position_scale: null` obtains `arena_scale` from the task at model construction, so an
`envSpacing` change cannot silently leave a hard-coded XY scale behind. Ablations mask
components without changing the shared `15 -> 64` encoder: `position3` zeros dimensions
3:15, `pose9` zeros 9:15, and `full15` keeps every valid component.

## Shape flow

For `N` scenes, `M` humanoids, `O` objects, `L = 2M + O`, `D = 64`, `H = 2`, and
`d_head = 32`:

```text
simulator body/object/target state
 -> human nodes       [N,M,227]
 -> object nodes      [N,O,39]
 -> target nodes      [N,M,3]
 -> entity kinematics [N,L,13]
 -> stored scene obs  [N, M*227 + O*39 + M*3 + L*13]
                    = [N, 256*M + 52*O]

tokenizers + type embeddings
 -> X                 [N,L,64]

directed source-frame construction
 -> G                 [N,L,L,15]
    = dp(3) + dR tangent/normal(6) + dv(3) + dw(3)
 -> shared pair MLP   [N,L,L,64]
 -> geometry score    [N,2,L,L]
 -> geometry message  [N,2,L,L,32]

each transformer layer
 -> Q/K/V             [N,2,L,32]
 -> QK + A2 semantic bias + geometry score
 -> softmax-weighted (V_j + geometry message_ij)
 -> updated H/O/T     [N,L,64]

readout
 -> human tokens      [N,M,64]
 -> actor head        [N,M,action_dim] -> [N*M,action_dim]
 -> critic head       [N,M,1]          -> [N*M,1]
```

The actor and critic have separate parameters, matching the existing `separate: True`
policy, but each performs only one forward per scene. H/O/T all remain queries and are
updated at every layer.

## PPO storage

The clean path replaces only the policy observation buffers with `(T,N,scene_obs)`.
Actions, values, log-probabilities, rewards, dones, and AMP observations stay per agent.
Before minibatching, they are grouped as `(N*T,M,...)`. A scene minibatch is passed
through the policy once; agent axes are flattened only inside the PPO loss. No scene
observation is duplicated in rollout storage or model compute.

## Validation snapshot

CPU tests cover the 227/39/3 node sizes and full observation width, selective node
normalization, env-origin translation invariance, directed-geometry SE(2) invariance,
heading-only H/O position frames, full relative orientation, exact Target/self masks,
`REL_NONE` geometry, 3/9/15 ablations, score/message gradients, constant parameter
count, A2 embedding/projection gradients, A1 fallback, legacy layout, and forwards for
`(M,O) = (1,1), (2,2), (3,4)`. The original
edge-geometry implementation also completed one-iteration clean and legacy Isaac Gym
smoke runs; those runs predate the node and Geo revisions described above.

### Checks for the next real rollout

- Log p95/p99 absolute values for scaled `dp`, `dv`, and `dw`. Values consistently much
  larger than roughly 2-3 indicate that the fixed velocity scales need retuning.
- Verify H/O quaternion norms stay near one and Target quaternions remain exactly
  identity. Bad quaternion norms invalidate both heading extraction and `dR`.
- Inspect highly tilted objects. Heading is obtained by projecting the object's local X
  axis onto XY, matching TokenHSI; it becomes numerically ambiguous if that axis is near
  vertical.
- Assert Target-pair dimensions 3:15 and every diagonal Geo entry remain exactly zero.
- Keep `clip_observations` unset/infinite for this layout. Clipping the appended raw 13-D
  records before Geo construction would corrupt quaternion and relative-motion inputs.
- Recheck actor/critic Geo gradient magnitudes. Zero or exploding values would indicate
  that the fixed scales or score/message heads need adjustment.
- Expect A2 embedding/MLP gradients to be zero on the very first backward because its
  final projection starts at exact zero. They must become nonzero after that projection
  receives its first update.
- Start a fresh clean-scene checkpoint: revised node tokenizer/RMS shapes and changed Geo
  semantics are intentionally incompatible with earlier clean-scene checkpoints. A1 and
  A2 semantic branches also have different state-dict keys and are not strict-load
  compatible with one another.

An RTX PRO 6000 inference microbenchmark with `N=1024` scenes produced the following
actor-forward measurements (200 timed iterations after 50 warm-up iterations):

| M | O | legacy ms | clean+geo ms | legacy peak MiB | clean peak MiB |
|---:|---:|----------:|-------------:|----------------:|---------------:|
| 1 | 1 | 0.809 | 1.293 | 42.2 | 48.3 |
| 2 | 3 | 1.615 | 1.552 | 110.5 | 95.5 |
| 3 | 4 | 2.481 | 2.029 | 210.8 | 145.2 |
| 4 | 5 | 7.351 | 4.492 | 347.9 | 201.8 |

At M=1 the pairwise geometry overhead outweighs the saved duplicate transformer work
(there is no duplicate at M=1). From M=2 in this snapshot, one-pass compute is faster
and uses less peak memory, with the gap widening as M grows. Full training throughput
remains workload- and simulator-dependent.
