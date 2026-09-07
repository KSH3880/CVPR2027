# Clean scene multi-agent policy

## Modes

- `policyObsMode: clean_scene` builds and stores one scene observation per simulator
  environment. The actor and critic each encode that scene once and read all humanoid
  tokens.
- `policyObsMode: legacy_multirow` preserves the original ego-first A1 implementation
  and its checkpoint/input layout.

The clean ablations live under `network.transformer`:

```yaml
relation_bias: true
geometry:
  enable: true
  use_score: true
  use_message: true
```

`legacy_multirow` is the exact old A1 baseline. `clean_scene` with geometry disabled is
a relation-only clean-node ablation, not an exact reproduction of the old baseline,
because owner-relative features have intentionally moved out of the nodes.

## Feature split

The legacy policy used 230-D humanoid, 39-D object, and 6-D target tokens. In
`clean_scene` these become:

- Human, 223-D: the original body self-state in the human's own heading frame. The
  arena `(x,y,cos(yaw),sin(yaw))` feature and the always-zero relative root position are
  removed.
- Object, 30-D: object-frame linear velocity, object-frame angular velocity, and eight
  object-local bbox corners. Owner-relative position/orientation and owner-frame bbox
  corners are removed.
- Target, 1-D: a constant intrinsic feature; the type embedding distinguishes targets.
  Both legacy relative target vectors (`human -> target` and `object -> target`) are
  removed.

The rollout appends one compact 13-D kinematic record `(position, quaternion, linear
velocity, angular velocity)` per entity. This is not tokenized as node content. It is
used algebraically on forward to construct geometry, avoiding storage of an `L x L`
tensor at every rollout step.

Human positions and object/target positions are stored relative to the simulator env
origin. Human entity orientation is its heading quaternion. Object orientation is its
full physical quaternion. A target has no physical orientation, so its owner's human
heading is used as a virtual target frame; its velocities are zero.

## Shape flow

For `N` scenes, `M` humanoids, `O` objects, `L = 2M + O`, `D = 64`, `H = 2`, and
`d_head = 32`:

```text
simulator body/object/target state
 -> human nodes       [N,M,223]
 -> object nodes      [N,O,30]
 -> target nodes      [N,M,1]
 -> entity kinematics [N,L,13]
 -> stored scene obs  [N, M*223 + O*30 + M + L*13]

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
 -> QK + A1 bias + geometry score
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

CPU tests cover directed-geometry SE(2) invariance, score/message ablations, gradients,
constant parameter count, and forwards for `(M,O) = (1,1), (2,2), (3,4)`. An Isaac Gym
smoke run completed one clean PPO iteration for `N=2,M=2,O=3`, saved/restored its
checkpoint, and completed the evaluation loop. The legacy mode also completed the same
one-iteration smoke run.

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
