# v0 3000: box-position input audit

## Scope

Checkpoint: `output/ma_carry_relation_v0/CarryRelationV0_10-14-53-33/nn/CarryRelationV0_00003000.pth`.

Separate headless evaluation, seed 42, 32 environments, 2 agents, 3 boxes,
deterministic actions, at most 600 steps per cohort. Cohorts: mixed RSI
(`loco,pickUp,carryWith,putDown`, probabilities `.5,.1,.3,.1`), loco-only,
pickUp-only. Behavior counts include each scene's first episode only.
This is not a recording of the user's exact one-environment viewer trajectory.
Training processes, checkpoints, and production code were not changed by this audit.

Temporary audit script: `/tmp/carry_box_input_audit.py`.
Complete setup/results log: `/tmp/box_input_audit_3000.log`.

## Position pipeline

Resolved physical box handles independently to simulator actor and rigid-body
indices. Actor IDs matched the task's `_box_actor_ids`. Logical box slots for
the first two objects matched the two agents' physical assignments throughout.
Captured actual actor-encoder input after observation normalization.

Maximum absolute discrepancies across the three cohorts:

| Comparison | Maximum error |
| --- | ---: |
| Box actor-root position vs box rigid-body position | 0 m |
| Raw observation box position vs physical box minus environment origin | 0 m |
| Actual actor-input box position vs physical box minus environment origin | 0 m |
| Actual actor-input goal position vs task goal minus environment origin | 0 m |
| Pose block before vs after normalization | 0 |
| Raw object bounding-box points vs assigned object's size-specific points | 0 m |
| GTA object-transform translation vs expected scaled position | 0 |
| Recomputed relation phi vs runtime phi | 0 |
| Reset reference-motion box XY vs initialized assigned box XY | 0.0000076294 m |

These checks found no stale-position, assignment-order, or pose-normalization
mismatch in the tested path. They do not prove all geometry/network logic is
correct or that the policy has learned to use positions appropriately.
Goal checks establish agreement with the task's goal tensor, not an independent
visual inspection of the viewer marker.

## Empty-space low-posture observations

Conservative numerical proxy, NOT a classifier of a pickup animation:
after 2 seconds, root height < 0.75 m, both hands below 0.55 m,
and the hand midpoint more than 1 m from every box's enclosing sphere.
The last condition is a lower bound on distance to each box surface.
Crouching/falling can also satisfy this proxy.

| Initial cohort | Agents with at least one qualifying frame | Qualifying agent-frames |
| --- | ---: | ---: |
| Mixed | 38 / 64 | 1269 |
| Loco | 44 / 64 | 1713 |
| PickUp | 7 / 64 | 114 |

Example: loco cohort, environment 23, agent 1, 3.333 s:

- Root height: 0.74963 m.
- Hand heights: 0.41189 m and 0.39459 m.
- Hand-midpoint distance to every box surface: at least 2.21352 m.
- Holding phi: 2.36144e-14.
- Weighted holding delta: 4.97527e-15.
- Weighted holding velocity / at velocity / success bonus: 0 / 0 / 0.
- Power and total task reward: -0.00386076.
- AMP reward before final task/AMP mixing: 2.12411.

The reward did not treat this empty-space posture as successful holding.
AMP observes humanoid kinematics, not box/goal coordinates, so it can award
motion-style reward away from objects. This example does not establish AMP as
the sole cause, nor does raw reward magnitude establish gradient dominance.

## Input sensitivity

At steps 30 and 120, shifted assigned boxes by +0.5 m in X in a temporary
observation, recomputed relation phi, measured deterministic action change,
then restored state/observation before stepping physics. RMS action differences:

| Cohort | Step 30 | Step 120 |
| --- | ---: | ---: |
| Mixed | 0.0464420 | 0.0458838 |
| Loco | 0.00141420 | 0.00269648 |
| PickUp | 0.0499480 | 0.0513650 |

The model responds to this box-related input intervention; it is not completely
input-independent. Phi also changes, so this is not a positions-only attribution.
Different cohort responses do not by themselves establish correct/incorrect
directional control or a calibrated sensitivity threshold.

## Interpretation

Current evidence favors an insufficiently learned connection between object
location and task-appropriate behavior over a position-copying/indexing bug.
The user's exact pickup-looking pose still needs visual confirmation; the
headless proxy only establishes low postures away from all boxes.
The separate state2_signed progress experiment remains a reasonable test of
navigation reward shaping, not a proven fix for this symptom. No AMP or holding
reward change was made as part of this diagnosis.
