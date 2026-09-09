# C13 simultaneous carry viewer

This path runs the coordinator's original two-agent carry setup. It does not
instantiate the sequential-stack task or its phase machine. Both agents execute
their own `root -> box -> goal` plans at the same time, while C13 jointly
selects their paths and speed profiles.

Hand contact is allowed by default (`COORD_ALLOW_HAND_CONTACT=1`) because a
valid low-box pickup otherwise trips the inherited fall detector. Other low
body contacts still terminate the episode.

The masteer actor's learned straight root-to-box pickup geometry is retained by
default (`COORD_PRESERVE_PICKUP_APPROACH=1`). C13 still controls its speed and
controls both agents' complete box-to-goal carry paths. This keeps collision
timing/path coordination without feeding the pickup actor an unseen final
approach direction.

The coordinator changes from pickup to carry only after the box remains lifted
for 12 consecutive action steps (`COORD_GRASP_STABLE_STEPS`). This prevents a
one-frame lift during hand closure from replacing the pickup steer window.

The supplied `Humanoid_*.pth` restores the existing 340-D masteer actor and
normalization state. The task geometry and commands come from
`HumanoidMACoordCarry`; no sequential-stack state is loaded from that file.
The viewer enables evaluation reset mode, so every episode starts from the
`loco_carry` approach state instead of sampling training RSI states such as
`carryWith` or `putDown`.

```bash
MA_GPU=0 bash scripts/masteer/view_coord_carry.sh \
  TokenHSI-masteer/output/sequential_stack/anti_feat_top_s2/Humanoid_00011300.pth
```

`TokenHSI-coord/output/c13.pth` is selected automatically when it is the only
C13 candidate. `COORD_CKPT=/absolute/path.pth` overrides it. The viewer defaults
to the training-compatible `free` layout and `MS_MRAND=4`. Enable the
time-aligned crossing stress scene explicitly:

```bash
MS_SCEN=cross MS_VIEW_TIMED_CROSS=1 MA_GPU=0 \
  bash scripts/masteer/view_coord_carry.sh /abs/Humanoid_*.pth
```

The older `view_coord_sequential_stack.sh` remains the opt-in stack executor;
it intentionally activates only the agent owned by the current stack phase.
