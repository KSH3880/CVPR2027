# ms18-compatible Joint Trajectory Predictor V1

This directory is a pure-PyTorch planner. Importing it does not import Isaac
Gym. The trained planner is loaded separately from the frozen ms18 executor;
the executor checkpoint is neither converted nor saved again.

## Contract

- V1: two agents, Carry only, Free/Cross, one selected joint trajectory.
- Input per stable A/B slot: root xy/yaw/xy velocity, own box xyz/xy velocity,
  goal xy, and observable approach/carry phase. There is no semantic role,
  scenario id, GT trajectory, or joint state in the input schema.
- Coordinates use the root midpoint and A0 yaw as one shared local frame.
- Output: 33 points per agent. `P0=root`, `P16=box`, and `P32=goal` are hard
  anchors. The model only predicts the 30 internal residuals.
- Speed output: four spatial segments and four classes
  `{0.375, 0.75, 1.125, 1.5} m/s`. Stop is intentionally out of V1.
- Runtime resamples the coarse path through `steer_path.resample()` to the
  inherited `[2,320,2]`, `DS=0.1m` buffer. Every action step still emits six
  root-local xy points, so ms18 continues to receive the same 12-D steer token.

`load_checkpoint()` checks schema version, model config, normalizer, agent and
waypoint counts, `DS/V`, six steer points, 1.6 s horizon, all four speeds,
candidate `K=1`, and training dataset hash. Any mismatch raises before rollout.

## 1. Generate the offline teacher data

Run from `TokenHSI-masteer/`:

```bash
python -m trajectory_predictor.generate_dataset --split train
python -m trajectory_predictor.generate_dataset --split val
python -m trajectory_predictor.generate_dataset --split test
```

The defaults are 200k/20k/20k scene seeds and write below
`../runs/trajectory_predictor/data/`. Use `--count` for a smoke run.
Unrelated root/box/goal samples start at least 1m apart. Synthetic carry-phase
states then move each root next to its own lifted box, because that first leg is
physically complete; cross-agent roots and boxes remain separated.

Each state gets a straight candidate and `gen_full_v2()` candidates from fixed
seeds 0 through 7. The batched oracle rejects endpoint/buffer/35-degree paths,
time-aligns both roots, requires 1m clearance, slows A1's conflict quarter
through 1.125/0.75/0.375 m/s before trying A0, then ranks valid pairs by
makespan, length, and curvature. Unsolved scenes retain the maximum-clearance
diagnostic label with `oracle_valid=false` and are excluded by training.

## 2. Pretrain and evaluate

```bash
python -m trajectory_predictor.train \
  --train ../runs/trajectory_predictor/data/joint_v1_train.pt \
  --val ../runs/trajectory_predictor/data/joint_v1_val.pt \
  --output-dir ../runs/trajectory_predictor/checkpoints \
  --save-every 5

# Run this as a separate CPU process while training continues on the GPU.
CUDA_VISIBLE_DEVICES="" python -m trajectory_predictor.watch_videos \
  --checkpoint-dir ../runs/trajectory_predictor/checkpoints \
  --data ../runs/trajectory_predictor/data/joint_v1_val.pt \
  --output-dir ../runs/trajectory_predictor/videos \
  --final-epoch 100

python -m trajectory_predictor.eval_openloop \
  --checkpoint ../runs/trajectory_predictor/checkpoints/joint_v1_best.pth \
  --data ../runs/trajectory_predictor/data/joint_v1_test.pt
```

The fixed loss is waypoint Huber `1.0`, speed CE `0.5`, smoothness `0.1`, and
time-aligned collision loss `0.5`. Best selection is lexicographic: higher
collision-free rate first, then lower ADE. The open-loop command exits 0 only
for endpoint error <=1cm, speed accuracy >=95%, path feasibility >=99%, and
1m-clearance rate >=99% on oracle-valid samples.
The first pretrain starts from random weights and uses no external backbone.

Training is also Isaac-Gym-free. Every epoch is appended to
`joint_v1_metrics.jsonl`; every five epochs the default run atomically writes a
numbered checkpoint. The independent CPU watcher discovers each completed PTH
and writes the matching MP4:

```text
checkpoints/joint_v1_e0005.pth   videos/joint_v1_e0005.mp4
checkpoints/joint_v1_e0010.pth   videos/joint_v1_e0010.mp4
...
```

Each MP4 uses the same first four oracle-valid validation scenes. Dashed lines
are oracle paths, solid lines are predictions, and the two colored dots move
with the predicted four-segment speed commands. The title includes predicted
clearance and speeds. This visualization loads the numbered PTH on CPU and
does not start Isaac Gym or consume a second GPU context. Since it is a
separate process, MP4 encoding never blocks the next training epoch. Use
`--save-every N` to change the checkpoint interval and the watcher's `--scenes`,
`--fps`, and `--max-frames` to control videos.

A previously saved checkpoint can be rendered again independently:

```bash
python -m trajectory_predictor.visualize_checkpoint \
  --checkpoint ../runs/trajectory_predictor/checkpoints/joint_v1_e0020.pth \
  --data ../runs/trajectory_predictor/data/joint_v1_val.pt \
  --output ../runs/trajectory_predictor/videos/joint_v1_e0020.mp4
```

The repository-level launchers use the `tokenhsi` environment and keep GPU
training and CPU video rendering in separate processes:

```bash
nohup bash scripts/trajectory_predictor/train_joint_v1.sh \
  > runs/trajectory_predictor/logs/joint_v1_train.log 2>&1 &
nohup bash scripts/trajectory_predictor/watch_joint_v1.sh \
  > runs/trajectory_predictor/logs/joint_v1_video_watch.log 2>&1 &
```

The training launcher generates any missing deterministic train/val/test split
before pretraining. `TP_GPU`, `TP_EPOCHS`, `TP_SAVE_EVERY`, `TP_BATCH_SIZE`, and
`TP_WORKERS` can override its defaults.

## 3. Run in front of frozen ms18

The positional argument remains the ms18 executor tag or PTH:

```bash
MS_TASK=HumanoidMAPlannerCarry \
MS_PLANNER_CKPT=runs/trajectory_predictor/checkpoints/joint_v1_best.pth \
MS_REPLAN_STEPS=6 MS_VIZ=7 ENVS=1 MS_CAM=top \
bash scripts/masteer/view_local.sh ms18_maskteam_origscale_c06_s0
```

The wrapper replans from refreshed simulator state on reset, first lift, and
every six action steps. Invalid first output uses the analytic straight
`root->box->goal` fallback; later invalid outputs retain the prior valid joint
plan. At exit, `MS_PLANNER_SUMMARY` reports replan, invalid, and fallback counts.

Diagnostic providers use the identical executor:

- `MS_PLANNER_PROVIDER=gt`: inherited ms18 GT provider, with no planner PTH.
- `MS_PLANNER_PROVIDER=oracle`: CPU candidate oracle from the current state.
- `MS_PLANNER_PROVIDER=analytic`: straight-path fallback only.
- default `learned`: strict `MS_PLANNER_CKPT` load.

For queued evaluation, pass these after the tag sidecar through
`MS_EVAL_OVERRIDE`, for example:

```bash
MS_EVAL_SUFFIX=planner \
MS_EVAL_OVERRIDE="MS_TASK=HumanoidMAPlannerCarry MS_PLANNER_PROVIDER=learned MS_PLANNER_CKPT=/absolute/path/joint_v1_best.pth MS_REPLAN_STEPS=6" \
bash scripts/masteer/eval_one.sh ms18_maskteam_origscale_c06_s0 0 512
```

Acceptance order is GT ms18, oracle+ms18, then learned+ms18. Compare `carry` and
`place` within 5 percentage points at each bridge, require `lat_root<=0.25m`,
and planner invalid rate below 1%. Report Free and Cross separately; Cross also
uses existing `colEp`, `encd`, and makespan/timing fields.

## 4. Two DAgger rounds

Collect actually visited replan states without feeding labels to the policy:

```bash
MS_TASK=HumanoidMAPlannerCarry \
MS_PLANNER_CKPT=runs/trajectory_predictor/checkpoints/joint_v1_best.pth \
MS_PLANNER_DAGGER_OUT=runs/trajectory_predictor/dagger/r1_states.pt \
MS_REPLAN_STEPS=6 ENVS=1 MS_CAM=top \
bash scripts/masteer/view_local.sh ms18_maskteam_origscale_c06_s0
```

Relabel and fine-tune while keeping whole episodes in one split:

```bash
python -m trajectory_predictor.relabel_dagger \
  --input ../runs/trajectory_predictor/dagger/r1_states.pt --round 1

python -m trajectory_predictor.train \
  --train ../runs/trajectory_predictor/data/joint_v1_train.pt \
          ../runs/trajectory_predictor/dagger/dagger_r1_train.pt \
  --val ../runs/trajectory_predictor/dagger/dagger_r1_val.pt \
  --init-checkpoint ../runs/trajectory_predictor/checkpoints/joint_v1_best.pth \
  --name joint_v1_dagger_r1
```

Repeat with `--round 2` and the round-1 best PTH. Stop after round 2 or earlier
if the closed-loop metrics no longer improve. `MS_PLANNER_DAGGER_MAX` bounds the
number of retained states (default one million).

## Verification

```bash
python -m unittest discover -s trajectory_predictor/tests -v
```

These tests cover pure import, rigid-coordinate round trips, hard anchors,
resample shape, speed classes, oracle clearance, dataset I/O, fail-closed
checkpoint loading, and bit-exact output before/after checkpoint save/load.
