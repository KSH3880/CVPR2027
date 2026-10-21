# Stage1 native4 zero-shot diagnostic

Attention PNG capture additionally supports mixed OPEN/PUSH via the opt-in
`stage2_jointpush/scripts/stage1_native4_attention.sh` wrapper. It uses agent-specific
door/box/goal bindings and applies the original two-agent reward per pair. This does
not change the PUSH-only fall-ablation defaults described below. PNGs contain four
actor layers/two heads, with canonical 16-token labels; raw QK/bias/attention and
physical bindings are saved alongside them.

Original Stage1 weights, **4 humans / 4 real doors / 4 real boxes**, `Start` initialization, PUSH-only evaluation. No learning, Stage2 weight, or coordination-head expansion. This is the portable version of the locally verified native4 setting.

From the repository root:

```bash
# Headless: 64 independent scenes, deterministic actions, seed123, max300steps
STAGE1_CHECKPOINT=/absolute/path/OpenPushPush180ContactAtWarmstart_00029000.pth \
TOKENHSI_GPU=0 bash stage2_jointpush/scripts/stage1_native4_zeroshot.sh eval

# Native Isaac Gym viewer; requires a working display on that host
STAGE1_CHECKPOINT=/absolute/path/OpenPushPush180ContactAtWarmstart_00029000.pth \
TOKENHSI_GPU=0 bash stage2_jointpush/scripts/stage1_native4_zeroshot.sh view
```

`STAGE1_CHECKPOINT` is optional when the shared resolver finds `paths.local.yaml` or the usual local/server Stage1 location. Explicit invalid paths fail. `TOKENHSI_CONDA_ENV` defaults through the shared runtime setup; override it if the server environment has a different name.

Optional: `OUTPUT_PATH`, `NUM_ENVS`, `PROBE_STEPS` (eval horizon, default300), `SEED` (default123), `PARTITION_PUSH_MOTION_FILE` (installed motion manifest). Outputs default to a new `runs/jointpush/stage1_native4_<mode>_<timestamp>/` directory. `result.json` contains every scene's first fall or horizon, and `manifest.json` records checkpoint path/hash, source epoch and intervention. Model tensors must remain unchanged during evaluation. Headless evaluation supports other scene counts, including a short smoke check:

```bash
NUM_ENVS=2 PROBE_STEPS=32 bash stage2_jointpush/scripts/stage1_native4_zeroshot.sh eval
```

## Files and prerequisites

- `setup.py` materializes saved Stage1 env/train settings on the current machine; source checkpoint remains unchanged.
- `native4_task.py` is a frozen diagnostic adaptation of `HumanoidMAOpenPush`, with four physical door DOFs and eight object slots. Production tasks are not patched on disk.
- `launch.py` installs diagnostic-only count/graph/PUSH-reward adaptations and loads the original Stage1 model strictly. View mode uses the regular player loop.
- The wrapper uses the existing Isaac Gym/conda runtime and Blackwell fusion compatibility launcher.
- Checkpoint `.pth`, Isaac Gym, conda dependencies and motion datasets are **not bundled in Git**. The default converted motion manifest and its referenced loco/Push180 datasets must be installed, or supply `PARTITION_PUSH_MOTION_FILE`.

## Preserved settings and intervention

`Start` initialization, original loco reset, real door geometry/hinge/closing torque, 1.3m cube /20kg boxes, contact/credit/eligibility and original source actor/normalizer weights are retained. The original two-box random scene sampler is duplicated at y=-10/+10m; four door lanes are y=-15/-5/+5/+15m. Every box has a distinct owner, with minimum6m box separation. The source checkpoint must match the recorded native setting (`Start`, lane spacing10m).

The source2-person task has a fixed2-person OPEN reward branch. This diagnostic is **PUSH-only**: it extends the existing `push_reward` and credit/eligibility update to4 agents and does not claim OPEN-task support. The converted motion manifest is an explicit override; inactive evaluation interaction-RSI bank loading is skipped, while the remaining source contract is checked. Real doors still simulate, collide and close.

Previously measured local native4 result:53/64 scenes had a fall within300steps (native2:0/64). Scene failure means at least one human falls; surviving peers are censored. This is an initial-stability diagnostic, not a complete task-success measurement. Workspace extent also changes, so it is not a pure token-count causal experiment. Server trajectories may differ with hardware/software versions; compare aggregate rates rather than expecting identical frames.
