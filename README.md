# TokenHSI-MultiAgent

Multi-agent extension of [**TokenHSI**](https://github.com/liangpan99/TokenHSI) (CVPR 2025 Oral) —
Pan et al., *Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization*.

The original `carry` task places **one humanoid and one box** in a scene. This fork extends it to
**M humanoids, O boxes (O ≥ M) and M goals**: each agent is randomly assigned a distinct box, and
the remaining `O − M` boxes stay unassigned as distractors. Entities (humanoid / object / goal) are
split into per-type tokenizers, and the assignment relation is injected as a non-learned relation
matrix that becomes a learnable attention bias.

> Everything from the upstream repo — datasets, checkpoints, single-task training paths — is left
> untouched. Motion data is reused read-only and the multi-agent runs write to their own output paths.

## What is added

| Area | Files |
|---|---|
| Environment | `tokenhsi/env/tasks/multi_agent/` — `humanoid_ma.py`, `humanoid_ma_carry.py`, `vec_task_wrapper_ma.py` |
| Learning | `tokenhsi/learning/multi_agent/` — `amp_network_builder_ma.py`, `ma_agent.py`, `ma_players.py` |
| Configs | `tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml`, `tokenhsi/data/cfg/train/rlg/amp_ma_carry{,_watch}.yaml` |
| Scripts | `tokenhsi/scripts/multi_agent/` — train / test / watch / remote-GUI helpers |

Upstream files touched (registration and multi-agent plumbing only): `tokenhsi/run.py`,
`tokenhsi/utils/config.py`, `tokenhsi/utils/parse_task.py`, `tokenhsi/learning/amp_players.py`.

New algorithm keys registered in `run.py`: algo `ma`, player `ma`, network `amp_multi_agent`.

## Documentation

Start with the short current references (mostly in Korean):

- [`AGENTS.md`](AGENTS.md) — Codex reading order and repository conventions
- [`changelog.md`](changelog.md) — recent changes and validation results
- [`structure.md`](markdowns/structure.md) — code map and task entry points
- [`config.md`](markdowns/config.md) — all experiment configs, training, evaluation, and remote VNC

Read detailed references as needed:

- [`ma_clean_scene_gta.md`](markdowns/ma_clean_scene_gta.md) — current GTA policy architecture
- [`relation_diagnostics.md`](tokenhsi/docs/relation_diagnostics.md) — placement metrics and traces
- [`USAGE.md`](markdowns/USAGE.md), [`PORTING_GUIDE.md`](markdowns/PORTING_GUIDE.md) — installation and machine setup

## Setup

This checkout uses the existing `tokenhsi` environment on the RTX PRO 6000 server.
Training uses **2048 environments**. See [`USAGE.md`](markdowns/USAGE.md) for the
installed environment and data paths.

```bash
conda activate tokenhsi
nvidia-smi
```

Assets that are **not** in this repository and must be fetched separately:

- [SMPL body models](https://smpl.is.tue.mpg.de/) → `body_models/smpl/`
- Pre-processed motion & object data → [Hugging Face](https://huggingface.co/datasets/lianganimation/TokenHSI),
  extracted into `tokenhsi/data/dataset_*/`
- Pre-trained checkpoints → `output/` (see the upstream README)

See [`markdowns/USAGE.md`](markdowns/USAGE.md) for a verified end-to-end setup, including remote GUI
rendering over noVNC.

## Usage

The helper scripts contain no absolute paths: the repo root is derived from the script's own
location, and everything else is auto-detected with an environment-variable override.

| Variable | Default | Meaning |
|---|---|---|
| `TOKENHSI_CONDA_ENV` | the env you already activated, else `tokenhsi` | conda env to activate |
| `CONDA_BASE` | `conda info --base`, then the usual install prefixes | conda installation prefix |
| `TOKENHSI_GPU` | `0` | physical GPU for both CUDA and viewer rendering |
| `X11VNC` / `VNC_DIR` | `x11vnc` on `PATH` | x11vnc binary, or the prefix of a user-local install |
| `NOVNC_DIR` | `/usr/share/novnc`, `~/opt/novnc`, … | directory containing `vnc.html` |
| `WEBSOCKIFY` | `PATH` → conda env → noVNC bundle | websockify executable |
| `PORT` | `6080` | noVNC web port |

So a machine whose env is named differently just needs, e.g.
`TOKENHSI_CONDA_ENV=my-env sh tokenhsi/scripts/multi_agent/ma_carry_train.sh 2 2048 3`.

```bash
# Train: <num_agents> <num_envs> <num_objects>
sh tokenhsi/scripts/multi_agent/ma_carry_train.sh 2 2048 3

# Test / evaluate a checkpoint
sh tokenhsi/scripts/multi_agent/ma_carry_test.sh output/ma_carry/nn/xxx.pth 2 16 3

# View a saved checkpoint through noVNC on GPU 6
TOKENHSI_GPU=6 VNC_DIR="$HOME/opt/vnc" sh tokenhsi/scripts/multi_agent/run-gui.sh \
  sh tokenhsi/scripts/multi_agent/approach_distance_success_test.sh /path/to/checkpoint.pth 2 1 3
```

Or invoke the runner directly:

```bash
python ./tokenhsi/run.py --task HumanoidMACarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_carry.yaml \
    --cfg_env tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --num_envs 2048 --num_agents 2 --num_objects 3 \
    --output_path output/ma_carry --headless
```

The multi-agent task now defaults to `policyObsMode: clean_scene`: one intrinsic H/O/T
node set and one compact pose per token are stored per environment. A2 supplies typed
semantic relation bias, GTA aligns Q/K/V through token-wise SE(3) transforms, and all
humanoid actions/values are read from one scene encoding. Set
`policyObsMode: legacy_multirow` in the task config for the exact ego-first A1 baseline.
Semantic A1/A2 ablation is controlled by `relation_bias_mode` (`lookup|edge_mlp`).
The GTA path uses `gta.enable: true` and requires historical `geometry.enable: false`.
See [`markdowns/ma_clean_scene_gta.md`](markdowns/ma_clean_scene_gta.md) for the feature
split, normalization, transform convention, and shape flow.

## Acknowledgements

This work builds directly on [TokenHSI](https://github.com/liangpan99/TokenHSI) by Liang Pan,
Zeshi Yang, Zhiyang Dou, Wenjia Wang, Buzhen Huang, Bo Dai, Taku Komura and Jingbo Wang, which in
turn builds on [ASE](https://github.com/nv-tlabs/ASE), [PADL](https://github.com/nv-tlabs/PADL) and
[InterScene](https://github.com/liangpan99/InterScene). All credit for the base method belongs to
the original authors.

```bibtex
@inproceedings{pan2025tokenhsi,
  title={TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization},
  author={Pan, Liang and Yang, Zeshi and Dou, Zhiyang and Wang, Wenjia and Huang, Buzhen and Dai, Bo and Komura, Taku and Wang, Jingbo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## License

Released under the [MIT License](LICENSE), matching the upstream TokenHSI license. External
libraries and datasets (AMASS, SAMP, OMOMO, SMPL, IsaacGym) remain subject to their own terms.
