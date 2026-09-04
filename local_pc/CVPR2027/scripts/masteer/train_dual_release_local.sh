#!/bin/bash
# Continue ms18 on the paired shelf/solid-box release-and-clear curriculum.
# The underlying stage1 actor backbone remains frozen (MA_NOFREEZE is unset).
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-ms19_dualsupport_release_c06_s0}

export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
export MA_GPU=${MA_GPU:-0}
export MS_TASK=HumanoidMADualSupportRelease
export MA_TOKEN=mask
export MA_TOKENIZER_ZERO=1
export MA_NOFREEZE=0
export MA_INIT_CKPT=${MA_INIT_CKPT:-$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid.pth}

export MS_AGENTS=2
export MS_ENVS=${MS_ENVS:-2048}
export MS_CP=${MS_CP:-4}
export MS_SPACING=${MS_SPACING:-5}
export MS_ITERS=${MS_ITERS:-3000}
export MS_SAVE_LATEST=100
export MS_SAVE_ARCHIVE=3000
export MS_SEED=${MS_SEED:-0}

# Preserve the ms18 carry/steering settings.
export MA_SEP=${MA_SEP:-4.0}
export MA_SPAWN_GAP=${MA_SPAWN_GAP:-1.0}
export MA_C=${MA_C:-0}
export MA_BETA=${MA_BETA:-0}
export MS_MRAND=${MS_MRAND:-4}
export MS_M_LO=${MS_M_LO:-0.25}
export MS_VEL_W=${MS_VEL_W:-1}
export MS_REWARD_OUTER=${MS_REWARD_OUTER:-1}
export MS_POS_C=${MS_POS_C:-0.6}
export MS_CLIP=${MS_CLIP:-1}
export MS_SCEN=free
export MS_GRADCHK=${MS_GRADCHK:-1}

# Paired random geometry: only the material below the top surface differs.
export REL_TOP_LO=${REL_TOP_LO:-0.25}
export REL_TOP_HI=${REL_TOP_HI:-0.90}
export REL_SUPPORT_MODE=${REL_SUPPORT_MODE:-shelf}
export REL_RELEASE_SCALE=${REL_RELEASE_SCALE:-0.25}
export REL_SUPPORT_XY_LO=${REL_SUPPORT_XY_LO:-0.35}
export REL_SUPPORT_XY_HI=${REL_SUPPORT_XY_HI:-0.65}
export REL_SUPPORT_MARGIN=${REL_SUPPORT_MARGIN:-0.08}
export REL_TARGET_DIST_LO=${REL_TARGET_DIST_LO:-2.0}
export REL_TARGET_DIST_HI=${REL_TARGET_DIST_HI:-5.0}
export REL_TARGET_JITTER_DEG=${REL_TARGET_JITTER_DEG:-50}
export REL_CLEAR_LO=${REL_CLEAR_LO:-0.8}
export REL_CLEAR_HI=${REL_CLEAR_HI:-1.2}

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
