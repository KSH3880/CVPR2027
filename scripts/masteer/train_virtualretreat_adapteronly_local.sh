#!/bin/bash
# Train only TokenHSI's existing internal adapter on carry rehearsal + release.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-ms25_virtualretreat_adapteronly_r50_s0}

export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
export MS_DDP_N=1
export MA_GPU=7
export MS_TASK=HumanoidMASequentialStackRelease
export MA_TOKEN=mask
export MA_TOKENIZER_ZERO=1
export MA_NOFREEZE=0
export MA_ADAPTER_ONLY=${MA_ADAPTER_ONLY:-1}
export MA_FREEZE_INPUT_RMS=${MA_FREEZE_INPUT_RMS:-1}
export MA_FREEZE_NEW_CARRY=${MA_FREEZE_NEW_CARRY:-1}
export MA_INIT_CKPT=${MA_INIT_CKPT:-$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth}

export MS_AGENTS=2
export MS_ENVS=${MS_ENVS:-2048}
export MS_CP=${MS_CP:-4}
export MS_SPACING=${MS_SPACING:-5}
export MS_ITERS=${MS_ITERS:-3000}
export MS_SAVE_LATEST=${MS_SAVE_LATEST:-100}
export MS_SAVE_ARCHIVE=${MS_SAVE_ARCHIVE:-3000}
export MS_SEED=${MS_SEED:-0}

# Exact ms18 carry/steering settings before the state-based release switch.
export MA_SEP=${MA_SEP:-0}
export MA_SPAWN_GAP=${MA_SPAWN_GAP:-1.5}
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
export STACK_REHEARSAL_FRAC=0.5
export STACK_VIRTUAL_RETREAT_BOX=1
export STACK_CLEAR_SIGNED=1

# Stable-placement release reward and centralized dependency controller.
export STACK_SIZE_MARGIN=${STACK_SIZE_MARGIN:-0.05}
export STACK_GOAL_X=${STACK_GOAL_X:-4.0}
export STACK_GOAL_Y=${STACK_GOAL_Y:-0.0}
export STACK_STAGE_DIST=${STACK_STAGE_DIST:-2.0}
export STACK_STAGE_Z=${STACK_STAGE_Z:-0.90}
export STACK_STAGE_TOL=${STACK_STAGE_TOL:-0.45}
export STACK_BODY_CLEAR=${STACK_BODY_CLEAR:-1.0}
export STACK_RETREAT_DIST=${STACK_RETREAT_DIST:-1.5}
export STACK_RETREAT_SCALE=${STACK_RETREAT_SCALE:-0.5}
export STACK_XY_TOL=${STACK_XY_TOL:-0.10}
export STACK_Z_TOL=${STACK_Z_TOL:-0.06}
export STACK_ENTRY_LIN=${STACK_ENTRY_LIN:-0.25}
export STACK_ENTRY_ANG=${STACK_ENTRY_ANG:-1.0}
export STACK_ENTRY_STEPS=${STACK_ENTRY_STEPS:-5}
export STACK_HAND_CLEAR=${STACK_HAND_CLEAR:-0.15}
export STACK_HAND_STEPS=${STACK_HAND_STEPS:-5}
export STACK_STABLE_LIN=${STACK_STABLE_LIN:-0.08}
export STACK_STABLE_ANG=${STACK_STABLE_ANG:-0.20}
export STACK_TOP_STEPS=${STACK_TOP_STEPS:-20}
export STACK_DROP_XY=${STACK_DROP_XY:-0.25}
export STACK_TRANSITION_BONUS=${STACK_TRANSITION_BONUS:-3.0}
export STACK_CLEAR_BONUS=${STACK_CLEAR_BONUS:-1.5}
export STACK_CLEAR_PROGRESS_W=${STACK_CLEAR_PROGRESS_W:-0.7}
export STACK_CLEAR_SPEED_W=${STACK_CLEAR_SPEED_W:-0.3}
export STACK_CLEAR_STEER_W=${STACK_CLEAR_STEER_W:-0.25}
export STACK_CLEAR_VEL_K=${STACK_CLEAR_VEL_K:-4.0}
export STACK_CLEAR_LAT_K=${STACK_CLEAR_LAT_K:-1.0}
export STACK_CLEAR_MIN_SPEED=${STACK_CLEAR_MIN_SPEED:-0.05}
export STACK_CLEAR_HARD_GATE=${STACK_CLEAR_HARD_GATE:-1}

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
