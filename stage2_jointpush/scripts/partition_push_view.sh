#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ "$#" != 1 ]; then
  echo "Usage: bash $0 /path/to/PartitionPushCa4_checkpoint.pth" >&2
  exit 2
fi
CHECKPOINT=$(realpath -e "$1")
export HEADLESS=0
# Restrict Vulkan discovery to NVIDIA on this host. Automatic ICD discovery
# produces a black native framebuffer despite successful off-screen rendering.
if [ -z "${VK_ICD_FILENAMES:-}" ] && [ -z "${VK_DRIVER_FILES:-}" ] && \
   [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi
export NUM_ENVS=${NUM_ENVS:-1}
export REPEATS=${REPEATS:-1000000}
export OUTPUT_PATH=${OUTPUT_PATH:-$ROOT/../runs/jointpush/partition_push_view}
exec bash "$ROOT/scripts/partition_push_test.sh" "$CHECKPOINT"
