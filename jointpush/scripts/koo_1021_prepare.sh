#!/bin/bash
set -eu
source "$(dirname "$0")/koo_1021_common.sh"
export CUDA_VISIBLE_DEVICES=''
exec "$KOO_PYTHON" "$KOO_PROJECT/jointpush/scripts/koo_1021_prepare.py" "$@"
