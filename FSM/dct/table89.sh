#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin=${TABLE89_PYTHON:-python}

case "${1:---prepare-only}" in
    --prepare-only) exec "$python_bin" FSM/dct/run_cohort.py --prepare-only ;;
    --run) exec "$python_bin" FSM/dct/run_cohort.py ;;
    --run-local-gpu0) exec "$python_bin" FSM/dct/run_cohort.py --local-gpu0 ;;
    --follower-timeline) exec "$python_bin" FSM/dct/follower_timeline.py "${@:2}" ;;
    --post-at) exec "$python_bin" FSM/dct/post_at_time.py \
        --suite runs/table89_delay/cohort --output runs/table89_delay/post_at ;;
    *) printf 'Usage: %s [--prepare-only|--run|--run-local-gpu0|--post-at|--follower-timeline]\n' "$0" >&2; exit 2 ;;
esac
