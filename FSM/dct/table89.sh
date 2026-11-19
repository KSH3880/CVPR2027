#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin=${TABLE89_PYTHON:-python}

case "${1:---prepare-only}" in
    --prepare-only) exec "$python_bin" FSM/dct/run_cohort.py --prepare-only ;;
    --run) exec "$python_bin" FSM/dct/run_cohort.py ;;
    --post-at) exec "$python_bin" FSM/dct/post_at_time.py \
        --suite runs/table89_delay/cohort --output runs/table89_delay/post_at ;;
    *) printf 'Usage: %s [--prepare-only|--run|--post-at]\n' "$0" >&2; exit 2 ;;
esac
