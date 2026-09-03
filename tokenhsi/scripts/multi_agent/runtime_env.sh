#!/bin/bash

TOKENHSI_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)

if command -v conda >/dev/null 2>&1; then
    conda_base=$(conda info --base)
elif [ -x "$HOME/anaconda3/bin/conda" ]; then
    conda_base="$HOME/anaconda3"
else
    echo "conda를 찾을 수 없습니다." >&2
    exit 1
fi

. "$conda_base/etc/profile.d/conda.sh"
if ! conda activate tokenhsi_jhh; then
    echo "tokenhsi_jhh conda 환경을 활성화할 수 없습니다." >&2
    exit 1
fi
cd "$TOKENHSI_ROOT" || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

if ! ldconfig -p 2>/dev/null | awk '$1 == "libcuda.so" { found=1 } END { exit !found }'; then
    cuda_driver=$(ldconfig -p 2>/dev/null | awk '$1 == "libcuda.so.1" { print $NF; exit }')
    if [ -z "$cuda_driver" ]; then
        echo "libcuda.so.1을 찾을 수 없습니다." >&2
        exit 1
    fi

    tokenhsi_cuda_compat=$(mktemp -d "${TMPDIR:-/tmp}/tokenhsi-libcuda.XXXXXX")
    ln -s "$cuda_driver" "$tokenhsi_cuda_compat/libcuda.so"
    export LD_LIBRARY_PATH="$tokenhsi_cuda_compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    trap 'rm -f "$tokenhsi_cuda_compat/libcuda.so"; rmdir "$tokenhsi_cuda_compat"' EXIT
fi
