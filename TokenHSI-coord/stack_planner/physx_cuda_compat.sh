#!/bin/bash
# Keep the unversioned CUDA driver name expected by this PhysX build available.
# /tmp is intentionally avoided because server cleanup removes it between runs.

if [ -z "${ROOT:-}" ]; then
    echo "physx_cuda_compat.sh requires ROOT" >&2
    return 2
fi

PHYSX_LIB_DIR=${PHYSX_LIB_DIR:-"$ROOT/.runtime/physx-lib"}
CUDA_DRIVER_SO1=$(ldconfig -p 2>/dev/null | awk '$1 == "libcuda.so.1" { print $NF; exit }')
if [ -z "$CUDA_DRIVER_SO1" ] || [ ! -r "$CUDA_DRIVER_SO1" ]; then
    echo "readable libcuda.so.1 not found; NVIDIA driver install is incomplete" >&2
    return 2
fi

mkdir -p "$PHYSX_LIB_DIR"
if [ ! -e "$PHYSX_LIB_DIR/libcuda.so" ]; then
    ln -s "$CUDA_DRIVER_SO1" "$PHYSX_LIB_DIR/libcuda.so"
fi
if [ ! -r "$PHYSX_LIB_DIR/libcuda.so" ]; then
    echo "invalid PhysX CUDA compatibility link: $PHYSX_LIB_DIR/libcuda.so" >&2
    return 2
fi
export PHYSX_LIB_DIR
export LD_LIBRARY_PATH="$PHYSX_LIB_DIR:${LD_LIBRARY_PATH:-}"
