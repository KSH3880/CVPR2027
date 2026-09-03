#!/bin/bash
# Shared runtime setup, sourced by the multi-agent scripts.
#
# Nothing here is machine specific -- everything is discovered or overridable:
#
#   TOKENHSI_CONDA_ENV  conda env to activate.
#                       Default: the env you already activated, else "tokenhsi".
#                       Set this if you named your env something else.
#   CONDA_BASE          conda install prefix, if auto-detection fails
#                       (e.g. CONDA_BASE=/opt/miniconda3).
#   TOKENHSI_GPU        value for CUDA_VISIBLE_DEVICES (default: 0).
#
# The repo root is derived from this script's own location, so the checkout can
# live anywhere.

# Locate this file, whether it was sourced by another script or run directly.
# $BASH_SOURCE (element 0, no array syntax so dash/sh does not choke) is exact
# under bash; under a POSIX shell it is unset and $0 is the sourcing script,
# which sits in this same directory.
if [ -n "${BASH_SOURCE:-}" ]; then
    _tokenhsi_self=$BASH_SOURCE
else
    _tokenhsi_self=$0
fi
TOKENHSI_ROOT=$(cd "$(dirname "$_tokenhsi_self")/../../.." && pwd)
export TOKENHSI_ROOT

# --- conda environment -----------------------------------------------------
# If the caller is already inside a conda env and did not ask for a specific
# one, stay where we are instead of guessing a name.
if [ -z "${TOKENHSI_CONDA_ENV:-}" ] && [ -n "${CONDA_DEFAULT_ENV:-}" ] \
    && [ "${CONDA_DEFAULT_ENV}" != "base" ]; then
    TOKENHSI_CONDA_ENV=$CONDA_DEFAULT_ENV
fi
TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}

if [ "${CONDA_DEFAULT_ENV:-}" != "$TOKENHSI_CONDA_ENV" ]; then
    conda_base=${CONDA_BASE:-}
    if [ -z "$conda_base" ] && command -v conda >/dev/null 2>&1; then
        conda_base=$(conda info --base 2>/dev/null)
    fi
    if [ -z "$conda_base" ]; then
        for candidate in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" \
                         "$HOME/mambaforge" "$HOME/conda" /opt/conda /opt/miniconda3; do
            if [ -x "$candidate/bin/conda" ]; then
                conda_base=$candidate
                break
            fi
        done
    fi
    if [ -z "$conda_base" ]; then
        echo "conda not found. Set CONDA_BASE=/path/to/your/conda," >&2
        echo "or activate the environment yourself before running this script." >&2
        exit 1
    fi

    . "$conda_base/etc/profile.d/conda.sh"
    if ! conda activate "$TOKENHSI_CONDA_ENV"; then
        echo "could not activate conda env '$TOKENHSI_CONDA_ENV'." >&2
        echo "Set TOKENHSI_CONDA_ENV to the name of your environment, e.g." >&2
        echo "  TOKENHSI_CONDA_ENV=my-env $0 ..." >&2
        exit 1
    fi
fi
export TOKENHSI_CONDA_ENV

cd "$TOKENHSI_ROOT" || exit 1

# --- GPU selection ---------------------------------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${TOKENHSI_GPU:-${CUDA_VISIBLE_DEVICES:-0}}

# --- libcuda shim ----------------------------------------------------------
# Some driver packages ship libcuda.so.1 but no unversioned libcuda.so, which
# IsaacGym's loader wants. Provide it through a throwaway symlink dir.
#
# ldconfig lives in /sbin on many distros and is absent entirely elsewhere, so
# a failed lookup is a "can't tell" rather than an error -- if CUDA really is
# missing, IsaacGym reports it far more precisely than we could here.
tokenhsi_ldconfig=$(command -v ldconfig || true)
if [ -z "$tokenhsi_ldconfig" ] && [ -x /sbin/ldconfig ]; then
    tokenhsi_ldconfig=/sbin/ldconfig
fi

if [ -n "$tokenhsi_ldconfig" ] && \
   ! "$tokenhsi_ldconfig" -p 2>/dev/null | awk '$1 == "libcuda.so" { found=1 } END { exit !found }'; then
    cuda_driver=$("$tokenhsi_ldconfig" -p 2>/dev/null | awk '$1 == "libcuda.so.1" { print $NF; exit }')
    if [ -n "$cuda_driver" ]; then
        tokenhsi_cuda_compat=$(mktemp -d "${TMPDIR:-/tmp}/tokenhsi-libcuda.XXXXXX")
        ln -s "$cuda_driver" "$tokenhsi_cuda_compat/libcuda.so"
        export LD_LIBRARY_PATH="$tokenhsi_cuda_compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        trap 'rm -f "$tokenhsi_cuda_compat/libcuda.so"; rmdir "$tokenhsi_cuda_compat"' EXIT
    fi
fi
