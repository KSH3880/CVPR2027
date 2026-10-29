#!/bin/sh
# Sourced by run-gui.sh: bind CUDA and the Mesa Vulkan selection layer to one GPU.
# Isaac Gym's Vulkan instance cannot use Mesa's PCI-address selector on this
# server. With the NVIDIA ICD, its non-CPU device order matches nvidia-smi here.
# Verified with both CUDA and graphics contexts on the requested PRO 6000.
# https://docs.mesa3d.org/envvars.html

gui_gpu_requested=${TOKENHSI_GPU:-${CUDA_VISIBLE_DEVICES:-0}}
case "$gui_gpu_requested" in
    *,*|'') echo "GUI requires one GPU index or UUID, got '$gui_gpu_requested'" >&2; exit 1 ;;
esac
if ! gui_gpu_info=$(nvidia-smi --id="$gui_gpu_requested" \
    --query-gpu=index,uuid --format=csv,noheader,nounits); then
    echo "Cannot resolve GUI GPU '$gui_gpu_requested' with nvidia-smi" >&2
    exit 1
fi
gui_gpu_info=$(printf '%s' "$gui_gpu_info" | tr -d '[:blank:]')
IFS=, read -r TOKENHSI_GUI_GPU_INDEX gui_gpu_uuid <<EOF
$gui_gpu_info
EOF
case "$gui_gpu_uuid" in
    GPU-*) ;;
    *) echo "Invalid GPU metadata from nvidia-smi: $gui_gpu_info" >&2; exit 1 ;;
esac
case "$TOKENHSI_GUI_GPU_INDEX" in
    ''|*[!0-9]*) echo "Invalid GPU index from nvidia-smi: $gui_gpu_info" >&2; exit 1 ;;
esac

# runtime_env.sh in the child must retain the resolved physical GPU, even when
# CUDA indices are reordered. Inside the process this single GPU becomes cuda:0.
export TOKENHSI_GPU="$gui_gpu_uuid"
export TOKENHSI_GUI_GPU_INDEX
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gui_gpu_uuid"
export DRI_PRIME="$TOKENHSI_GUI_GPU_INDEX!"
# Exclude software/other-vendor ICDs from the viewer's device enumeration.
for gui_nvidia_icd in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
    if [ -f "$gui_nvidia_icd" ]; then
        export VK_DRIVER_FILES="$gui_nvidia_icd"
        export VK_ICD_FILENAMES="$gui_nvidia_icd"
        break
    fi
done
case ":${VK_INSTANCE_LAYERS:-}:" in
    *:VK_LAYER_MESA_device_select:*) ;;
    *) VK_INSTANCE_LAYERS="VK_LAYER_MESA_device_select${VK_INSTANCE_LAYERS:+:$VK_INSTANCE_LAYERS}" ;;
esac
export VK_INSTANCE_LAYERS
# Inherited overrides must not defeat the explicit GPU selection.
unset NODEVICE_SELECT MESA_VK_DEVICE_SELECT
export MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1
