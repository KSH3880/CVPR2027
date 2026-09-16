#!/usr/bin/env python3
"""Fail closed unless Vulkan exposes exactly the requested physical GPU."""

from __future__ import annotations

import ctypes as C
import os
import subprocess
import sys


DEFAULT_ALLOWED_GPUS = {"6", "7"}
VK_SUCCESS = 0
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES = 1000071004


def fail(message: str) -> "NoReturn":
    print(f"VULKAN_GPU_GUARD: {message}", file=sys.stderr)
    raise SystemExit(2)


def normalize_uuid(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("gpu-"):
        value = value[4:]
    return value.replace("-", "")


allowed_raw = os.environ.get(
    "TOKENHSI_ALLOWED_GPUS", ",".join(sorted(DEFAULT_ALLOWED_GPUS, key=int))
)
allowed_gpus = {value.strip() for value in allowed_raw.split(",") if value.strip()}
if not allowed_gpus or any(not value.isdigit() for value in allowed_gpus):
    fail("TOKENHSI_ALLOWED_GPUS must be a comma-separated list of GPU indices")
if len(sys.argv) != 2 or sys.argv[1] not in allowed_gpus:
    allowed_text = ", ".join(sorted(allowed_gpus, key=int))
    fail(
        f"physical GPU must be one of: {allowed_text} "
        "(override explicitly with TOKENHSI_ALLOWED_GPUS)"
    )

if "NODEVICE_SELECT" in os.environ:
    fail("NODEVICE_SELECT is set and would disable the Vulkan device-select layer")

gpu = sys.argv[1]
selector = os.environ.get("TOKENHSI_DRI_PRIME", f"{gpu}!")
selector_disabled = selector == "none"
if not selector_disabled and not selector.endswith("!"):
    fail("DRI_PRIME selector must end in ! so Vulkan exposes only one device")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
# VK_LAYER_MESA_device_select is an implicit layer. Explicitly adding it through
# VK_INSTANCE_LAYERS can load it twice on local Mesa/Vulkan installations and
# duplicate the physical-device list.
os.environ.pop("VK_INSTANCE_LAYERS", None)
if selector_disabled:
    os.environ.pop("DRI_PRIME", None)
else:
    os.environ["DRI_PRIME"] = selector

try:
    expected = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
except (OSError, subprocess.CalledProcessError) as exc:
    fail(f"cannot query physical GPU {gpu} UUID: {exc}")

expected_uuid = normalize_uuid(expected)


class ApplicationInfo(C.Structure):
    _fields_ = [
        ("sType", C.c_uint32),
        ("pNext", C.c_void_p),
        ("pApplicationName", C.c_char_p),
        ("applicationVersion", C.c_uint32),
        ("pEngineName", C.c_char_p),
        ("engineVersion", C.c_uint32),
        ("apiVersion", C.c_uint32),
    ]


class InstanceCreateInfo(C.Structure):
    _fields_ = [
        ("sType", C.c_uint32),
        ("pNext", C.c_void_p),
        ("flags", C.c_uint32),
        ("pApplicationInfo", C.POINTER(ApplicationInfo)),
        ("enabledLayerCount", C.c_uint32),
        ("ppEnabledLayerNames", C.POINTER(C.c_char_p)),
        ("enabledExtensionCount", C.c_uint32),
        ("ppEnabledExtensionNames", C.POINTER(C.c_char_p)),
    ]


class PhysicalDeviceIDProperties(C.Structure):
    _fields_ = [
        ("sType", C.c_uint32),
        ("pNext", C.c_void_p),
        ("deviceUUID", C.c_ubyte * 16),
        ("driverUUID", C.c_ubyte * 16),
        ("deviceLUID", C.c_ubyte * 8),
        ("deviceNodeMask", C.c_uint32),
        ("deviceLUIDValid", C.c_uint32),
    ]


class PhysicalDeviceProperties2(C.Structure):
    _fields_ = [
        ("sType", C.c_uint32),
        ("pNext", C.c_void_p),
        ("properties", C.c_ubyte * 4096),
    ]


try:
    vulkan = C.CDLL("libvulkan.so.1")
except OSError as exc:
    fail(f"cannot load libvulkan.so.1: {exc}")

vulkan.vkCreateInstance.argtypes = [
    C.POINTER(InstanceCreateInfo),
    C.c_void_p,
    C.POINTER(C.c_void_p),
]
vulkan.vkCreateInstance.restype = C.c_int32
vulkan.vkEnumeratePhysicalDevices.argtypes = [
    C.c_void_p,
    C.POINTER(C.c_uint32),
    C.POINTER(C.c_void_p),
]
vulkan.vkEnumeratePhysicalDevices.restype = C.c_int32
vulkan.vkGetPhysicalDeviceProperties2.argtypes = [
    C.c_void_p,
    C.POINTER(PhysicalDeviceProperties2),
]
vulkan.vkDestroyInstance.argtypes = [C.c_void_p, C.c_void_p]

api_1_2 = (1 << 22) | (2 << 12)
app = ApplicationInfo(
    VK_STRUCTURE_TYPE_APPLICATION_INFO,
    None,
    b"tokenhsi-vulkan-gpu-guard",
    1,
    b"tokenhsi-vulkan-gpu-guard",
    1,
    api_1_2,
)
create_info = InstanceCreateInfo(
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    None,
    0,
    C.pointer(app),
    0,
    None,
    0,
    None,
)
instance = C.c_void_p()
rc = vulkan.vkCreateInstance(C.byref(create_info), None, C.byref(instance))
if rc != VK_SUCCESS:
    fail(f"vkCreateInstance failed with rc={rc}; device-select layer is unavailable")

try:
    count = C.c_uint32()
    rc = vulkan.vkEnumeratePhysicalDevices(instance, C.byref(count), None)
    if rc != VK_SUCCESS:
        fail(f"vkEnumeratePhysicalDevices(count) failed with rc={rc}")
    devices = (C.c_void_p * count.value)()
    rc = vulkan.vkEnumeratePhysicalDevices(instance, C.byref(count), devices)
    if rc != VK_SUCCESS:
        fail(f"vkEnumeratePhysicalDevices(list) failed with rc={rc}")

    actual_uuids = []
    for device in devices:
        id_props = PhysicalDeviceIDProperties(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
            None,
        )
        props = PhysicalDeviceProperties2(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            C.cast(C.pointer(id_props), C.c_void_p),
        )
        vulkan.vkGetPhysicalDeviceProperties2(device, C.byref(props))
        actual_uuids.append(bytes(id_props.deviceUUID).hex())

    allow_extra = os.environ.get("TOKENHSI_VULKAN_ALLOW_EXTRA", "0") == "1"
    if count.value != 1 and not allow_extra:
        exposed = ", ".join(
            f"{index}:{uuid}" for index, uuid in enumerate(actual_uuids)
        )
        fail(
            f"DRI_PRIME={selector} exposed {count.value} Vulkan devices "
            f"({exposed}), expected exactly 1"
        )

    if not actual_uuids:
        fail(f"DRI_PRIME={selector} exposed no Vulkan devices")
    matches = [index for index, uuid in enumerate(actual_uuids) if uuid == expected_uuid]
    if len(matches) != 1:
        fail(
            f"physical GPU {gpu} UUID {expected_uuid} occurs at Vulkan indices "
            f"{matches}, expected exactly one match"
        )
    selected_index = matches[0]
    if not allow_extra and selected_index != 0:
        fail(
            f"DRI_PRIME={selector} selected Vulkan index 0 UUID {actual_uuids[0]}, "
            f"but physical GPU {gpu} is at index {selected_index} ({expected_uuid})"
        )
finally:
    vulkan.vkDestroyInstance(instance, None)

print(
    f"VULKAN_GPU_GUARD: physical GPU {gpu}, UUID {expected_uuid}, "
    f"Vulkan index {selected_index}, DRI_PRIME={selector}",
    file=sys.stderr,
)
if os.environ.get("TOKENHSI_VULKAN_OUTPUT_INDEX", "0") == "1":
    print(selected_index)
else:
    print(selector)
