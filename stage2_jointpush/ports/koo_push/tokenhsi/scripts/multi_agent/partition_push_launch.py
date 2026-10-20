"""Launch legacy Isaac Gym with Blackwell-compatible TorchScript settings."""
import runpy
import sys
from pathlib import Path

import isaacgym  # Must be imported before torch.
import torch

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12:
    # Legacy NVRTC cannot compile fusion kernels for sm_120.
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    for name in ('_jit_set_nvfuser_enabled', '_jit_set_texpr_fuser_enabled'):
        if hasattr(torch._C, name):
            getattr(torch._C, name)(False)
    print('[partition_push] Disabled legacy TorchScript fusion for Blackwell')

entry = Path(__file__).resolve().parents[2] / 'run.py'
sys.path.insert(0, str(entry.parent))
sys.argv[0] = str(entry)
runpy.run_path(str(entry), run_name='__main__')
