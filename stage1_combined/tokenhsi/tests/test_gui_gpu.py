"""GPU selection must survive child runtime setup and inherited overrides."""
import os
from pathlib import Path
import subprocess

import pytest


HELPER = Path(__file__).resolve().parents[1] / 'scripts/multi_agent/gui_gpu_env.sh'


@pytest.fixture
def gpu_env(tmp_path):
    stub = tmp_path / 'nvidia-smi'
    stub.write_text('''#!/bin/sh
case "$1" in
  --id=6|--id=GPU-test-six) echo '6, GPU-test-six' ;;
  --id=0) echo '0, GPU-test-zero' ;;
  *) exit 1 ;;
esac
''')
    stub.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + ':' + os.environ['PATH'])
    for key in ('TOKENHSI_GPU', 'CUDA_VISIBLE_DEVICES', 'VK_INSTANCE_LAYERS'):
        env.pop(key, None)
    # The old hardcoded/default selectors must not silently win.
    env.update(DRI_PRIME='0!', NODEVICE_SELECT='1', MESA_VK_DEVICE_SELECT='10de:0000')
    return env


@pytest.mark.parametrize('selection,index,uuid', [
    ({'TOKENHSI_GPU': '6', 'CUDA_VISIBLE_DEVICES': '0'}, '6', 'GPU-test-six'),
    ({'CUDA_VISIBLE_DEVICES': '6'}, '6', 'GPU-test-six'),
    ({'TOKENHSI_GPU': 'GPU-test-six'}, '6', 'GPU-test-six'),
    ({}, '0', 'GPU-test-zero'),
])
def test_gpu_selection_exports_one_consistent_device(gpu_env, selection, index, uuid):
    gpu_env.update(selection)
    gpu_env['VK_INSTANCE_LAYERS'] = 'VK_LAYER_example'
    result = subprocess.run(['sh', '-c', 'set -e; . "$1"; env', 'test', str(HELPER)],
                            env=gpu_env, text=True, capture_output=True, check=True)
    values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    assert values['TOKENHSI_GUI_GPU_INDEX'] == index
    assert values['TOKENHSI_GPU'] == values['CUDA_VISIBLE_DEVICES'] == uuid
    assert values['DRI_PRIME'] == index + '!'
    assert values['MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE'] == '1'
    assert values['VK_INSTANCE_LAYERS'] == 'VK_LAYER_MESA_device_select:VK_LAYER_example'
    assert 'NODEVICE_SELECT' not in values
    assert 'MESA_VK_DEVICE_SELECT' not in values


@pytest.mark.parametrize('selection', ['6,0', '999'])
def test_invalid_gpu_does_not_fall_back_to_zero(gpu_env, selection):
    gpu_env['TOKENHSI_GPU'] = selection
    result = subprocess.run(['sh', '-c', 'set -e; . "$1"', 'test', str(HELPER)],
                            env=gpu_env, text=True, capture_output=True)
    assert result.returncode != 0
    assert selection in result.stderr
