"""Shell environment selection and common train/eval argument routing."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from utils.experiments import ROOT, NAMES


def test_runtime_activates_default_even_when_another_conda_env_is_active(tmp_path):
    base = tmp_path/'conda'
    script = base/'etc/profile.d/conda.sh'
    script.parent.mkdir(parents=True)
    script.write_text('conda() { export CONDA_DEFAULT_ENV="$2"; }\n')
    env = dict(os.environ, CONDA_BASE=str(base), CONDA_DEFAULT_ENV='unrelated',
               MPS_GPU_ROOT=str(tmp_path/'no_mps'), TOKENHSI_GPU='0')
    for key in ('TOKENHSI_CONDA_ENV', 'CUDA_MPS_PIPE_DIRECTORY'):
        env.pop(key, None)
    result = subprocess.run(['bash', '-c', 'source "$1" && printf "SELECTED=%s" "$CONDA_DEFAULT_ENV"',
        '--', str(ROOT/'tokenhsi/scripts/multi_agent/runtime_env.sh')], env=env,
        cwd=ROOT, text=True, capture_output=True, check=True)
    assert 'SELECTED=tokenhsi' in result.stdout


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('mode', ('train', 'test'))
def test_wrappers_route_arguments_through_the_same_launcher(tmp_path, name, mode):
    fake_bin = tmp_path/'bin'
    fake_bin.mkdir()
    capture = tmp_path/'args.json'
    python = fake_bin/'python'
    python.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
        'if sys.argv[1].endswith("output_paths.py"): print(os.environ["OUTPUT_PATH"])\n'
        'else: open(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
    python.chmod(0o755)
    env = dict(os.environ, PATH=str(fake_bin)+':'+os.environ['PATH'],
        CAPTURE=str(capture), CONDA_DEFAULT_ENV='tokenhsi', TOKENHSI_CONDA_ENV='tokenhsi',
        TOKENHSI_GPU='0', MPS_GPU_ROOT=str(tmp_path/'no_mps'),
        OUTPUT_PATH='output_etc/check/wrapper_test', HEADLESS='1', CHECK='1')
    for key in ('CUDA_MPS_PIPE_DIRECTORY', 'RESUME_CHECKPOINT', 'ENV_CONFIG', 'MAX_ITERATIONS'):
        env.pop(key, None)
    args=[]
    if mode=='test':
        checkpoint=tmp_path/'model.pth'
        checkpoint.touch()
        args=[str(checkpoint)]
    subprocess.run(['bash', str(ROOT/f'tokenhsi/scripts/multi_agent/{name}_{mode}.sh'), *args],
                   cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    argv=json.loads(capture.read_text())
    assert argv[0]=='tokenhsi/run.py'
    assert argv[argv.index('--cfg_train')+1].endswith('/'+name+'.yaml')
    assert argv[argv.index('--seed')+1]=='42'
    assert '--headless' in argv
    assert argv[argv.index('--num_envs')+1]==('1' if mode=='test' else '2048' if name in (NAMES[0], 'stage1_push') else '1024')
    if mode=='train':
        assert argv[argv.index('--max_iterations')+1]=='1'
