import json
from types import SimpleNamespace

import isaacgym
import pytest

import evaluate


@pytest.mark.parametrize('explicit_push', [False, True])
def test_doorprogress_evaluation_uses_coupled_task_jointpush_data_and_push_scores(monkeypatch, tmp_path, explicit_push):
    name = 'stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress'
    checkpoint = tmp_path/'model.pth'
    checkpoint.write_bytes(b'unit-test-checkpoint')
    output = tmp_path/'output'
    args = ['evaluate.py', name, str(checkpoint), '--envs', '64', '--seeds', '0', '--output', str(output)]
    if explicit_push:
        args += ['--tasks', 'PUSH']
    monkeypatch.setattr(evaluate.sys, 'argv', args)
    monkeypatch.setattr(evaluate, 'resolve_output_path', lambda experiment, mode, path: path)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(evaluate.subprocess, 'run', run)
    monkeypatch.setattr(evaluate, 'summarize', lambda path: {})
    evaluate.main()
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[cmd.index('--task')+1] == 'HumanoidMACoupledPush'
    assert cmd[cmd.index('--motion_file')+1] == 'tokenhsi/data/dataset_stage1_jointpush.yaml'
    assert cmd[cmd.index('--num_envs')+1] == '64'
    assert cmd[cmd.index('--coupled_assignment')+1] == 'ordered'
    manifest = json.loads((output/'manifest.json').read_text())
    assert 'PUSH' in manifest['tasks'] and manifest['horizon'] == 900
    assert manifest['formal_report'] is False
