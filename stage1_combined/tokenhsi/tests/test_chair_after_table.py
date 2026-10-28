"""Reservation guards, including the real checkout output-path contract."""
import importlib.util
import json
import os
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('chair_after_table',
    ROOT/'tokenhsi/scripts/multi_agent/chair_after_table.py')
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)


@pytest.fixture
def config(tmp_path):
    run = tmp_path/'runs/chair_reservation'
    code = run/'code'
    (code/'tokenhsi/utils').mkdir(parents=True)
    shutil.copy2(ROOT/'tokenhsi/utils/output_paths.py', code/'tokenhsi/utils/output_paths.py')
    return dict(run=str(run), code=str(code), predecessor_exit=str(run/'table_exit.json'),
                predecessor_checkpoint=str(run/'table10000.pth'), predecessor_run='/table')


def test_fixed_preflight_uses_real_checkout_path_contract(config):
    code = Path(config['code'])
    expected = code/'output_etc/check/chair_preflight/train'
    assert scheduler.validate_output_paths(config) == expected
    resolve = __import__('runpy').run_path(str(code/'tokenhsi/utils/output_paths.py'))['resolve_output_path']
    with pytest.raises(ValueError, match='Verification output'):
        resolve('chair_adapt_residual', 'train', str(Path(config['run'])/'output_etc/check/train'),
                check=True, max_iterations=2, root=code)


def test_failed_or_partial_table_cannot_trigger_chair(config):
    assert scheduler.predecessor_status(config) == 'waiting'
    marker = Path(config['predecessor_exit'])
    marker.write_text(json.dumps(dict(status='failed', returncode=1)))
    with pytest.raises(RuntimeError, match='successfully'):
        scheduler.predecessor_status(config)
    success = dict(status='complete', returncode=0, total_epochs=10000, run='/table')
    marker.write_text(json.dumps(success))
    with pytest.raises(RuntimeError, match='checkpoint'):
        scheduler.predecessor_status(config)
    Path(config['predecessor_checkpoint']).touch()
    assert scheduler.predecessor_status(config) == 'complete'
    marker.write_text(json.dumps(dict(success, run='/another_table')))
    with pytest.raises(RuntimeError, match='match'):
        scheduler.predecessor_status(config)


def test_reused_pid_does_not_count_as_predecessor():
    identity = scheduler.process_identity(os.getpid())
    assert scheduler.matching_process(identity)
    assert not scheduler.matching_process(dict(identity, start_ticks='different'))


def test_existing_launch_cannot_start_second_training(config):
    Path(config['run'], 'state.json').write_text(json.dumps(dict(launch_attempted=True)))
    path = Path(config['run'], 'config.json')
    path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match='already attempted'):
        scheduler.main(path)
