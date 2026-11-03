"""Production/check routing, including legacy paths and symlink escapes."""
from pathlib import Path
import subprocess

import pytest

from utils.output_paths import resolve_output_path


@pytest.mark.parametrize('experiment', ['stage1', 'stage2_before', 'stage2_before_rsi433', 'stage2_before_nonemask_rsi433_near50'])
def test_training_and_evaluation_are_separate(tmp_path, experiment):
    def output(mode, **kwargs):
        return resolve_output_path(experiment, mode, root=tmp_path, **kwargs)

    assert output('train') == 'output/' + experiment
    assert output('train', requested='output') == 'output/' + experiment
    assert output('train', check=True) == 'output_etc/check/' + experiment + '/train'
    assert output('train', max_iterations=1) == 'output_etc/check/' + experiment + '/train'
    assert output('eval') == 'output_etc/check/' + experiment + '/eval'
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('mode,requested,kwargs,expected', [
    ('train', 'output/fixed_check', {}, 'output_etc/check/fixed_check'),
    ('train', 'output/smoke/fixed', {}, 'output_etc/check/smoke/fixed'),
    ('train', 'output/short', {'max_iterations': 2}, 'output_etc/check/short'),
    ('eval', 'output/fixed', {}, 'output_etc/check/fixed'),
    ('eval', 'output_etc/check/custom', {}, 'output_etc/check/custom'),
    ('train', 'output/fixed_resume', {}, 'output/fixed_resume'),
])
def test_explicit_and_legacy_paths(tmp_path, mode, requested, kwargs, expected):
    for path in (requested, str(tmp_path / requested)):
        assert resolve_output_path('stage1', mode, path, root=tmp_path,
                                   **kwargs) == expected


@pytest.mark.parametrize('requested', ['output_etc/old_check', '../outside', '/tmp/outside'])
def test_verification_cannot_write_outside_check_root(tmp_path, requested):
    with pytest.raises(ValueError, match='Verification output must be under'):
        resolve_output_path('stage1', 'eval', requested, root=tmp_path)


def test_symlinks_cannot_escape_check_root(tmp_path):
    checks = tmp_path / 'output_etc/check'
    checks.mkdir(parents=True)
    (tmp_path / 'outside').mkdir()
    (checks / 'escape').symlink_to(tmp_path / 'outside', target_is_directory=True)
    with pytest.raises(ValueError, match='Verification output must be under'):
        resolve_output_path('stage1', 'eval', str(checks / 'escape/run'), root=tmp_path)


def test_shell_check_default_is_bounded_and_exports_output():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ['bash', '-c', 'set -eu; unset MAX_ITERATIONS OUTPUT_PATH; CHECK=1; '
         '. tokenhsi/scripts/multi_agent/output_paths.sh; '
         'tokenhsi_select_output stage1 train; '
         'test "$MAX_ITERATIONS" = 1; '
         'bash -c \'test "$OUTPUT_PATH" = output_etc/check/stage1/train\''],
        cwd=str(root), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
