"""Experiment config isolation; these tests never start a simulator or training."""
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / 'tokenhsi/scripts/multi_agent/relation_variant.sh'
ENV_DIR = ROOT / 'tokenhsi/data/cfg/multi_agent'


def profile(variant, seed=None):
    env = dict(os.environ, RELATION_VARIANT=variant)
    env.pop('SEED', None)
    if seed is not None:
        env['SEED'] = str(seed)
    return subprocess.run(['bash', '-c',
        'set -eu; . "$1"; printf "%s\\n" "$RELATION_ENV_CFG" "$RELATION_OUTPUT_DEFAULT" '
        '"$RELATION_EXPERIMENT" "${SEED:-unset}"', 'profile', str(PROFILE)],
        env=env, text=True, capture_output=True)


def test_state2_changes_only_state_delta_weight():
    original = yaml.safe_load((ENV_DIR / 'amp_humanoid_ma_carry_relation.yaml').read_text())
    state2 = yaml.safe_load((ENV_DIR / 'amp_humanoid_ma_carry_relation_state2.yaml').read_text())
    assert original['env']['relationReward']['state_delta_weight'] == 1.
    assert state2['env']['relationReward']['state_delta_weight'] == 2.
    validate_relation_config(state2['env']['relationReward'])
    state2['env']['relationReward']['state_delta_weight'] = 1.
    assert state2 == original
    train = yaml.safe_load((ROOT / 'tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation.yaml').read_text())
    assert original['env']['numEnvs'] == 2048
    assert train['params']['config']['minibatch_size'] == 16384
    assert train['params']['config']['mini_epochs'] == 6


def test_variant_routes_and_seed():
    old, new = profile('v0'), profile('state2')
    assert old.returncode == new.returncode == 0
    a, b = old.stdout.splitlines(), new.stdout.splitlines()
    assert a[1:] == ['output/ma_carry_relation_v0', '', 'unset']
    assert b[1:] == ['output/ma_carry_relation_state2', 'CarryRelationState2', '9896']
    assert a[0] != b[0]
    assert all((ROOT / p[0]).is_file() for p in (a, b))
    assert profile('state2', seed=42).stdout.splitlines()[-1] == '42'
    assert profile('unknown').returncode != 0


def test_state2_signed_changes_only_progress_mode_and_isolates_checkpoints():
    original = yaml.safe_load((ENV_DIR / 'amp_humanoid_ma_carry_relation_state2.yaml').read_text())
    signed = yaml.safe_load((ENV_DIR / 'amp_humanoid_ma_carry_relation_state2_signed.yaml').read_text())
    cfg = signed['env']['relationReward']
    validate_relation_config(cfg)
    saved = {'relation_metadata': checkpoint_metadata(cfg)}
    check_checkpoint_metadata(saved, checkpoint_metadata(cfg))
    with pytest.raises(ValueError, match='reward config differs'):
        check_checkpoint_metadata(saved, checkpoint_metadata(original['env']['relationReward']))
    with pytest.raises(ValueError, match='reward config differs'):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(original['env']['relationReward'])},
                                  checkpoint_metadata(cfg))
    assert cfg['state_delta_weight'] == 2.
    assert cfg['velocity_progress_weight'] == .2
    assert cfg['progress'].pop('mode') == 'signed_linear'
    assert signed == original
    cfg['progress']['mode'] = 'invalid'
    with pytest.raises(ValueError, match='progress.mode'):
        validate_relation_config(cfg)


def test_signed_variant_routes_and_seed():
    result = profile('state2_signed')
    assert result.returncode == 0
    values = result.stdout.splitlines()
    assert values[1:] == ['output/ma_carry_relation_state2_signed', 'CarryRelationState2Signed', '9896']
    assert (ROOT / values[0]).is_file()
    assert profile('state2_signed', seed=42).stdout.splitlines()[-1] == '42'


@pytest.mark.parametrize('testing', [False, True])
def test_signed_script_arguments_without_starting_python(tmp_path, testing):
    env = dict(os.environ, RELATION_VARIANT='state2_signed', CONDA_DEFAULT_ENV='tokenhsi',
               TOKENHSI_CONDA_ENV='tokenhsi')
    for key in ['SMOKE', 'SEED', 'MAX_ITERATIONS', 'RESUME_CHECKPOINT', 'OUTPUT_PATH', 'HEADLESS']:
        env.pop(key, None)
    script = PROFILE.parent / ('ma_carry_relation_test.sh' if testing else 'ma_carry_relation_train.sh')
    args = []
    if testing:
        ckpt = tmp_path / 'model.pth'
        ckpt.touch()
        args = [str(ckpt)]
    result = subprocess.run(['bash', '-c',
        'python() { printf "%s\\n" "$@"; }; export -f python; bash "$@"',
        'script', str(script), *args], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    flags = result.stdout.splitlines()
    def value(name): return flags[flags.index(name) + 1]
    assert value('--cfg_env').endswith('amp_humanoid_ma_carry_relation_state2_signed.yaml')
    assert value('--output_path') == 'output/ma_carry_relation_state2_signed'
    assert value('--seed') == '9896'
    assert '--headless' in flags
    if not testing:
        assert value('--num_envs') == '2048'
        assert value('--num_agents') == '2' and value('--num_objects') == '3'
        assert value('--experiment') == 'CarryRelationState2Signed'
        assert '--checkpoint' not in flags and '--resume' not in flags


@pytest.mark.parametrize('script', ['ma_carry_relation_train.sh', 'ma_carry_relation_test.sh',
                                     'relation_variant.sh'])
def test_shell_syntax(script):
    subprocess.run(['bash', '-n', str(PROFILE.parent / script)], check=True)
