from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from utils.stage2_jointpush import check_checkpoint, checkpoint_contract, validate_env

BASE = 'stage2_jointpush_goal_saturation_density3_cone90'
NAME = BASE + '_coupled_straight'


def test_only_coupled_goal_angle_and_run_name_change():
    folder = Path('tokenhsi/data/cfg/multi_agent')
    baseline = yaml.safe_load((folder / (BASE + '.yaml')).read_text())
    variant = yaml.safe_load((folder / (NAME + '.yaml')).read_text())
    validate_env(baseline['env'])
    validate_env(variant['env'])
    check_checkpoint({'stage2_jointpush_contract': checkpoint_contract(variant['env'])}, variant['env'])
    with pytest.raises(ValueError, match='settings do not match'):
        check_checkpoint({'stage2_jointpush_contract': checkpoint_contract(baseline['env'])}, variant['env'])
    assert variant['env']['stage2JointPush']['goal_half_angle_degrees'] == 0.
    restored = deepcopy(variant)
    restored['env']['stage2JointPush']['goal_half_angle_degrees'] = 45.
    restored['experiment']['name'] = baseline['experiment']['name']
    assert restored == baseline
    folder = Path('tokenhsi/data/cfg/train/rlg')
    baseline = yaml.safe_load((folder / (BASE + '.yaml')).read_text())
    variant = yaml.safe_load((folder / (NAME + '.yaml')).read_text())
    assert variant['params']['config']['name'] == 'Stage2JointPushGoalSaturationDensity3Cone90CoupledStraight'
    variant['params']['config']['name'] = baseline['params']['config']['name']
    assert variant == baseline


@pytest.mark.parametrize('angle', [-1., 1., 90., float('nan'), float('inf')])
def test_unsupported_coupled_angles_are_rejected(angle):
    config = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent', NAME + '.yaml').read_text())
    config['env']['stage2JointPush']['goal_half_angle_degrees'] = angle
    with pytest.raises(ValueError, match='sampling contract'):
        validate_env(config['env'])
