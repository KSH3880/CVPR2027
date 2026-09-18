from pathlib import Path

import pytest
import yaml

from utils.rsi_curriculum import SkillInitCurriculum
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata


CONFIG_DIR = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent'


def load_config():
    return yaml.safe_load((CONFIG_DIR / 'approach_rsi.yaml').read_text())


def schedule():
    env = load_config()['env']
    return SkillInitCurriculum(env['skill'], env['skillInitProb'], env['skillInitCurriculum'])


@pytest.mark.parametrize('epoch, expected, blend', [
    (0, [0, .5, .4, .1, 0], 0),
    (500, [0, .5, .4, .1, 0], 0),
    (1000, [0, .5, .4, .1, 0], 0),
    (1001, [0, .5, .3997, .1002, .0001], .001),
    (1500, [0, .5, .25, .2, .05], .5),
    (1999, [0, .5, .1003, .2998, .0999], .999),
    (2000, [0, .5, .1, .3, .1], 1),
    (9000, [0, .5, .1, .3, .1], 1),
])
def test_absolute_epoch_schedule(epoch, expected, blend):
    probabilities, actual_blend = schedule().at_epoch(epoch)
    assert probabilities == pytest.approx(expected)
    assert actual_blend == pytest.approx(blend)
    assert sum(probabilities) == pytest.approx(1)


def test_config_changes_only_rsi_and_remains_checkpoint_compatible():
    base = yaml.safe_load((CONFIG_DIR /
        'amp_humanoid_ma_carry_relation_state02_near_dir_success10_approach.yaml').read_text())
    new = load_config()
    new['env'].pop('skillInitCurriculum')
    assert new == base
    check_checkpoint_metadata(
        {'relation_metadata': checkpoint_metadata(base['env']['relationReward'])},
        checkpoint_metadata(new['env']['relationReward']))


@pytest.mark.parametrize('change', [
    {'initialProb': [0, 1]},
    {'initialProb': [0, .5, .4, .2, 0]},
    {'initialProb': [0, .5, .4, .2, -.1]},
    {'initialProb': [0, .5, float('nan'), .1, 0]},
    {'warmupEpochs': -1},
    {'warmupEpochs': 2000},
    {'endEpoch': 999},
    {'endEpoch': 2000.5},
])
def test_invalid_schedule_fails_early(change):
    env = load_config()['env']
    env['skillInitCurriculum'].update(change)
    with pytest.raises(ValueError):
        SkillInitCurriculum(env['skill'], env['skillInitProb'], env['skillInitCurriculum'])
