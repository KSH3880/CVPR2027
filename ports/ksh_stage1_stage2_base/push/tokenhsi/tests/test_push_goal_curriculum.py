from isaacgym import gymapi
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from utils.stage1_push import validate_env, sample_boxes_and_goals, check_checkpoint
from utils.push_goal_curriculum import distance_range, migrate_checkpoint
from learning.multi_agent.ma_agent import MAAgent

NAMES = ['stage1_push_depth180_h100_rsi1m_at5deg', 'stage1_push_depth180_h080_rsi1m']


def env(name, new=True):
    suffix = '_goal_curriculum' if new else ''
    return yaml.safe_load(Path(f'tokenhsi/data/cfg/multi_agent/{name}{suffix}.yaml').read_text())['env']


@pytest.mark.parametrize('epoch,hi', [(0,5), (999,5), (1000,7), (2999,7), (3000,10), (9000,10)])
@pytest.mark.parametrize('name', NAMES)
def test_ranges_and_world_goals(name, epoch, hi):
    cfg = env(name); validate_env(cfg)
    assert distance_range(cfg, epoch) == [1, hi]
    sizes = torch.tensor([1.3,1.8,1.3]).expand(2048,4,3)
    states, goals, forward, distances = sample_boxes_and_goals(2048,'cpu',torch.zeros(2048,3),cfg,sizes,epoch)
    assert distances.min() >= 1 and hi-.05 < distances.max() <= hi
    assert torch.allclose(goals-states[:,:2,:3],forward*distances[...,None],atol=2e-6)
    # Full goal corridors have >8m clearance from every other source object.
    for owner in range(2):
        for other in range(4):
            if owner == other: continue
            assert ((states[:,owner,:2]-states[:,other,:2]).norm(dim=-1)-hi).min() > 8


@pytest.mark.parametrize('name', NAMES)
def test_migration_preserves_training_state_and_rejects_other_experiment(name):
    old,new=env(name,False),env(name)
    weights=dict(epoch=777,frame=12345,model={'x':torch.randn(4)},optimizer={'state':torch.ones(3)},
                 relation_experiment_config={'env':old,'extra':'keep'})
    with pytest.raises(ValueError):check_checkpoint(weights,new)
    migrated=migrate_checkpoint(weights,new)
    check_checkpoint(migrated,new)
    assert weights['relation_experiment_config']['env']==old
    for key in ('epoch','frame','model','optimizer'): assert migrated[key] is weights[key]
    assert migrated['relation_experiment_config']['extra']=='keep'
    other=deepcopy(new);other['pushTask']['height_m']+=.1
    with pytest.raises(ValueError):migrate_checkpoint(weights,other)


def test_training_hook_before_first_reset_and_epoch_boundary():
    task=SimpleNamespace(cfg={'env':env(NAMES[0])})
    logged=[]
    agent=object.__new__(MAAgent)
    agent.vec_env=SimpleNamespace(env=SimpleNamespace(task=task))
    agent.writer=SimpleNamespace(add_scalar=lambda *args:logged.append(args))
    agent.frame=1234
    for epoch,hi in [(500,5),(999,5),(1000,7),(3000,10)]:
        agent.epoch_num=epoch;agent._update_training_curriculum()
        assert task._push_goal_epoch==epoch and logged[-1]==('push/goal_max_m',hi,1234)


def test_legacy_sampling_and_invalid_schedule():
    assert distance_range(env(NAMES[0],False),9999)==[1,4]
    bad=env(NAMES[0]);bad['pushGoalRandomization']['curriculum'][1]['epoch']=999
    with pytest.raises(ValueError):validate_env(bad)
