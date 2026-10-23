from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from utils.push_navigation import approach_distance, curriculum, sample_spawns, spawn_clearance
from utils.stage1_push import check_contract, check_navigation_transfer, contract

ROOT = Path(__file__).resolve().parents[2]
ENV = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360.yaml').read_text())['env']


def test_navigation_keeps_original_task_settings_and_requires_explicit_transfer():
    original = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push.yaml').read_text())['env']
    for key in original:
        if key != 'stage1Push':
            assert ENV[key] == original[key]
    saved = {'stage1_push_contract': contract(original)}
    assert ENV['stage1Push']['contact_midpoint_only'] is True
    check_navigation_transfer(saved, ENV)
    with pytest.raises(ValueError, match='warm-start'):
        check_contract(saved, ENV)
    for key, value in [('mass', 30.), ('friction', .3)]:
        changed = deepcopy(ENV)
        changed['stage1Push']['box'][key] = value
        with pytest.raises(ValueError, match='physics'):
            check_navigation_transfer(saved, changed)
    restored = {'stage1_push_contract': contract(ENV)}
    check_contract(restored, ENV)
    moved = deepcopy(ENV)
    moved['stage1Push']['warm_start_checkpoint'] = '/a/moved/checkpoint.pth'
    check_contract(restored, moved)
    previous = deepcopy(ENV)
    previous['stage1Push'].pop('contact_midpoint_only')
    previous_checkpoint = {'stage1_push_contract':contract(previous)}
    check_navigation_transfer(previous_checkpoint, ENV)
    with pytest.raises(ValueError, match='warm-start'):
        check_contract(previous_checkpoint, ENV)


@pytest.mark.parametrize('epoch,expected', [(0,0),(999,0),(1000,1),(2999,1),(3000,2),(9000,2)])
def test_curriculum_boundaries_and_full_evaluation(epoch, expected):
    options = ENV['stage1Push']
    assert curriculum(options, epoch)[0] == expected
    assert curriculum(options, epoch, evaluation=True)[0] == 2
    final = curriculum(options, evaluation=True)[1]
    assert final['goal_max'] == 10 and final['radius_max'] == 6
    assert final['half_angle_degrees'] == 180


@pytest.mark.parametrize('half_angle', [0., 180.])
def test_spawns_follow_requested_direction_and_respect_separation(half_angle):
    torch.manual_seed(42)
    n = 3000
    boxes = torch.zeros(n,4,13)
    boxes[..., 6] = 1
    boxes[:,0,0], boxes[:,1,0] = -6., 6.
    boxes[:,2,1], boxes[:,3,1] = -9., 9.
    boxes[...,2] = .65
    sizes = torch.full((n,4,3), 1.3)
    normals = torch.zeros(n,2,3)
    yaw = 2*torch.pi*torch.rand(n,2)
    normals[...,0], normals[...,1] = -yaw.sin(), yaw.cos()
    boxes[:,:2,5], boxes[:,:2,6] = (yaw/2).sin(), (yaw/2).cos()
    interaction = torch.zeros(n,2,dtype=torch.bool)
    positions = sample_spawns(boxes, sizes, normals, boxes[:,:2,:3], interaction,
                             (1.65,6.), half_angle, .45)
    delta = positions[...,:2]-boxes[:,:2,:2]
    radius = delta.norm(dim=-1)
    assert radius.min() >= 1.65 and radius.max() <= 6
    assert not spawn_clearance(positions, boxes, sizes, .45).any()
    cosine = (delta*normals[...,:2]).sum(-1)/radius
    if half_angle == 0:
        torch.testing.assert_close(cosine, -torch.ones_like(cosine))
    else:
        assert (cosine > .8).float().mean() > .15
        assert (cosine < -.8).float().mean() > .15
    assert radius.max() > 5.9
    interaction[:,0] = True
    kept = sample_spawns(boxes, sizes, normals, positions, interaction, (1.65,6.), half_angle, .45)
    torch.testing.assert_close(kept[:,0], positions[:,0])


@pytest.mark.parametrize('mode,evaluation,navigation,message', [
    ('invalid', True, True, 'random or straight'),
    ('straight', False, True, 'PUSH 360 evaluation'),
    ('straight', True, False, 'PUSH 360 evaluation'),
])
def test_spawn_mode_rejects_invalid_or_training_overrides(mode, evaluation, navigation, message):
    from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
    env = deepcopy(ENV)
    if not navigation:
        env['stage1Push'].pop('navigation')
    task = object.__new__(HumanoidMAStage1Push)
    with pytest.raises(ValueError, match=message):
        task.__init__({'env':env, 'args':SimpleNamespace(eval=evaluation, push_spawn_mode=mode)})


@pytest.mark.parametrize('mode,evaluation,message', [
    ('invalid', True, 'corrected or raw'),
    ('raw', False, 'requires evaluation'),
])
def test_rsi_mode_rejects_invalid_or_training_overrides(mode, evaluation, message):
    from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
    task = object.__new__(HumanoidMAStage1Push)
    args = SimpleNamespace(eval=evaluation, push_rsi_mode=mode)
    with pytest.raises(ValueError, match=message):
        task.__init__({'env':deepcopy(ENV), 'args':args})


def test_approach_reward_favors_the_correct_face_and_routes_around_box():
    roots = torch.tensor([[[0.,-3.,0.], [0.,3.,0.], [-3.,0.,0.], [3.,0.,0.], [.2,3.,0.]]], requires_grad=True)
    centers = torch.zeros_like(roots)
    normals = torch.zeros_like(roots)
    normals[...,1] = 1
    distance = approach_distance(roots, centers, normals, torch.full_like(roots,.65), 1., .45)
    assert torch.isfinite(distance).all()
    torch.testing.assert_close(distance[0,0], torch.tensor(1.35))
    assert distance[0,1] > 4.65
    torch.testing.assert_close(distance[0,2], distance[0,3])
    distance[0,4].backward()
    assert torch.isfinite(roots.grad).all()
    assert roots.grad[0,4,0] < 0


def test_push_horizon_preserves_old_task_timeouts_and_falls(monkeypatch):
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
    monkeypatch.setattr(HumanoidMACarry, '_compute_reset', lambda task: None)
    task = object.__new__(HumanoidMAStage1Push)
    task._push_options = ENV['stage1Push']
    task._push_horizon = 1800
    task._push_env = torch.tensor([False, True, True, True])
    task.cfg = {'env': {'episodeLength':600}}
    task.num_envs, task.num_agents = 4, 2
    task.progress_buf = torch.tensor([599,599,1799,20])
    task._terminate_buf = torch.tensor([0,0,0,0,0,0,1,0])
    task.reset_buf = torch.zeros(4,dtype=torch.long)
    task._compute_reset()
    assert task.reset_buf.tolist() == [1,0,1,1]


def test_resume_sets_curriculum_before_the_first_reset(monkeypatch):
    from learning.multi_agent.ma_agent import MAAgent
    from learning.multi_agent.stage1_push_agent import Stage1PushAgent
    monkeypatch.setattr(MAAgent, 'set_weights', lambda agent, weights: None)
    agent = object.__new__(Stage1PushAgent)
    task = SimpleNamespace(cfg={'env':ENV}, _push_curriculum_epoch=0)
    agent.vec_env = SimpleNamespace(env=SimpleNamespace(task=task))
    agent.set_weights({'stage1_push_contract':contract(ENV), 'epoch':3001})
    assert task._push_curriculum_epoch == 3001


def test_curriculum_diagnostics_obey_scalar_tensor_logging_contract(monkeypatch):
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
    monkeypatch.setattr(HumanoidMACarry, 'consume_relation_diagnostics', lambda task: {})
    task = object.__new__(HumanoidMAStage1Push)
    task._push_options = ENV['stage1Push']
    task._push_curriculum_epoch, task._is_eval, task.device = 1000, False, 'cpu'
    task._push_straight_spawn = False
    task._push_metric_count = 0
    task._push_metric_sums = torch.zeros(5)
    task._retention_sums = torch.zeros(4,2)
    task._push_env = torch.tensor([True,False])
    values = task.consume_relation_diagnostics()
    assert all(value.numel() == 1 for value in values.values())
    assert values['push/curriculum_stage'].item() == 1
    assert values['push/goal_max_m'].item() == 7
