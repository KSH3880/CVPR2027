"""Shared PUSH source is 2-10m, goal is 2-5m; solo and legacy stay intact."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from utils.coupled_push_scene import sample_push
from utils.coupled_push_spec import group_sizes, validate_env, check_checkpoint, checkpoint_contract
from utils.experiments import native_config


def task_for_sampling(shared, explicit=True):
    n = len(shared)
    groups = torch.arange(4).expand(n, -1).clone()
    groups[shared] = 0
    states = torch.zeros(n, 24, 13)
    states[..., :2] = 1000.
    states[..., 6] = 1.
    roots = torch.zeros(n, 4, 13)
    roots[..., :2] = torch.tensor([[0., 0.], [100., 0.], [0., 100.], [100., 100.]])
    roots[shared, :, :2] = 0.
    env = native_config('stage2_coupled_push_region_goals')['env']
    if not explicit:
        env['stage2JointPush'].pop('placement_order', None)
        for key in ('source_distance', 'goal_distance'):
            env['stage2JointPush'].pop(key)
    task = SimpleNamespace(
        _env_origins=torch.zeros(n, 3), _regions=torch.arange(4).expand(n, -1).clone(),
        _kinematic_humanoid_rigid_body_states=roots[:, :, None].clone(),
        cfg={'env': env}, device='cpu', _groups=groups, _members=group_sizes(groups),
        _templates_by_owner=torch.full((n, 4), 5), _humanoid_root_states=roots,
        _agent_box_assignment=groups, _box_size=torch.full((n, 24, 3), .2),
        _box_states=states, _tar_pos=torch.zeros(n, 4, 3),
        _logical_box_order=torch.cat((groups, torch.arange(12, 16).expand(n, -1)), -1),
        _gather_entity_slots=lambda values, order: values.gather(
            1, order[..., None].expand(-1, -1, values.shape[-1])),
    )
    task._box_size[shared, 0] = torch.tensor([7.2, 1.6, 1.5])
    task._logical_box_values = lambda values, ids: task._gather_entity_slots(
        values[ids], task._logical_box_order[ids])
    task._assigned_box_values = lambda values, ids: task._gather_entity_slots(
        values[ids], task._agent_box_assignment[ids])
    return task


def test_joint_and_solo_ranges_in_one_sampler_batch():
    torch.manual_seed(119)
    shared = torch.arange(2048) % 2 == 0
    task = task_for_sampling(shared)
    assert not sample_push(task, torch.arange(len(shared))).any()
    boxes = task._box_states.gather(1, task._agent_box_assignment[..., None].expand(-1, -1, 13))
    source = (boxes[..., :2] - task._humanoid_root_states[..., :2]).norm(dim=-1)
    goal = (task._tar_pos[..., :2] - boxes[..., :2]).norm(dim=-1)
    for distance, high in ((source, 10.), (goal, 5.)):
        assert (distance[shared] >= 2-1e-5).all() and (distance[shared] <= high+1e-5).all()
        assert (distance[shared] < 2.2).any() and (distance[shared] > high-.2).any()
        assert (distance[~shared] >= 2-1e-5).all() and (distance[~shared] <= 10+1e-5).all()


def test_solo_is_bitwise_identical_with_or_without_joint_override():
    shared = torch.zeros(64, dtype=torch.bool)
    current, legacy = task_for_sampling(shared), task_for_sampling(shared, explicit=False)
    for task in (current, legacy):
        torch.manual_seed(912)
        assert not sample_push(task, torch.arange(len(shared))).any()
    assert torch.equal(current._box_states, legacy._box_states)
    assert torch.equal(current._tar_pos, legacy._tar_pos)


def test_saved_legacy_config_keeps_original_shared_range_and_contract():
    env = native_config('stage2_coupled_push_region_goals')['env']
    legacy = deepcopy(env)
    legacy['stage2JointPush'].pop('placement_order', None)
    for key in ('source_distance', 'goal_distance'):
        legacy['stage2JointPush'].pop(key)
    for cfg in (env, legacy):
        validate_env(cfg)
        check_checkpoint({'coupled_push_contract': checkpoint_contract(cfg)}, cfg)
    with pytest.raises(ValueError, match='settings differ'):
        check_checkpoint({'coupled_push_contract': checkpoint_contract(legacy)}, env)
    task = task_for_sampling(torch.ones(128, dtype=torch.bool), explicit=False)
    torch.manual_seed(421)
    assert not sample_push(task, torch.arange(128)).any()
    source = (task._box_states[:, 0, :2] - task._humanoid_root_states[:, 0, :2]).norm(dim=-1)
    goal = (task._tar_pos[:, 0, :2] - task._box_states[:, 0, :2]).norm(dim=-1)
    assert (source >= 2-1e-5).all() and (goal >= 2-1e-5).all()


def test_new_config_only_changes_shared_distance_settings():
    original = native_config('stage2_coupled_push')['env']
    current = native_config('stage2_coupled_push_region_goals')['env']
    for key in ('source_distance', 'goal_distance'):
        assert current['stage2JointPush'].pop(key) == [2., 10. if key == 'source_distance' else 5.]
    assert 'placement_order' not in current['stage2JointPush']
    current['stage2JointPush'].pop('goal_representation')
    assert current == original


@pytest.mark.parametrize('count', [2, 3, 4])
@pytest.mark.parametrize('high', [5., 10.])
def test_box_first_people_fit_every_real_box_profile(count, high):
    from utils.stage1_jointpush import box_profiles
    from utils.coupled_push_scene import box_first_people, axes
    sizes=torch.tensor(box_profiles()); sizes[:,0]*=count
    n=len(sizes); torch.manual_seed(61)
    centers=torch.randn(n,3,2)*10; yaw=torch.rand(n,3)*6.28
    regions=torch.rand(n,count).argsort(-1)
    people,feasible=box_first_people(centers,yaw,sizes,regions,
        torch.full((n,),count),torch.full((n,),2.),torch.full((n,),high),.5,
        torch.rand(n,3,count))
    assert feasible.all()
    relative=people-centers[:,:,None]
    distance=relative.norm(dim=-1)
    assert ((distance>=2-1e-5)&(distance<=high+1e-5)).all()
    local=(relative.unsqueeze(-2)*axes(yaw)[:,:,None]).sum(-1)
    assert (local[...,1] <= -sizes[:,None,None,1]/2-.5).all()
    gap=(people[:,:,:,None]-people[:,:,None,:]).norm(dim=-1)
    mask=~torch.eye(count,dtype=torch.bool)
    assert (gap[...,mask]>=1.).all()


def test_people_first_sampling_keeps_humans_fixed():
    task = task_for_sampling(torch.ones(256, dtype=torch.bool))
    roots = task._humanoid_root_states.clone()
    bodies = task._kinematic_humanoid_rigid_body_states.clone()
    torch.manual_seed(24)
    assert not sample_push(task, torch.arange(256)).any()
    assert torch.equal(task._humanoid_root_states, roots)
    assert torch.equal(task._kinematic_humanoid_rigid_body_states, bodies)
    assert 'placement_order' not in task.cfg['env']['stage2JointPush']


def test_translation_preserves_pose_velocity_and_unselected_people():
    from utils.coupled_push_scene import translate_people
    torch.manual_seed(81)
    task=SimpleNamespace(_humanoid_root_states=torch.randn(8,4,13),
        _kinematic_humanoid_rigid_body_states=torch.randn(8,4,15,13))
    old_roots=task._humanoid_root_states.clone()
    old_bodies=task._kinematic_humanoid_rigid_body_states.clone()
    envs=torch.tensor([1,5]); mask=torch.tensor([[True,False,True,False],[False,True,False,True]])
    positions=torch.randn(2,4,2)*10
    translate_people(task,envs,mask,positions)
    row,owner=mask.nonzero(as_tuple=True); selected=envs[row]
    moved=torch.zeros(8,4,dtype=torch.bool); moved[selected,owner]=True
    assert torch.equal(task._humanoid_root_states[~moved],old_roots[~moved])
    assert torch.equal(task._kinematic_humanoid_rigid_body_states[~moved],old_bodies[~moved])
    assert torch.equal(task._humanoid_root_states[...,2:],old_roots[...,2:])
    assert torch.equal(task._kinematic_humanoid_rigid_body_states[...,2:],old_bodies[...,2:])
    before=old_bodies[...,:2]-old_roots[:,:,None,:2]
    after=task._kinematic_humanoid_rigid_body_states[...,:2]-task._humanoid_root_states[:,:,None,:2]
    torch.testing.assert_close(before,after,atol=2e-6,rtol=1e-5)
