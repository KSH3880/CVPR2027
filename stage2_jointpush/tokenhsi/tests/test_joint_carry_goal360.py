from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from env.tasks.multi_agent.joint_carry_reset import JointCarryReset
from utils.joint_carry_amp import check_ablation_checkpoint, validate_ablation
from utils.joint_carry_goals import GOAL360, invalid_goals, sample_goals, clear_terminal_goals
from utils.joint_carry_rsi import load_paired_snapshots, rotate_vectors
from utils.joint_carry_spec import validate_env
from utils.mixed_carry_spec import sample_graph
from test_joint_carry_amp_ablation import configs, ROOT


def heading(q):
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def fake_task(env, skill, preset, n=512):
    pools = load_paired_snapshots(ROOT.parent/'joint_carry/carry_joint_corrected', 'cpu')
    reference = pools['carryWith']
    bodies = reference['body_state'].shape[2]
    dofs = reference['dof_position'].shape[2]
    task = SimpleNamespace(cfg={'env':env}, device='cpu', num_envs=n, num_agents=2,
        num_objects=4, num_bodies=bodies, _skill=[skill], _is_eval=True,
        _skill_init_prob=torch.ones(1), _env_origins=torch.randn(n, 3)*10,
        _box_size=torch.tensor([[.52,.8,.4],[.52,.8,.4],[.52,.8,.3],[.52,.8,.4]]).expand(n,-1,-1),
        _humanoid_root_states=torch.zeros(n,2,13), _dof_pos=torch.zeros(n,2,dofs),
        _dof_vel=torch.zeros(n,2,dofs), _every_env_init_dof_pos=torch.zeros(n,2,dofs),
        _kinematic_humanoid_rigid_body_states=torch.zeros(n,2,bodies,13),
        _box_states=torch.zeros(n,4,13), _tar_pos=torch.zeros(n,2,3),
        _stage2_rsi_counts=torch.zeros(1),
        _sample_episode_graph=lambda ids:None, _sample_box_assignments=lambda ids:None,
        relation_runtime=SimpleNamespace(graph=sample_graph(n,env['relationGraph'],preset=preset)))
    task._env_origins[:,2] = 0
    reset = JointCarryReset(task)
    if skill == 'loco':
        reset.locomotion = lambda count: tuple(reference[key][:1].expand(count,*reference[key].shape[1:]).clone()
            for key in ('root_state','dof_position','dof_velocity','body_state'))
    return task, reset


def test_config_contract_and_backward_compatibility():
    base, base_train = configs('locoamp_ca4_task_embedding')
    env, train = configs('locoamp_ca4_goal360_task_embedding')
    validate_env(env); validate_ablation(env, train)
    restored = deepcopy(env)
    assert restored['jointCarryRsi'].pop('goalSampling') == GOAL360
    assert restored == base
    changed_train = deepcopy(train)
    changed_train['params']['config']['name'] = base_train['params']['config']['name']
    assert changed_train == base_train
    saved = {'joint_carry_ablation':env['jointCarryAblation'], 'joint_carry_goal_sampling':GOAL360}
    check_ablation_checkpoint(saved,env)
    with pytest.raises(ValueError,match='goal sampling'):
        check_ablation_checkpoint(saved,base)
    old = {'joint_carry_ablation':base['jointCarryAblation']}
    check_ablation_checkpoint(old,base)
    with pytest.raises(ValueError,match='goal sampling'):
        check_ablation_checkpoint(old,env)
    bad=deepcopy(env);bad['jointCarryRsi']['goalSampling']['distance']=[0,1]
    with pytest.raises(ValueError,match='goal sampling'):
        validate_env(bad)


@pytest.mark.parametrize('skill', ['loco','pickUp','carryWith','putDown'])
@pytest.mark.parametrize('preset',['joint_carry_at','joint_carry_ontop'])
def test_real_paired_reset_pose_velocity_partial_reset_and_goals(skill,preset):
    if skill == 'putDown' and preset == 'joint_carry_ontop':
        pytest.skip('Floor putDown is disabled for joint ON_TOP')
    base,_=configs('locoamp_ca4_task_embedding')
    env,_=configs('locoamp_ca4_goal360_task_embedding')
    torch.manual_seed(37)
    task,reset=fake_task(env,skill,preset)
    torch.manual_seed(37)
    legacy,old_reset=fake_task(base,skill,preset)
    ids=torch.arange(0,task.num_envs,2)
    before=task._humanoid_root_states[1::2].clone()
    torch.manual_seed(101);old_reset.reset(ids,finalize=False)
    torch.manual_seed(101);reset.reset(ids,finalize=False)
    for name in ('_humanoid_root_states','_dof_pos','_dof_vel','_kinematic_humanoid_rigid_body_states'):
        torch.testing.assert_close(getattr(task,name),getattr(legacy,name),rtol=0,atol=0)
    torch.testing.assert_close(task._box_states[:,0],legacy._box_states[:,0],rtol=0,atol=0)
    torch.testing.assert_close(task._humanoid_root_states[1::2],before,rtol=0,atol=0)
    torch.testing.assert_close(task._tar_pos[ids,0],task._tar_pos[ids,1],rtol=0,atol=0)
    displacement=task._tar_pos[ids,0,:2]-task._box_states[ids,0,:2]
    if skill=='putDown':
        pool=reset.pools[skill]
        row=torch.searchsorted(pool['source_frame'],reset.source_frame[ids])
        yaw=heading(task._humanoid_root_states[ids,0,3:7])-heading(pool['root_state'][row,0,3:7])
        with np.load(ROOT.parent/'joint_carry/carry_joint_corrected/putDown/cooperative_rsi_candidate.npz') as data:
            expected=torch.tensor(data['box_position'][-1,:2],dtype=torch.float)-pool['box_state'][row,0,:2]
        expected=rotate_vectors(torch.nn.functional.pad(expected,(0,1)),yaw)[:,:2]
        torch.testing.assert_close(displacement,expected,atol=4e-6,rtol=1e-5)
        radius=task._box_size[ids,:,:2].norm(dim=-1)/2
        obstacle_distance=(task._tar_pos[ids,0,None,:2]-task._box_states[ids,1:,:2]).norm(dim=-1)
        assert (obstacle_distance >= radius[:,:1]+radius[:,1:]+.1-1e-5).all()
    else:
        distance=displacement.norm(dim=-1)
        assert distance.min()>2.2-1e-5 and distance.max()<2.8+1e-5
        # Undo the common scene yaw using the unchanged support orientation.
        yaw=heading(task._box_states[ids,2,3:7])
        relative=rotate_vectors(torch.nn.functional.pad(displacement,(0,1)),-yaw)
        bins=((torch.atan2(relative[:,1],relative[:,0])+torch.pi)/(2*torch.pi)*8).long().clamp_max(7)
        assert torch.bincount(bins,minlength=8).min()>0
    if preset=='joint_carry_ontop':
        torch.testing.assert_close(task._box_states[ids,2,:2],task._tar_pos[ids,0,:2])
        torch.testing.assert_close(task._box_states[ids,2,2],torch.full((len(ids),),.15))
        assert task._box_states[ids,2,7:].count_nonzero()==0


def test_sampler_collision_rejection_and_empty_batch():
    env,_=configs('locoamp_ca4_goal360_task_embedding')
    task,_=fake_task(env,'carryWith','joint_carry_ontop',1024)
    boxes=task._box_states.clone();boxes[...,6]=1
    boxes[:,1,0]=2.5;boxes[:,3,:2]=torch.tensor([-3.,2.])
    bodies=task._kinematic_humanoid_rigid_body_states[...,:3].clone()
    bodies[:,:,0,2]=1
    ontop=torch.ones(1024,dtype=torch.bool)
    xy=sample_goals(boxes,task._box_size,bodies,ontop,GOAL360)
    assert not invalid_goals(xy,boxes,task._box_size,bodies,ontop,.1).any()
    assert invalid_goals(torch.tensor([[2.5,0.]]).expand(1024,-1),boxes,task._box_size,bodies,ontop,.1).all()
    assert sample_goals(boxes[:0],task._box_size[:0],bodies[:0],ontop[:0],GOAL360).shape==(0,2)


def test_all_putdown_frames_keep_terminal_goal_clear():
    pool=load_paired_snapshots(ROOT.parent/'joint_carry/carry_joint_corrected','cpu',True)['putDown']
    xy=pool['terminal_box_xy']-pool['box_state'][:,0,:2]
    n=len(xy)
    boxes=torch.zeros(n,4,13);boxes[...,6]=1
    boxes[:,1,:2]=torch.tensor([-3.,-2.]);boxes[:,2,:2]=torch.tensor([0.,3.5]);boxes[:,3,:2]=torch.tensor([-3.,2.])
    before=boxes.clone()
    sizes=torch.tensor([.52,.8,.4]).expand(n,4,3)
    moved=clear_terminal_goals(boxes,sizes,xy,.1)
    torch.testing.assert_close(moved[:,0],before[:,0],rtol=0,atol=0)
    radius=sizes[...,:2].norm(dim=-1)/2
    assert ((xy[:,None]-moved[:,1:,:2]).norm(dim=-1)>=radius[:,:1]+radius[:,1:]+.1-1e-5).all()
    for a in range(4):
        for b in range(a):
            assert ((moved[:,a,:2]-moved[:,b,:2]).norm(dim=-1)>=radius[:,a]+radius[:,b]+.1).all()
    assert (moved[...,:2].norm(dim=-1)+radius<=5.).all()
    body=pool['body_state'][...,:3].clone()
    body[...,:2]-=pool['box_state'][:,0,None,None,:2]
    moved[:,1:,2]=sizes[:,1:,2]/2
    relative=body[:,:,None]-moved[:,None,1:,None,:3]
    assert not (relative.abs()<sizes[:,None,1:,None]/2+.1).all(-1).any()
