from isaacgym import gymapi
from copy import deepcopy
from pathlib import Path
import pytest
import torch
import yaml
from utils.door_hold import closing_torque, hand_support_contact, update_hold, validate_hold, support_with_grace
from utils.edge_door_spec import compile_graph, validate_env
from env.tasks.multi_agent.humanoid_ma_door_hold import DoorHoldRuntime

ROOT=Path(__file__).resolve().parents[1]


def cfg():
    return yaml.safe_load((ROOT/'data/cfg/multi_agent/door_open_hold_amp.yaml').read_text())['env']


def test_single_person_config_and_old_configs_validate():
    e=cfg();validate_hold(e);validate_env(e)
    assert compile_graph(e['relationGraph'],1,1).ids==('open_0',)
    assert e['numEnvs']==2048 and e['skillInitProb']==[1,0]
    for name in ['door_open_bones_left_amp','door_open_unified_approach_amp','door_pull_unified_approach_amp']:
        validate_env(yaml.safe_load((ROOT/f'data/cfg/multi_agent/{name}.yaml').read_text())['env'])
    bad=deepcopy(e);bad['doorHold']['duration']=0
    with pytest.raises(ValueError):validate_hold(bad)


def test_spring_opposes_opening_and_damps_motion_without_policy_door_action():
    spring=cfg()['door']['autoClose']
    values=closing_torque(torch.tensor([0.,1.,1.,10.]),torch.tensor([0.,0.,-2.,10.]),spring)
    torch.testing.assert_close(values,torch.tensor([0.,-2.,-1.,-4.]))


def test_preloaded_door_resists_first_touch_and_rejects_excess_preload():
    env=cfg()
    env['door']['autoClose']=dict(stiffness=16.,damping=4.,max_torque=32.,preload_torque=4.)
    validate_hold(env)
    torque=closing_torque(torch.tensor([0.,.01,1.57]),torch.tensor([0.,.5,5.]),env['door']['autoClose'])
    assert torque[0]==-4. and torque[1]<-4. and torque[2]==-32.
    env['door']['autoClose']['preload_torque']=33.
    with pytest.raises(ValueError):validate_hold(env)


def test_contact_requires_own_hand_near_handle_and_push_reaction():
    spec=cfg()['doorHold'];handles=torch.zeros(1,1,3);hands=torch.zeros(1,1,2,3)
    hand_forces=torch.tensor([[[[0.,-2.,0.],[0.,0.,0.]]]])
    leaf=torch.tensor([[[0.,2.,0.]]]);angles=torch.zeros(1,1)
    assert hand_support_contact(hands,handles,hand_forces,leaf,angles,spec).item()
    hands[:]=1
    assert not hand_support_contact(hands,handles,hand_forces,leaf,angles,spec).item()
    hands[:]=0
    assert not hand_support_contact(hands,handles,-hand_forces,leaf,angles,spec).item()
    assert not hand_support_contact(hands,handles,hand_forces,torch.zeros_like(leaf),angles,spec).item()


def test_continuous_hold_resets_on_release_or_angle_drop_and_partial_reset():
    e=cfg();runtime=DoorHoldRuntime(2,compile_graph(e['relationGraph'],1,1),e['relationReward'],'cpu',e['doorHold'],1.)
    roots=torch.zeros(2,1,3);handles=torch.zeros(2,1,3);hands=torch.zeros(2,1,2,3);angles=torch.full((2,1),1.6)
    runtime.contact[:]=True
    runtime.posture[:]=True
    for _ in range(4):
        result=runtime.step(roots,hands,handles,angles)
        assert not runtime.done.any() and not result['success_component'].any()
    runtime.step(roots,hands,handles,angles)
    assert runtime.done.all()
    runtime.contact[0]=False;angles[1]=1.
    runtime.step(roots,hands,handles,angles)
    assert not runtime.elapsed.any() and not runtime.done.any()
    runtime.elapsed[:]=3;runtime.reset(torch.tensor([0]),angles[:1])
    assert runtime.elapsed[0]==0 and runtime.elapsed[1]==3


def test_angle_only_never_succeeds_even_when_open_for_a_long_time():
    elapsed=torch.zeros(1,1)
    for _ in range(100):
        elapsed,success=update_hold(elapsed,torch.tensor([[1.7]]),torch.tensor([[False]]),1.15,5.,.1)
        assert not success.any() and not elapsed.any()


def test_contact_grace_requires_prior_hit_and_expires_or_clears_on_hand_release():
    age=torch.tensor([[.2]]);near=torch.tensor([[True]]);hit=torch.tensor([[False]])
    age,contact=support_with_grace(age,hit,near,1/30,.1)
    assert not contact.any()
    age,contact=support_with_grace(age,~hit,near,1/30,.1)
    assert contact.all()
    for _ in range(3):
        age,contact=support_with_grace(age,hit,near,1/30,.1)
        assert contact.all()
    age,contact=support_with_grace(age,hit,near,1/30,.1)
    assert not contact.any()
    age,contact=support_with_grace(age,~hit,near,1/30,.1)
    age,contact=support_with_grace(age,hit,~near,1/30,.1)
    assert not contact.any()


def test_standing_posture_requires_height_upright_and_one_grounded_supporting_foot():
    import math
    from utils.door_hold import standing_posture
    spec=cfg()['doorHold']['posture']
    height=torch.tensor([[1.]])
    quat=torch.tensor([[[0.,0.,0.,1.]]])
    feet=torch.zeros(1,1,2,3);feet[...,2]=.12
    forces=torch.zeros_like(feet);forces[...,0,2]=10
    def valid():return standing_posture(height,quat,feet,forces,torch.ones_like(height),spec)
    assert valid().all()  # One foot supports; no requirement for two feet.
    height[:]=.7;assert valid().all()
    height[:]=.69;assert not valid().any();height[:]=1
    quat[...,0]=math.sin(math.radians(41)/2);quat[...,3]=math.cos(math.radians(41)/2)
    assert not valid().any()
    quat[...,0]=math.sin(math.radians(39)/2);quat[...,3]=math.cos(math.radians(39)/2)
    assert valid().all()
    forces.zero_();assert not valid().any()
    forces[...,0,2]=10;feet[...,0,2]=.4
    assert not valid().any()  # A force on a raised foot is not ground support.


def test_single_and_joint_door_require_90_degrees_and_continuous_valid_posture():
    import math
    from env.tasks.multi_agent.open_push_reward import OpenPushRuntime
    for joint in (False,True):
        e=cfg() if not joint else yaml.safe_load((ROOT/'data/cfg/multi_agent/open_push_unified_amp.yaml').read_text())['env']
        c=e['relationReward'];assert c['door']['target_angle']==math.pi/2 and c['door']['angle_tolerance']==0
        if joint:
            rt=OpenPushRuntime(1,e['relationGraph'],c,e['doorHold'],1.,'cpu',e['pushBox']['size'])
            boxes=torch.zeros(1,2,13);boxes[...,2]=.55;boxes[...,6]=1
            goals=boxes[...,:3].clone();goals[...,1]=1.5
            rt.reset(torch.tensor([0]),torch.zeros(1,2,dtype=torch.long),boxes)
            def step():return rt.step(roots,hands,handles,angles,boxes,goals)
        else:
            rt=DoorHoldRuntime(1,compile_graph(e['relationGraph'],1,1),c,'cpu',e['doorHold'],1.)
            def step():return rt.step(roots,hands,handles,angles)
        roots=torch.zeros(1,e['numAgents'],3);handles=torch.zeros_like(roots)
        hands=torch.zeros(1,e['numAgents'],2,3);angles=torch.full((1,e['numAgents']),math.radians(89.99))
        rt.contact[:]=True;rt.posture[:]=True
        for _ in range(6):step()
        assert not rt.done.any() and not rt.elapsed.any()
        angles[:]=math.pi/2
        for _ in range(4):step()
        assert (rt.elapsed==4).all() and not rt.done.any()
        rt.posture[:]=False;bad=step()
        assert not rt.elapsed.any() and not rt.done.any()
        interaction=bad['interaction_component'] if not joint else bad['state']-c['state_reward_weight']*.5
        torch.testing.assert_close(interaction,torch.zeros_like(interaction))
        progress=bad['progress_component'] if not joint else bad['progress']
        assert not progress.any()
        rt.posture[:]=True
        for _ in range(5):step()
        assert rt.done.all()
        rt.posture[:]=False;step()
        assert not rt.done.any() and not rt.elapsed.any()


def test_joint_push_success_is_independent_of_door_posture_guard():
    from env.tasks.multi_agent.open_push_reward import OpenPushRuntime
    e=yaml.safe_load((ROOT/'data/cfg/multi_agent/open_push_unified_amp.yaml').read_text())['env']
    rt=OpenPushRuntime(1,e['relationGraph'],e['relationReward'],e['doorHold'],1.,'cpu',e['pushBox']['size'])
    boxes=torch.zeros(1,2,13);boxes[...,2]=.55;boxes[...,6]=1
    rt.reset(torch.tensor([0]),torch.ones(1,2,dtype=torch.long),boxes)
    goals=boxes[...,:3].clone();goals[...,1]=1.5
    boxes[...,:3]=goals;rt.push_eligible[:]=True;rt.posture[:]=False
    roots=goals.clone();hands=goals[:,:,None].expand(1,2,2,3).clone()
    rt.step(roots,hands,roots,torch.zeros(1,2),boxes,goals)
    assert rt.done.all()


def test_foot_support_collects_early_physics_impulse_but_not_raised_foot():
    from types import SimpleNamespace
    from utils.door_hold import collect_foot_support,current_posture
    positions=torch.zeros(1,1,4,3);positions[...,0,2]=1.;positions[...,2:,2]=.12
    rotations=torch.zeros(1,1,4,4);rotations[...,3]=1.
    forces=torch.zeros_like(positions);forces[...,2,2]=10.
    task=SimpleNamespace(_posture_body_ids=[0,1,2,3],_posture_foot_forces=torch.zeros(1,1,2,3),
        _rigid_body_pos=positions,_rigid_body_rot=rotations,_contact_forces=forces,
        _standing_pelvis_height=torch.ones(1,1),_hold_spec=cfg()['doorHold'])
    collect_foot_support(task);forces.zero_();collect_foot_support(task)
    assert current_posture(task).all()  # Early impulse survives final physics tick.
    task._posture_foot_forces.zero_();collect_foot_support(task)
    assert not current_posture(task).any()  # Not carried into next control step.
    positions[...,2,2]=.5;forces[...,2,2]=10;collect_foot_support(task)
    assert not current_posture(task).any()
