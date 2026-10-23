"""Force-enabled midpoint CONTACT: thresholds, direction, runtime and contracts."""
from isaacgym import gymapi
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from utils.push_point import local_points, world_points, midpoint_force_contact
from utils.stage1_push import validate_env, check_checkpoint
from utils.edge_scenario_spec import compose_canonical_graph
from utils.edge_context_spec import HOLDING, AT
from env.tasks.multi_agent.humanoid_ma_stage1_push import PushRuntime


@pytest.mark.parametrize('name', ['stage1_push_midpoint_safe', 'stage1_push_depth180_safe', 'stage1_push_midpoint_safe_at5deg'])
def test_defaults_and_checkpoint_separation(name):
    env=yaml.safe_load(Path(f'tokenhsi/data/cfg/multi_agent/{name}.yaml').read_text())['env']
    validate_env(env)
    assert env['pushTask']['require_contact_force'] is True
    assert env['pushTask']['min_hand_force_n']==env['pushTask']['min_box_force_n']==.5
    old=deepcopy(env);old['pushTask']['require_contact_force']=False
    for key in ('min_hand_force_n','min_box_force_n'):old['pushTask'].pop(key)
    validate_env(old)
    with pytest.raises(ValueError):check_checkpoint({'relation_experiment_config':{'env':old}},env)
    check_checkpoint({'relation_experiment_config':{'env':env}},env)
    for bad in [None,False,0,-1,float('nan'),float('inf')]:
        invalid=deepcopy(env);invalid['pushTask']['min_hand_force_n']=bad
        with pytest.raises(ValueError,match='force'):validate_env(invalid)


@pytest.mark.parametrize('hand1,hand2,box,offset,passed', [
    (.5,.5,.5,0,True),(.499,.5,.5,0,False),(.5,.499,.5,0,False),
    (.5,.5,.499,0,False),(0,0,0,0,False),(-1,-1,1,0,False),
    (1,1,-1,0,False),(1,1,1,.02,False)])
@pytest.mark.parametrize('yaw',[False,True])
def test_same_step_geometry_and_normal_force(hand1,hand2,box,offset,passed,yaw):
    boxes=torch.zeros(1,1,13);boxes[...,2]=.65;boxes[...,6]=1
    side=torch.full((1,1),2);size=torch.full((1,1,3),1.3)
    local=local_points(side,size,torch.full((1,1),.65))
    if yaw:boxes[...,3:7]=torch.tensor([0.,0.,2**-.5,2**-.5])
    hands=world_points(boxes,local)[...,None,:].expand(-1,-1,2,-1).clone()
    hands[...,2]+=offset
    # Inward normal on -Y face, rotated by yaw when requested.
    normal=torch.tensor([-1.,0.,0.] if yaw else [0.,1.,0.])
    forces=-torch.tensor([hand1,hand2])[None,None,:,None]*normal
    box_forces=box*normal[None,None,:]
    result=midpoint_force_contact(hands,forces,boxes,box_forces,size,side,local,
        {'z_reward_sigma_m':.04},{'min_hand_force_n':.5,'min_box_force_n':.5})
    assert result.item()==passed


def test_runtime_uses_measured_hits_partial_reset_and_keeps_at_saturation():
    env=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_midpoint_safe.yaml').read_text())['env']
    graph=compose_canonical_graph(torch.full((2,2),3))
    task=SimpleNamespace(cfg={'env':env},_push_hits=torch.tensor([[False,True],[True,False]]))
    runtime=PushRuntime(task,2,graph,env['relationReward'],'cpu')
    hold=graph.edge_relation==HOLDING;at=graph.edge_relation==AT
    phi=torch.ones(2,4);phi[at]=0;z=torch.zeros_like(phi)
    result=runtime.step(phi,z,z,z,z)
    expected=task._push_hits.gather(1,graph.edge_owner)
    assert torch.equal(result['own_success'][hold],expected[hold])
    runtime.reset(torch.tensor([1]),phi[1:],z[1:],z[1:],z[1:])
    assert torch.equal(runtime.own_success[1,hold[1]],expected[1,hold[1]])
    task._push_hits.zero_();phi[at]=1
    result=runtime.step(phi,z,z,z,z)
    assert not result['own_success'][hold].any()
    assert result['reward_saturated'].all()


@pytest.mark.parametrize('synchronized',[False,True])
def test_physics_substep_requires_simultaneous_position_and_force(synchronized):
    from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
    boxes=torch.zeros(1,2,13);boxes[...,2]=.65;boxes[...,6]=1
    size=torch.full((1,2,3),1.3);side=torch.full((1,2),2)
    local=local_points(side,size,torch.full((1,2),.65))
    target=world_points(boxes,local)[...,None,:].expand(-1,-1,2,-1).clone()
    task=SimpleNamespace(
        cfg={'env':{'pushTask':{'contact_objective':'midpoint_only_v1','require_contact_force':True,
                              'min_hand_force_n':.5,'min_box_force_n':.5},
                    'pushPointReward':{'z_reward_sigma_m':.04}}},
        control_freq_inv=2,sim=None,_box_states=boxes,_box_size=size,
        _push_side=side,_push_local=local,_key_body_ids=torch.tensor([0,1]),
        _push_hits=torch.zeros(1,2,dtype=torch.bool),render=lambda:None)
    counter=[0]
    def refresh():
        i=counter[0];counter[0]+=1
        task._rigid_body_pos=target.clone()
        if i==1:task._rigid_body_pos[...,2]+=.1
        active=synchronized if i==0 else True
        task._contact_forces=torch.zeros_like(target);task._contact_forces[...,1]=-float(active)
        task._ontop_box_contact=torch.zeros(1,2,3);task._ontop_box_contact[...,1]=float(active)
    task._refresh_sim_tensors=refresh
    task.gym=SimpleNamespace(simulate=lambda sim:None,fetch_results=lambda sim,wait:None)
    HumanoidMAStage1Push._physics_step(task)
    assert task._push_hits.all().item()==synchronized
