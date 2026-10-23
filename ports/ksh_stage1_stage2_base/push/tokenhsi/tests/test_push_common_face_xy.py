"""Dynamic finite common-face geometry and Stage1 CONTACT integration."""
from isaacgym import gymapi
from pathlib import Path
from types import SimpleNamespace
import math
import pytest
import torch
import yaml
from utils.push_point import common_face_xy, common_face_state, common_face_contact
from utils.stage1_push import validate_env, check_checkpoint
from utils.edge_scenario_spec import compose_canonical_graph
from utils.edge_context_spec import HOLDING, AT
from env.tasks.multi_agent.humanoid_ma_stage1_push import PushRuntime, HumanoidMAStage1Push


def scene():
    boxes=torch.zeros(1,2,13);boxes[...,2]=.65;boxes[...,6]=1
    size=torch.ones(1,2,3)*1.3
    hands=torch.tensor([[[[-.3,-.65,8.],[.3,-.65,-8.]]]]).expand(1,2,2,3).clone()
    goals=torch.tensor([[[0.,3.,.65],[0.,3.,.65]]])
    return hands,boxes,size,goals


def test_no_midpoint_or_height_penalty_and_finite_edge():
    h,b,s,g=scene()
    assert torch.equal(common_face_state(h,b,s,g),torch.ones(1,2))
    h[:,:,1,0]=.95
    assert torch.allclose(common_face_xy(h,b,s,g)[0][...,1],torch.ones(1,2)*.3)
    assert torch.allclose(common_face_state(h,b,s,g),torch.ones(1,2)*math.exp(-.9))


def test_max_hand_and_same_face_not_independent_faces():
    h,b,s,g=scene();g[...,:2]=torch.tensor([3.,3.])
    h[:,:,0,:2]=torch.tensor([0.,-.65]);h[:,:,1,:2]=torch.tensor([-.65,0.])
    d,_,_=common_face_xy(h,b,s,g)
    assert torch.allclose(d.amax(-1),torch.ones(1,2)*.65)


def test_goal_changes_reselect_face_and_60_degree_boundary():
    h,b,s,g=scene();h[...,:2]=torch.tensor([-.65,0.])
    g[...,:2]=torch.tensor([math.cos(math.pi/3),math.sin(math.pi/3)])
    assert (common_face_xy(h,b,s,g)[2]==0).all()
    g[...,:2]=torch.tensor([math.cos(math.pi/3+.01),math.sin(math.pi/3+.01)])
    assert (common_face_xy(h,b,s,g)[2]==2).all()
    g[...,:2]=torch.tensor([-3.,0.])
    assert (common_face_xy(h,b,s,g)[2]==1).all()


def test_yaw_rotation_and_degenerate_projection_are_finite():
    h,b,s,g=scene();original=common_face_state(h,b,s,g)
    hxy=h[...,:2].clone();h[...,0]=-hxy[...,1];h[...,1]=hxy[...,0]
    b[...,3:7]=torch.tensor([0.,0.,2**-.5,2**-.5]);g[...,:2]=torch.tensor([-3.,0.])
    assert torch.allclose(common_face_state(h,b,s,g),original,atol=1e-6)
    b[...,3:7]=torch.tensor([2**-.5,0.,0.,2**-.5])
    assert torch.isfinite(common_face_state(h,b,s,g)).all()


@pytest.mark.parametrize('offset,force,expected',[(.0,.5,True),(.0,.49,False),(.2,.5,False)])
def test_success_uses_same_face_state_and_forces(offset,force,expected):
    h,b,s,g=scene();h[...,1]-=offset
    f=torch.zeros_like(h);f[...,1]=-force
    bf=torch.zeros(1,2,3);bf[...,1]=force
    spec=dict(require_contact_force=True,min_hand_force_n=.5,min_box_force_n=.5)
    assert (common_face_contact(h,f,b,bf,s,g,spec)==expected).all()


@pytest.mark.parametrize('base',['stage1_push_depth180_h080_rsi1m_phase','stage1_push_depth180_h100_rsi1m_at5deg_phase'])
def test_new_contract_and_runtime(base):
    folder=Path('tokenhsi/data/cfg/multi_agent')
    env=yaml.safe_load((folder/(base+'_face_xy.yaml')).read_text())['env']
    validate_env(env)
    previous=yaml.safe_load((folder/(base+'.yaml')).read_text())['env']
    with pytest.raises(ValueError,match='checkpoint objective/config differs'):
        check_checkpoint({'relation_experiment_config':{'env':previous}},env)
    graph=compose_canonical_graph(torch.full((1,2),3))
    task=SimpleNamespace(cfg={'env':env},_push_hits=torch.tensor([[False,True]]),_box_states=scene()[1])
    runtime=PushRuntime(task,1,graph,env['relationReward'],'cpu')
    phi=torch.ones(1,4);phi[graph.edge_relation==AT]=0;z=torch.zeros_like(phi)
    result=runtime.step(phi,z,z,z,z)
    holding=graph.edge_relation==HOLDING
    assert torch.equal(result['own_success'][holding],task._push_hits.gather(1,graph.edge_owner)[holding])
    task._push_hits.zero_();phi[:]=1
    result=runtime.step(phi,z,z,z,z)
    assert not result['own_success'][holding].any()
    assert result['reward_saturated'].all()


def test_no_eligible_face_has_zero_state():
    h,b,s,g=scene()
    b[...,3:7]=torch.tensor([0.,2**-.5,0.,2**-.5])
    g[...,:2]=torch.tensor([3.,0.])
    assert (common_face_state(h,b,s,g)==0).all()
