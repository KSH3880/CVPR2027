import math
from pathlib import Path
import pytest
import torch
import yaml
from env.tasks.multi_agent.joint_push_reward import DEFAULT_CONFIG, joint_push_reward, push_face_geometry


def scene():
    boxes=torch.zeros(1,13);boxes[:,2]=1.3;boxes[:,6]=1
    goals=torch.tensor([[0.,3.,1.3]])
    roots=torch.zeros(1,2,13);roots[...,6]=1;roots[...,1]=-1.2
    center,_,tangent,_,_=push_face_geometry(boxes,goals,[2.6,2.6,2.6],DEFAULT_CONFIG)
    anchors=center[:,None].expand(-1,2,-1).clone()
    anchors[...,:2]+=torch.tensor([-0.65,0.65])[None,:,None]*tangent[:,None]
    hands=anchors[:, :, None].expand(-1,-1,2,-1).clone()
    return roots,hands,boxes,goals


def test_symmetric_real_face_and_swap():
    roots,hands,boxes,goals=scene();r=joint_push_reward(roots,hands,boxes,goals,[2.6,2.6,2.6])
    torch.testing.assert_close(r['hand_midpoints'].mean(1),torch.tensor([[0.,-1.3,.8]]))
    torch.testing.assert_close(r['minimum_separation'],torch.tensor([.65]))
    assert r['coupled_contact'].all()
    s=joint_push_reward(roots[:,[1,0]],hands[:,[1,0]],boxes,goals,[2.6,2.6,2.6])
    torch.testing.assert_close(s['agent_task_reward'],r['agent_task_reward'][:,[1,0]])
    assert torch.equal(s['coupled_contact'],r['coupled_contact'])


def test_same_anchor_no_gate_and_gate_recloses():
    roots,hands,boxes,goals=scene();goals=boxes[:,:3].clone()
    r=joint_push_reward(roots,hands,boxes,goals,[2.6,2.6,2.6])
    assert r['done'].all()
    hands[:,1]=hands[:,0]
    r=joint_push_reward(roots,hands,boxes,goals,[2.6,2.6,2.6])
    assert not r['coupled_contact'].any() and not r['done'].any()
    assert (r['agent_task_reward']>0).all()
    assert (r['total'][...,1]==0).all()
    hands+=10
    r=joint_push_reward(roots,hands,boxes,goals,[2.6,2.6,2.6])
    assert (r['phi_raw'][...,0]<1).all() # placement never saturates contact


def test_components_goal_shared_and_height_tolerance():
    roots,hands,boxes,goals=scene();r=joint_push_reward(roots,hands,boxes,goals,[2.6,2.6,2.6])
    torch.testing.assert_close(r['total'],r['state_component']+r['progress_component']+r['success_component'])
    torch.testing.assert_close(r['total'][:,0,1],r['total'][:,1,1])
    goals=boxes[:,:3].clone();goals[:,2]+=.002
    assert not joint_push_reward(roots,hands,boxes,goals,[2.6,2.6,2.6])['done'].any()


def test_rigid_rotation_translation_and_goal_faces():
    roots,hands,boxes,goals=scene();base=joint_push_reward(roots,hands,boxes,goals,[2.6,2.6,2.6])
    theta=.8;c,s=math.cos(theta),math.sin(theta);rot=torch.tensor([[c,-s],[s,c]])
    shift=torch.tensor([3.,-2.,.4])
    for x in (roots,hands,boxes,goals):
        x[...,:2]=x[...,:2]@rot.T;x[...,:3]+=shift
    boxes[:,3:7]=torch.tensor([0.,0.,math.sin(theta/2),math.cos(theta/2)])
    r=joint_push_reward(roots,hands,boxes,goals,[2.6,2.6,2.6])
    torch.testing.assert_close(r['agent_task_reward'],base['agent_task_reward'],atol=1e-6,rtol=1e-5)
    expected=base['face_center'].clone();expected[...,:2]=expected[...,:2]@rot.T;expected+=shift
    torch.testing.assert_close(r['face_center'],expected)
    boxes.zero_();boxes[:,6]=1
    for g in ([2.,0.,0.],[-2.,0.,0.],[0.,-2.,0.],[0.,0.,0.]):
        center,normal,_,_,_=push_face_geometry(boxes,torch.tensor([g]),[2.6,2.6,2.6],DEFAULT_CONFIG)
        assert torch.isfinite(center).all()
        torch.testing.assert_close(center[0,:2],-normal[0]*torch.tensor([1.3,1.3]))


def test_yaml_contract_and_invalid_inputs():
    config=yaml.safe_load((Path(__file__).parents[1]/'data/cfg/multi_agent/joint_push_reward.yaml').read_text())
    assert config==DEFAULT_CONFIG
    roots,hands,boxes,goals=scene()
    with pytest.raises(ValueError):joint_push_reward(roots[:,:1],hands,boxes,goals,[2.6,2.6,2.6])
    with pytest.raises(ValueError):joint_push_reward(roots,hands,boxes,goals,[-1.,1.,1.])


def test_free_symmetric_spacing_and_reject_center_asymmetry_outside():
    roots,hands,boxes,goals=scene()
    for offset in (.4,.65,1.0):
        hands[:,:, :,0]=torch.tensor([-offset,offset])[None,:,None]
        r=joint_push_reward(roots,hands,boxes,goals,[2.6,2.6,2.6])
        assert r['coupled_contact'].all()
        torch.testing.assert_close(r['phi_raw'][...,0],torch.ones(1,2))
    for offsets in ((0.,0.),(-.2,.2),(.4,.8),(-.4,.9),(-1.6,1.6)):
        hands[:,:,:,0]=torch.tensor(offsets)[None,:,None]
        assert not joint_push_reward(roots,hands,boxes,goals,[2.6,2.6,2.6])['coupled_contact'].any()


def test_box_width_mass_density_and_scaled_separation():
    path=Path(__file__).parents[1]/'data/cfg/multi_agent/joint_push_scene.yaml'
    config=yaml.safe_load(path.read_text());spec=config['pushBox']
    assert spec['size']==[2.6,2.6,2.6] and spec['mass']==40.
    assert spec['mass']/math.prod(spec['size'])==pytest.approx(40/2.6**3)
    roots,hands,boxes,goals=scene()
    for width in (1.3,2.6,3.9):
        r=joint_push_reward(roots,hands,boxes,goals,[width,2.6,2.6])
        torch.testing.assert_close(r['minimum_separation'],torch.tensor([width*.25]))
