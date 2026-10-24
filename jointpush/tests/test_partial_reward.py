"""Physical-group reward regressions against the frozen RawCuboid baseline."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'jointpush/ports/jhh_raw_cuboid/tokenhsi'))
import isaacgym
import torch
from env.tasks.multi_agent.push_reward import contact_geometry
from utils.push_navigation import approach_distance as baseline_approach
from jointpush.partial.reward import components,approach_distance,rotate

SPEC=dict(contact_midpoint_only=True,contact_height=1.04,face_tolerance=.18,approach_stand_off=1.,clearance=.45)

def fixture():
    boxes=torch.zeros(1,2,13);boxes[...,6]=1;boxes[0,0,:3]=torch.tensor([0.,0.,.8]);boxes[0,1,:3]=torch.tensor([8.,0.,.8])
    sizes=torch.tensor([[[2.8,1.6,1.6],[2.8,1.6,1.6]]]);assignment=torch.tensor([[0,0,1,1]])
    local=torch.zeros(1,4,2,3);local[...,0]=torch.tensor([-.7,.7,-.7,.7])[None,:,None];local[...,0,0]+=.15;local[...,1,0]-=.15;local[...,1]=-.8;local[...,2]=.24
    box=boxes[:,assignment[0]];hands=box[...,None,:3]+local
    roots=torch.zeros(1,4,13);roots[...,6]=1;roots[...,:3]=box[...,:3];roots[...,1]=-2.
    goals=box[...,:3].clone();goals[...,1]=3.
    hand_force=torch.zeros_like(hands);hand_force[...,1]=-1
    box_force=torch.zeros(1,2,3);box_force[...,1]=2
    return dict(roots=roots,hands=hands,hand_force=hand_force,boxes=boxes,sizes=sizes,box_force=box_force,assignment=assignment,local_points=local,goals=goals,spec=SPEC,contact_geometry=contact_geometry)

def test_all_hands_gate_is_group_local():
    x=fixture();a=components(**x);assert a['group_gate'].all()
    x['hands'][0,0,1,1]=.8
    b=components(**x)
    assert b['group_gate'].tolist()==[[False,False,True,True]]
    assert not b['progress'][0,:2,1].any() and (b['progress'][0,2:,1]>0).all()
    assert (b['progress'][... ,0]>0).all()  # keep approach learnable

def test_at_uses_physical_center_and_is_region_invariant():
    x=fixture();a=components(**x)
    assert torch.allclose(a['raw_at_progress'],torch.full((1,4),1/3.9))
    x['local_points'][... ,0]*=-1
    b=components(**x)
    assert torch.equal(a['raw_at_progress'],b['raw_at_progress'])
    x['goals'][...,1]=1.
    assert (components(**x)['raw_at_progress']>a['raw_at_progress']).all()

def test_goal_center_success_despite_offset_regions():
    x=fixture();x['goals']=x['boxes'][:,x['assignment'][0],:3].clone()
    r=components(**x);assert r['done'].all()
    x['hands'][0,0,0,1]=.8
    assert components(**x)['done'].tolist()==[[False,False,True,True]]

def test_no_force_is_not_contact_success():
    x=fixture();x['hand_force'].zero_();x['box_force'].zero_()
    r=components(**x);assert not r['contact'].any();assert (r['phi'][...,0]>.99).all()

def test_wrong_goal_direction_closes_gate():
    x=fixture();x['goals'][...,1]=-3.
    r=components(**x);assert not r['group_gate'].any();assert not r['state'][...,1].any()

def test_rotated_physical_frame():
    x=fixture();q=torch.tensor([0.,0.,2**-.5,2**-.5])
    for key in ('hands','hand_force'):
        x[key]=rotate(q.expand(*x[key].shape[:-1],4),x[key])
    for key in ('boxes','roots'):
        x[key][...,:3]=rotate(q.expand(*x[key].shape[:-1],4),x[key][...,:3]);x[key][...,3:7]=q
    x['goals']=rotate(q.expand(1,4,4),x['goals']);x['box_force']=rotate(q.expand(1,2,4),x['box_force'])
    r=components(**x);assert r['group_gate'].all();assert torch.allclose(r['raw_at_progress'],torch.full((1,4),1/3.9),atol=1e-6)

def test_permutation_invariance():
    x=fixture();a=components(**x);order=torch.tensor([3,0,2,1])
    for key in ('roots','hands','hand_force','assignment','local_points','goals'):x[key]=x[key][:,order]
    b=components(**x)
    for key in ('reward','group_gate','raw_at_progress'):assert torch.allclose(b[key],a[key][:,order])

def test_navigation_matches_baseline_for_center_target():
    torch.manual_seed(4);roots=torch.randn(4,4,3)*4;centers=torch.zeros_like(roots);normals=torch.zeros_like(roots);normals[...,1]=1;half=torch.ones_like(roots)*.8
    a=approach_distance(roots,centers,normals,half,torch.zeros(4,4))
    b=baseline_approach(roots,centers,normals,half,1.,.45)
    assert torch.allclose(a,b)

def test_solo_matches_baseline_components_when_gate_open():
    x=fixture()
    for key in ('roots','hands','hand_force','assignment','local_points','goals'):x[key]=x[key][:,:1]
    x['sizes'][0,0,0]=1.4;x['local_points'][...,0]=torch.tensor([.15,-.15]);x['hands'][...,0]=torch.tensor([.15,-.15]);x['hand_force'].zero_()
    r=components(**x);phi,hit,point,normal=contact_geometry(x['hands'],x['boxes'][:,:1],x['sizes'][:,:1],x['hand_force'],x['box_force'][:,:1],True,1.04)
    assert r['group_gate'].all() and not hit.any()
    assert torch.allclose(r['state'][...,0],.2*phi)
    assert torch.allclose(r['state'][...,1],.2*torch.exp(-10*torch.tensor(9.)))
    assert torch.allclose(r['reward'],(r['state']+r['progress']+r['success']).sum(-1))
