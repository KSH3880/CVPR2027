"""Random AT goals, conservative clearance and PUSH-only 10deg gating."""
from pathlib import Path
import math
from copy import deepcopy
from isaacgym import gymapi  # before torch
import torch
import yaml
import pytest
from utils.push_scene import sample_random_at_scene,random_at_goal_valid,segment_distance,validate_scene,forward_xy
from utils.open_push_spec import validate_env
from env.tasks.multi_agent.open_push_reward import OpenPushRuntime,push_hand_targets,push_goal_opposite_xy_target,push_at_edge_reward,push_rsi_hand_targets

ROOT=Path(__file__).resolve().parents[2]


def test_contact_state_strict_threshold_and_force_boundaries():
    from env.tasks.multi_agent.open_push_reward import push_contact_near,push_hand_contact,push_debug_metrics
    cfg=config()['relationReward'];spec=cfg['push'];size=[1.3]*3
    # Equal scores exactly at the threshold fail the strict comparison.
    distance=torch.tensor([[[.1,.1]]])
    cfg['satisfaction_threshold']=torch.exp(-10*distance.square())[0,0,0].item()
    assert not push_contact_near(distance,spec,cfg).any()
    cfg['satisfaction_threshold']=.9
    boxes=torch.zeros(1,1,13);boxes[...,2]=.65;boxes[...,6]=1
    goals=boxes[...,:3].clone();goals[...,1]+=2
    hands=torch.tensor([[[[-.2,-.75,4.],[.2,-.75,-2.]]]])
    forces=torch.zeros_like(hands);forces[...,1]=-.5
    bf=torch.zeros(1,1,3);bf[...,1]=.5
    def contact():return push_hand_contact(hands,forces,boxes,bf,size,spec,goals,cfg)
    assert contact().all()  # 10cm: state exp(-.1) > .9, arbitrary hand heights
    hands[...,1]=-.80
    assert not contact().any()  # 15cm is within old 18cm, but state < .9
    metrics=push_debug_metrics(hands,forces,boxes,bf,boxes[...,:3],size,spec,
        contact(),torch.ones(1,1,dtype=torch.bool),goals=goals,config=cfg)
    assert not metrics['both_hands_near'].any()
    hands[...,1]=-.65
    forces[...,0,1]=-.499
    assert not contact().any()
    forces[...,0,1]=-.5;bf[...,1]=.499
    assert not contact().any()
    # A close midpoint must not hide a hand far away from the face.
    bf[...,1]=.5;hands[...,0,1]=-.35;hands[...,1,1]=-.95
    assert not contact().any()


@pytest.mark.parametrize('direction',[[0.,2.],[0.,-2.],[2.,0.],[-2.,0.],[2.,2.],[0.,0.]])
@pytest.mark.parametrize('yaw',[0.,.7,math.pi/2])
def test_goal_rear_contact_and_force_diagnostics(direction,yaw):
    from env.tasks.multi_agent.open_push_reward import push_hand_contact,push_forward_normal,push_debug_metrics
    env=config();cfg=env['relationReward'];spec=cfg['push'];size=env['pushBox']['size']
    boxes=torch.zeros(1,2,13);boxes[...,2]=.65
    boxes[...,5]=math.sin(yaw/2);boxes[...,6]=math.cos(yaw/2)
    goals=boxes[...,:3].clone();goals[...,:2]+=torch.tensor(direction)
    rear=push_goal_opposite_xy_target(boxes,goals,size)
    hands=torch.zeros(1,2,2,3);hands[...,:2]=rear.unsqueeze(-2)
    hands[...,0,2]=.3;hands[...,1,2]=1.2
    normal=push_forward_normal(boxes,spec,goals,cfg)
    expected=boxes[...,:2]-rear;expected/=expected.norm(dim=-1,keepdim=True)
    torch.testing.assert_close(normal[...,:2],expected)
    forces=-normal.unsqueeze(-2).expand_as(hands)*2
    box_forces=normal*3
    def contact(h=hands,f=forces):
        return push_hand_contact(h,f,boxes,box_forces,size,spec,goals,cfg)
    assert contact().all()
    assert not contact(f=-forces).any()
    one=forces.clone();one[...,1,:]=0
    assert not contact(f=one).any()
    far=hands.clone();far[...,:2]-=normal[...,:2].unsqueeze(-2)*.5
    assert not contact(h=far).any()
    metrics=push_debug_metrics(hands,forces,boxes,box_forces,boxes[...,:3],size,spec,
        contact(),torch.ones(1,2,dtype=torch.bool),goals=goals,config=cfg)
    assert metrics['both_hands_near'].all()
    torch.testing.assert_close(metrics['right_distance'],torch.zeros(1,2),atol=1e-6,rtol=0)
    torch.testing.assert_close(metrics['right_reaction_N'],torch.full((1,2),2.))
    torch.testing.assert_close(metrics['box_forward_force_N'],torch.full((1,2),3.))
    legacy=deepcopy(cfg);legacy['push_at'].pop('hand_reference')
    torch.testing.assert_close(push_hand_contact(hands,forces,boxes,box_forces,size,spec,goals,legacy),
        push_hand_contact(hands,forces,boxes,box_forces,size,spec))


@pytest.mark.parametrize('direction',[[0.,2.],[0.,-2.],[2.,0.],[-2.,0.],[2.,2.],[0.,0.]])
@pytest.mark.parametrize('yaw',[0.,.7,math.pi/2])
def test_rsi_matches_goal_rear_reward_and_rotates_pose(direction,yaw):
    from utils.open_push_rsi import align_push_pose
    env=config();cfg=env['relationReward'];size=env['pushBox']['size']
    boxes=torch.zeros(1,2,13);boxes[...,2]=.65
    boxes[...,5]=math.sin(yaw/2);boxes[...,6]=math.cos(yaw/2)
    goals=boxes[...,:3].clone();goals[...,:2]+=torch.tensor(direction)
    targets=push_rsi_hand_targets(boxes,goals,size,cfg)
    rear=push_goal_opposite_xy_target(boxes,goals,size)
    torch.testing.assert_close(targets[...,:2].mean(-2),rear)
    normal=boxes[...,:2]-rear;normal/=normal.norm(dim=-1,keepdim=True)
    body=torch.tensor([[[0.,0.,1.],[.2,.7,.8],[-.2,.7,.8],[.1,0.,.05]]])
    rot=torch.zeros(1,4,4);rot[...,3]=1
    vel=torch.zeros_like(body);vel[...,1]=1
    aligned,_,rotvel,_,_=align_push_pose(body,rot,vel,vel,targets[:,0],[1,2],.06)
    torch.testing.assert_close(aligned[:,[1,2],:2].mean(1),rear[:,0]-.06*normal[:,0],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(rotvel[:,0,:2],normal[:,0],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(aligned[...,2],body[...,2])
    legacy=deepcopy(cfg);legacy['push_at'].pop('hand_reference')
    torch.testing.assert_close(push_rsi_hand_targets(boxes,goals,size,legacy),push_hand_targets(boxes,size,cfg['push']))

def config():
    env=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_toy_random_at_10deg.yaml').read_text())['env']
    env['motion_file']='tokenhsi/data/dataset_open_push.yaml'
    return env


def test_random_goal_distribution_and_full_path_clearance():
    env=config();validate_env(env)
    xy,q,goals=sample_random_at_scene(2048,env,'cpu',torch.Generator().manual_seed(7))
    assert random_at_goal_valid(xy,q,goals,env).all()
    anchors=xy.new_tensor(env['pushScene']['owner_anchor_xy'])
    assert ((goals-anchors).abs()<=2).all()
    assert ((goals-anchors).norm(dim=-1)>=1).all()
    assert ((goals-xy).norm(dim=-1)>=1).all()
    # All four owner-local quadrants are represented; goals are not fixed 2m/yaw.
    for agent in range(2):
        relative=goals[:,agent]-anchors[agent]
        bins=(relative[:,0]>0).long()+2*(relative[:,1]>0).long()
        assert (torch.bincount(bins,minlength=4)>100).all()
    direction=(goals-xy)/(goals-xy).norm(dim=-1,keepdim=True)
    assert ((direction-forward_xy(q)).norm(dim=-1)>1).float().mean()>.3
    assert (goals-xy).norm(dim=-1).std()>.3


@pytest.mark.parametrize('c,d,expected',[
    ([0.,1.],[0.,-1.],0.),([2.,0.],[3.,0.],1.),([0.,1.],[1.,1.],1.),
    ([.5,0.],[.5,0.],0.),([2.,1.],[2.,1.],math.sqrt(2))])
def test_exact_segment_clearance(c,d,expected):
    result=segment_distance(torch.tensor([0.,0.]),torch.tensor([1.,0.]),torch.tensor(c),torch.tensor(d))
    assert float(result)==pytest.approx(expected,abs=1e-6)


@pytest.mark.parametrize('degrees,hand_ok',[(9.99,True),(10.,True),(10.01,False),(25.,False)])
def test_10deg_applies_to_push_but_not_at(degrees,hand_ok):
    env=config();cfg=env['relationReward']
    boxes=torch.zeros(1,2,13);boxes[...,2]=.65;boxes[...,6]=1
    start=boxes[...,:3].clone();goals=start.clone();goals[...,1]+=2
    rt=OpenPushRuntime(1,env['relationGraph'],cfg,env['doorHold'],1/30,'cpu',env['pushBox']['size'],env['manipulationContact'])
    rt.reset(torch.arange(1),torch.ones(1,2,dtype=torch.long),boxes)
    boxes[...,3]=math.sin(math.radians(degrees)/2);boxes[...,6]=math.cos(math.radians(degrees)/2)
    hands=push_hand_targets(boxes,env['pushBox']['size'],cfg['push'])
    # Make the XY midpoint exactly match the yaw-only rear target.
    hands[...,:2]=boxes[...,:2].unsqueeze(-2)+hands.new_tensor([[.2,-.65],[-.2,-.65]])
    result=rt.step(start,hands,start,torch.zeros(1,2),boxes,goals)
    assert bool(rt.own_success[:,::2].all())==hand_ok
    if not hand_ok:
        assert not result['edge_state'][:,::2].any() and not result['edge_progress'][:,::2].any()
    boxes[...,:3]=goals;hands+=10
    result=rt.step(start,hands,start,torch.zeros(1,2),boxes,goals)
    assert result['success'].all()  # placement still saturates PUSH, by design
    torch.testing.assert_close(result['reward'],torch.full((1,2),1.14))


def test_new_contract_and_validation_preserve_old_experiment():
    from utils.open_push_rsi import bank_contract
    import json
    env=config()
    data=json.loads(Path(env['openPushRSI']['bank_file']).read_text())
    assert data['contract']==bank_contract(env)
    old=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_toy_only_random_scene_at.yaml').read_text())['env']
    assert old['relationReward']['push']['max_tilt']==.35
    assert old['pushScene']['mode']=='owner_loco_random'
    assert env['relationReward']['push']['max_tilt']==pytest.approx(math.radians(10))
    bad=deepcopy(env);bad['pushScene']['random_at_goal']['reach']=.5
    with pytest.raises(ValueError):validate_scene(bad)


def test_reject_near_goals_and_crossing_paths():
    env=config()
    xy,q,goals=sample_random_at_scene(8,env,'cpu',torch.Generator().manual_seed(4))
    near_box=goals.clone();near_box[:,0]=xy[:,0]
    assert not random_at_goal_valid(xy,q,near_box,env).any()
    near_owner=goals.clone();near_owner[:,0]=goals.new_tensor(env['pushScene']['owner_anchor_xy'][0])
    assert not random_at_goal_valid(xy,q,near_owner,env).any()
    env['pushScene']['owner_anchor_xy']=[[0.,-2.],[0.,2.]]
    env['pushScene']['random_at_goal']['reach']=4.
    xy=torch.tensor([[[0.,-1.],[0.,1.]]]);goals=xy.flip(1)
    q=torch.tensor([[[0.,0.,0.,1.],[0.,0.,0.,1.]]])
    assert not random_at_goal_valid(xy,q,goals,env).any()


@pytest.mark.parametrize('yaw',[0.,.7,math.pi/2,math.pi])
@pytest.mark.parametrize('direction,expected',[
    ([0.,2.],[0.,-.3]),([0.,-2.],[0.,.3]),
    ([2.,0.],[-.4,0.]),([-2.,0.],[.4,0.]),
    ([2.,2.],[-.3,-.3]),([0.,0.],[0.,-.3])])
def test_goal_opposite_target_on_rotated_rectangle(yaw,direction,expected):
    # Non-square box catches using a fixed radius or the wrong box dimension.
    boxes=torch.zeros(1,2,13);boxes[...,:2]=torch.tensor([3.,-2.])
    boxes[...,5]=math.sin(yaw/2);boxes[...,6]=math.cos(yaw/2)
    c,s=math.cos(yaw),math.sin(yaw)
    def rotated(v):return torch.tensor([c*v[0]-s*v[1],s*v[0]+c*v[1]])
    goals=boxes[...,:3].clone();goals[...,:2]+=rotated(direction)
    target=push_goal_opposite_xy_target(boxes,goals,[.8,.6,1.3])
    torch.testing.assert_close(target,boxes[...,:2]+rotated(expected),atol=1e-6,rtol=1e-6)
    assert torch.isfinite(target).all()


def test_push_reward_follows_current_goal_and_box_position():
    env=config();cfg=env['relationReward'];size=env['pushBox']['size']
    boxes=torch.zeros(1,2,13);boxes[...,2]=.65;boxes[...,6]=1
    start=boxes[...,:3].clone();goals=start.clone();goals[...,1]-=2
    target=push_goal_opposite_xy_target(boxes,goals,size)
    hands=torch.zeros(1,2,2,3);hands[...,:2]=target.unsqueeze(-2)
    hands[...,0,2]=10.;hands[...,1,2]=-10.  # XY-only shaping stays height-free.
    r=push_at_edge_reward(start,hands,boxes,goals,start,size,cfg)
    assert r['own_success'][:,::2].all()
    old=deepcopy(cfg);old['push_at'].pop('hand_reference')
    assert not push_at_edge_reward(start,hands,boxes,goals,start,size,old)['own_success'][:,::2].any()
    goals[...,1]+=4
    assert not push_at_edge_reward(start,hands,boxes,goals,start,size,cfg)['own_success'][:,::2].any()
    boxes[...,0]+=.5
    target=push_goal_opposite_xy_target(boxes,goals,size);hands[...,:2]=target.unsqueeze(-2)
    assert push_at_edge_reward(start,hands,boxes,goals,start,size,cfg)['own_success'][:,::2].all()


def test_invalid_hand_reference_rejected():
    env=config();env['relationReward']['push_at']['hand_reference']='unknown'
    with pytest.raises(ValueError):validate_env(env)
