"""Variable-size physical geometry must agree with scalar-size reward paths."""
from pathlib import Path
import json
from isaacgym import gymapi
import torch
import yaml
from env.tasks.multi_agent.open_push_reward import (push_rsi_hand_targets,push_rear_xy,
    push_face_geometry,contact_at_reward,push_debug_metrics)
from utils.native_push_rsi import trim_runs,NativePushBank,native_contract
from utils.open_push_spec import validate_env


def test_variable_contact_and_targets_match_each_actual_asset():
    env=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/push_native_phase_xy.yaml').read_text())['env']
    validate_env(env)
    torch.manual_seed(41)
    boxes=torch.zeros(3,2,13);boxes[...,2]=.65
    angle=torch.rand(3,2)*6.28;boxes[...,5]=torch.sin(angle/2);boxes[...,6]=torch.cos(angle/2)
    goals=torch.randn(3,2,3);goals[...,:2]*=3;goals[...,2]=.65
    sizes=torch.tensor(env['pushBox']['size_profiles'])[[0,5,30,35,17,25]].reshape(3,2,3)
    hands=push_rsi_hand_targets(boxes,goals,sizes,env['relationReward'])+torch.randn(3,2,2,3)*.1
    roots=boxes[...,:3].clone();contact=torch.zeros(3,2,dtype=torch.bool)
    result=contact_at_reward(roots,hands,boxes,goals,contact,sizes,env['relationReward'])
    for e in range(3):
        for a in range(2):
            b=boxes[e:e+1,a:a+1];g=goals[e:e+1,a:a+1];h=hands[e:e+1,a:a+1];size=sizes[e,a].tolist()
            expected=contact_at_reward(roots[e:e+1,a:a+1].expand(-1,2,-1),h.expand(-1,2,-1,-1),b.expand(-1,2,-1),g.expand(-1,2,-1),contact[e:e+1,a:a+1].expand(-1,2),size,env['relationReward'])
            for key in result:
                if key=='reward':continue  # Team sharing uses the other owner's local reward.
                width=2 if result[key].shape[1]==4 else 1
                assert torch.allclose(result[key][e:e+1,a*width:(a+1)*width].float(),expected[key][:,:width].float(),atol=1e-6),key
            assert torch.allclose(push_rear_xy(h,b,g,size),push_rear_xy(hands,boxes,goals,sizes)[e:e+1,a:a+1],atol=1e-6)
    zeros=torch.zeros_like(hands);force=torch.zeros_like(roots)
    metrics=push_debug_metrics(hands,zeros,boxes,force,roots,sizes,env['relationReward']['push'],contact,contact,goals=goals,config=env['relationReward'])
    assert all(v.shape==(3,2) and torch.isfinite(v).all() for v in metrics.values())


def test_trim_excludes_short_and_edge_frames():
    valid=torch.zeros(80,dtype=torch.bool);valid[0:20]=True;valid[25:80]=True
    kept=trim_runs(valid,30)
    assert not kept[:31].any() and not kept[74:].any()
    assert kept[31:74].all()


def test_actual_sole_support_excludes_reported_airborne_frame():
    from utils.native_push_rsi import load_cpu_reference
    from utils.door_reference import rotate
    env=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/push_native_phase_xy.yaml').read_text())['env']
    env['motion_file']='tokenhsi/data/dataset_push_native_phase_xy.yaml'
    lib,names,geo=load_cpu_reference(env)
    payload=json.loads(Path('tokenhsi/data/push_native_phase_xy_grounded_rsi_bank.json').read_text())
    rows=payload['candidates']['push']
    assert rows and all(r['motion_id']!=7 for r in rows)
    mids=torch.tensor([r['motion_id'] for r in rows]+[7])
    times=torch.tensor([r['time'] for r in rows]+[5.1333336])
    body,rot,_,_=lib.get_motion_state_max(mids,times)
    points=body[:,geo['ground_ids']]+rotate(
        rot[:,geo['ground_ids']],geo['ground_offsets'].expand(len(times),-1,-1))
    sole=torch.stack([points[:,mask,2].amin(-1) for mask in geo['foot_masks']],-1)
    assert (sole[:-1].amin(-1)<=.02).all()
    assert (sole[-1]>.02).all()  # Screenshot: right 2.67cm / left 5.43cm.


def test_current_goal_range_and_md_pool_without_clip7():
    from utils.push_scene import sample_random_at_scene,random_at_goal_valid
    env=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/push_native_phase_xy.yaml').read_text())['env']
    validate_env(env)
    assert env['pushScene']['random_at_goal']['sampling']=='box_forward'
    assert env['pushScene']['random_at_goal']['min_distance']==2.
    assert env['pushScene']['random_at_goal']['max_distance']==5.
    assert env['openPushRSI']['excluded_motion_ids']==[7]
    assert 'max_support_sole_height_m' not in env['openPushRSI']
    bank=json.loads(Path(env['openPushRSI']['bank_file']).read_text())
    original=json.loads(Path('tokenhsi/data/push_native_phase_xy_rsi_bank.json').read_text())
    expected=[r for r in original['candidates']['push'] if r['motion_id']!=7]
    assert bank['candidates']['push']==expected
    assert len(expected)==949
    assert {r['motion_id'] for r in expected}=={0,3,4,8}
    xy,q,goals=sample_random_at_scene(2048,env,'cpu',torch.Generator().manual_seed(20261020))
    distance=(goals-xy).norm(dim=-1)
    assert ((distance>=2-1e-6)&(distance<=5+1e-6)).all()
    assert distance.min()<2.02 and distance.max()>4.98
    assert random_at_goal_valid(xy,q,goals,env).all()
    assert torch.allclose((xy-torch.tensor(env['pushScene']['owner_anchor_xy'])).norm(dim=-1).clamp(1,2),(xy-torch.tensor(env['pushScene']['owner_anchor_xy'])).norm(dim=-1),atol=1e-6)


def test_box_forward_targets_are_positive_local_y_only():
    from utils.push_scene import sample_random_at_scene,random_at_goal_valid,forward_xy
    env=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/push_native_phase_xy.yaml').read_text())['env']
    for seed in (1,42,20261020):
        xy,q,goals=sample_random_at_scene(2048,env,'cpu',torch.Generator().manual_seed(seed))
        f=forward_xy(q);delta=goals-xy;along=(delta*f).sum(-1)
        assert ((along>=2-1e-6)&(along<=5+1e-6)).all()
        assert (delta-along[...,None]*f).norm(dim=-1).max()<1e-5
        assert random_at_goal_valid(xy,q,goals,env).all()
        opposite=xy-delta
        assert not random_at_goal_valid(xy,q,opposite,env).any()
        sideways=xy+along[...,None]*torch.stack((-f[...,1],f[...,0]),-1)
        assert not random_at_goal_valid(xy,q,sideways,env).any()
        yaw=torch.atan2(f[...,1],f[...,0]);assert yaw.max()-yaw.min()>6.
