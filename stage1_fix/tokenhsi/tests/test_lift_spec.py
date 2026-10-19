"""Behavioural contracts for support/lift; no simulator imports."""
from pathlib import Path
import copy
import math
import sys
import torch
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tokenhsi')]
from utils.lift_spec import (support_geometry,lift_scores,reward_step,mask_lift_amp,
                            ARM_FEATURES,relation_matrix,validate_config)


def fixture():
    cfg=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/lift_stage1.yaml').read_text())['env']['liftTask']
    box=torch.zeros(1,2,13);box[...,6]=1;box[...,2]=.8
    size=torch.tensor(cfg['box_size']).expand(1,2,3)
    half_gap=(.258947-.1875)/2
    rel=torch.tensor([[half_gap,-.12,-.215],[half_gap,.12,-.215]]).expand(1,2,2,3)
    hand=rel+box[...,:3].unsqueeze(-2)
    fore=rel.clone();fore[...,0]=-half_gap
    elbow=(fore-cfg['support']['forearm_fraction']*rel)/(1-cfg['support']['forearm_fraction'])+box[...,:3].unsqueeze(-2)
    forces=torch.ones(1,2,4)
    target=box[...,:3].clone();target[...,2]=1.5
    roots=box[...,:3].clone();roots[...,0]=-1.5
    return cfg,roots,hand,elbow,box,size,forces,target,box[...,3:7].clone()


def scores(values):return lift_scores(*values[1:],values[0])


def test_balanced_four_point_support_and_saturation():
    v=fixture();s=scores(v)
    torch.testing.assert_close(s['support_state'],torch.ones(1,2))
    assert s['support_success'].all()
    # Near-perfect real contact saturates; approach progress follows HOLDING.
    v[6].fill_(.49)
    s=scores(v)
    assert (s['support_state_raw']<1).all() and s['support_success'].all()
    assert (s['support_state']==1).all() and (s['support_progress']==1).all()


def test_contact_loss_immediately_removes_lift_rewards():
    v=fixture();v[6][0,0,0]=0
    s=scores(v)
    assert s['support_state'][0,0]==0
    assert s['lift_state'][0,0]==0 and s['lift_progress'][0,0]==0
    assert s['support_state'][0,1]==1  # no teammate leakage
    expected=1/(1+(1.5-.5)/1.)
    assert abs(s['support_progress'][0,0].item()-expected)<1e-6


def test_centre_alone_cannot_reward_degenerate_support():
    v=fixture();v[2][...,:2]=0;v[3][...,:2]=0
    assert (scores(v)['support_state']==0).all()


def test_polygon_must_contain_box_com():
    v=fixture();v[2][...,0]+=.2;v[3][...,0]+=.2
    assert (scores(v)['support_state']==0).all()


def test_side_or_top_contact_not_underhand_support():
    v=fixture();v[2][...,2]+=.35;v[3][...,2]+=.35
    assert (scores(v)['support_state']==0).all()


def test_lift_uses_full_distance_and_both_live_gates():
    v=fixture();s=scores(v)
    v[7][...,2]=.8
    close=scores(v)
    assert (close['lift_state']>s['lift_state']).all()
    assert (close['lift_progress']>s['lift_progress']).all()
    v[7][...,0]+=.2
    assert (scores(v)['lift_state']<close['lift_state']).all()
    v=fixture();s=scores(v);v[8][...,2]=math.sin(math.radians(10)/2);v[8][...,3]=math.cos(math.radians(10)/2)
    angled=scores(v)
    torch.testing.assert_close(angled['lift_progress']/s['lift_progress'],torch.full((1,2),math.exp(-1)),atol=1e-5,rtol=1e-5)
    v[4][...,12]=1
    assert (scores(v)['angle_score']<angled['angle_score']).all()


def test_hold_is_consecutive_and_bonus_only_once():
    v=fixture();v[7][...,2]=.8;s=scores(v)
    elapsed=torch.zeros(1,2);achieved=torch.zeros(1,2,dtype=torch.bool)
    for _ in range(4):
        terms,elapsed,achieved,success=reward_step(s,elapsed,achieved,.1,v[0])
        assert not success.any() and (terms[...,-1]==0).all()
    invalid=copy.deepcopy(s);invalid['hold_valid'][0,0]=False
    _,elapsed,achieved,success=reward_step(invalid,elapsed,achieved,.1,v[0])
    assert elapsed[0,0]==0 and success[0,1]
    for _ in range(5):terms,elapsed,achieved,success=reward_step(s,elapsed,achieved,.1,v[0])
    assert success.all() and achieved.all()
    assert terms[0,0,-1]>0 and terms[0,1,-1]==0
    terms,*_=reward_step(s,elapsed,achieved,.1,v[0])
    assert (terms[...,-1]==0).all()


def test_amp_mask_every_history_frame_without_mutating_source():
    x=torch.randn(7,10,129,requires_grad=True)
    y=mask_lift_amp(x)
    assert not y[...,list(ARM_FEATURES)].any()
    keep=[j for j in range(129) if j not in ARM_FEATURES]
    torch.testing.assert_close(y[...,keep],x[...,keep])
    y.sum().backward()
    assert not x.grad[...,list(ARM_FEATURES)].any()
    assert len(keep)==91 and len(ARM_FEATURES)==38


def test_distinct_support_and_lift_relations_and_config():
    v=fixture();validate_config(v[0]);r=relation_matrix(2,2)
    assert torch.equal(r,r.T) and (r.diag()==1).all()
    assert r[0,2]==6 and r[2,4]==7 and r[0,3]==0 and r[2,5]==0
    assert r[1,3]==6 and r[3,5]==7


def test_support_lift_bias_reaches_actor_and_critic_and_is_trainable():
    from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
    torch.set_num_threads(1)
    train=yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/amp_ma_lift_stage1.yaml').read_text())
    builder=AMPMultiAgentBuilder();builder.load(train['params']['network'])
    net=builder.build('amp',actions_num=32,input_shape=(550,),amp_input_shape=(1290,),
        value_size=1,num_agents=2,num_objects=2,humanoid_obs_size=230,
        object_obs_size=39,goal_obs_size=6,observation_mode='clean_scene',
        scene_entity_sizes=[223,30,1],scene_kinematic_size=7,scene_arena_scale=5.,
        relation_graph_spec={'template':'support_lift_v1'},device='cpu')
    nodes=torch.randn(2,508)
    poses=torch.zeros(2,6,7);poses[...,6]=1
    obs=torch.cat((nodes,poses.flatten(1)),-1)
    for enc in (net.actor_encoder,net.critic_encoder):
        enc.gta_diagnostics_first_forward=False
        assert torch.equal(enc.rel_matrix,relation_matrix(2,2))
        enc.set_entity_counts(2,2)
        assert torch.equal(enc.rel_matrix,relation_matrix(2,2))
        bias=enc.build_relation_bias()
        assert bias.shape==(4,2,6,6) and torch.isfinite(bias).all()
        bias[...,0,2].sum().backward(retain_graph=True)
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in enc.edge_encoder.parameters())
    mu,sigma=net.eval_actor(obs);value=net.eval_critic(obs)
    assert mu.shape==sigma.shape==(4,32) and value.shape==(4,1)
    assert torch.isfinite(mu).all() and torch.isfinite(value).all()


def test_dataset_keeps_original_test_clip_out_of_training():
    data=ROOT/'tokenhsi/data'
    entries=yaml.safe_load((data/'dataset_lift_stage1.yaml').read_text())['motions']['lift']
    assert len(entries)==2 and all('__train_full/' in row['file'] for row in entries)
    assert all((data/row['file']).is_file() for row in entries)


def test_support_uses_asset_distal_forearm_endpoint():
    import xml.etree.ElementTree as ET
    asset=ET.parse(ROOT/'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml')
    for name in ('lift_stage1.yaml','lift_stage1_teamhoi.yaml'):
        cfg=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent'/name).read_text())['env']['liftTask']['support']
        for side in ('right','left'):
            body=asset.find(".//body[@name='%s_lower_arm']" % side)
            assert body is not None
            distal=abs(float(body.find('geom').get('fromto').split()[-1]))
            hand=body.find("body[@name='%s_hand']" % side)
            hand_dist=abs(float(hand.get('pos').split()[-1]))
            assert abs(cfg['forearm_fraction']*hand_dist-distal)<1e-8
            assert abs(hand_dist-distal-.071447)<1e-8
    v=fixture()
    support=support_geometry(v[2],v[3],v[4],v[5],v[6],v[0]['support'])
    assert (support['state']>.9).all()  # 7.14cm footprint can still succeed
