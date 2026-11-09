import numpy as np
from learning.multi_agent.random_demo import candidates


def fixture():
    s=np.zeros((50,1,8),bool);s[5:,:,0]=True;s[15:,:,:7]=True
    native=s.copy();valid=np.ones((50,1),bool);pair=np.zeros((50,1),bool);pair[15:]=True
    return s,native,valid,pair,np.array([False]),np.array([[0,1,0]]),1/30


def test_one_solo_failure_allowed_but_before_required():
    args=fixture();r=candidates(*args);assert r[0]['count']==7
    args[3][:]=False;assert candidates(*args)==[]


def test_early_follower_and_initial_gate_rejected():
    args=fixture();args[0][0,0,1]=True;assert candidates(*args)==[]
    args=fixture();args[4][:]=True;assert candidates(*args)==[]


def test_termination_and_brief_success_rejected():
    args=fixture();args[2][20:]=False;assert candidates(*args)==[]
    args=fixture();args[3][20:]=False;assert candidates(*args)==[]


def test_every_pair_must_succeed():
    args=list(fixture());args[3]=np.concatenate([args[3],np.zeros_like(args[3])],axis=1)
    args[4]=np.array([False,False]);args[5]=np.array([[0,1,0],[0,3,2]])
    assert candidates(*args)==[]


def test_near_wait_requires_both_followers_and_no_retreat():
    from learning.multi_agent.random_demo import near_waits
    roots=np.zeros((80,8,13));roots[:,1,0]=1.5;roots[:,3,0]=2.5
    native=np.zeros((80,8),bool);native[40:,[0,2]]=True
    valid=np.ones(80,bool);packet=np.zeros((8,5),int);packet[0,4]=24;packet[2,4]=26
    goals=np.zeros((8,3));pairs=[[1,0],[3,2]]
    assert near_waits(roots,native,valid,packet,goals,pairs,1/30,2) is None
    assert len(near_waits(roots,native,valid,packet,goals,pairs,1/30,3))==2
    roots[35,3,0]=3.1
    assert near_waits(roots,native,valid,packet,goals,pairs,1/30,3) is None


def test_sixteen_people_threshold_and_optional_zero_pairs():
    s=np.zeros((30,1,16),bool);s[:,:,:13]=True
    args=(s,s.copy(),np.ones((30,1),bool),np.zeros((30,0),bool),np.zeros(0,bool),np.empty((0,3),int),1/30)
    assert candidates(*args)==[]
    assert candidates(*args,minimum_pairs=0)[0]['count']==13
    assert candidates(*args,minimum_pairs=0,minimum_success=14)==[]


def test_bounded_reset_never_applies_to_training_or_formal_eval(monkeypatch):
    from types import SimpleNamespace
    from utils.demo_reset import bounded_demo_reset
    args=SimpleNamespace(test=True,eval=True,headless=True,eval_protocol='')
    task=SimpleNamespace(cfg={'args':args})
    monkeypatch.setenv('RANDOM_DEMO_SEARCH','1');monkeypatch.setenv('RANDOM_DEMO_BOUNDED_RESET','1')
    assert bounded_demo_reset(task)
    args.test=False;assert not bounded_demo_reset(task)
    args.test=True;args.eval_protocol='general';assert not bounded_demo_reset(task)


def test_task_physical_mask_ignores_spares_and_preserves_aliases():
    import torch
    from utils.demo_reset import task_physical_mask
    mapping=torch.tensor([[2,5,2,7,8],[3,3,4,5,6]])
    mask=task_physical_mask(mapping,torch.tensor([3,2]),9)
    assert mask[0].nonzero().flatten().tolist()==[2,5]
    assert mask[1].nonzero().flatten().tolist()==[3]


def test_removed_object_features_do_not_change_demo_human_encoding():
    import torch
    from test_stage2_push_common import build
    from test_before_object_alias import packet, observations
    torch.set_num_threads(1)
    net=build('stage2_before_push').eval();net.demo_task_boxes_only=True
    obs=observations(packet());changed=obs.clone()
    changed[:,892:1132].reshape(1,8,30)[:,5:]=1000
    changed[:,1136:1248].reshape(1,16,7)[:,9:12,:3]=1000
    with torch.no_grad():
        torch.manual_seed(123);a=net._humans(net.actor_encoder,obs)
        torch.manual_seed(123);b=net._humans(net.actor_encoder,changed)
    assert torch.allclose(a,b,atol=1e-5,rtol=1e-5)


def test_near_wait_extent_accepts_sway_but_rejects_travel():
    from learning.multi_agent.random_demo import near_waits
    roots=np.zeros((80,16,13));roots[:,1,0]=3.5
    roots[:,1,7]=0.8
    roots[:,1,1]=0.1*np.sin(np.arange(80))
    native=np.zeros((80,16),bool);native[40:,0]=True
    valid=np.ones(80,bool);packet=np.zeros((16,5),int);packet[0,4]=48
    goals=np.zeros((16,3));args=(roots,native,valid,packet,goals,[[1,0]],1/30,4)
    assert near_waits(*args) is None
    assert near_waits(*args,seconds=.5,extent=.5)
    roots[:,1,1]=np.arange(80)*.05
    assert near_waits(*args,seconds=.5,extent=.5) is None


def test_near_wait_requires_approach_not_spawn_idle():
    from learning.multi_agent.random_demo import near_waits
    roots=np.zeros((80,16,13));roots[:,1,0]=2.5
    native=np.zeros((80,16),bool);native[60:,0]=True
    valid=np.ones(80,bool);packet=np.zeros((16,5),int);packet[0,4]=48
    goals=np.zeros((16,3));args=(roots,native,valid,packet,goals,[[1,0]],1/30,4)
    assert near_waits(*args,seconds=.5,extent=.5,require_walk=True) is None
    roots[:31,1,0]=np.linspace(4,2.5,31)
    assert near_waits(*args,seconds=.5,extent=.5,require_walk=True)
    roots[:31,1,0]=np.linspace(1,2.5,31)
    assert near_waits(*args,seconds=.5,extent=.5,require_walk=True) is None
