from pathlib import Path
import torch
import yaml

from utils.stage2_jointpush import sample_groups,group_sizes,validate_env,checkpoint_contract,check_checkpoint
from utils.stage2_jointpush_rsi import sample_push_initialization,hand_targets,coupled_body_overlap
from utils.jointpush_geometry import push_rsi_hand_targets

ROOT = Path(__file__).resolve().parents[2]
ENV = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage2_jointpush_rsi.yaml').read_text())['env']


def test_config_keeps_stage1_rsi_and_separates_checkpoint_contract():
    stage1 = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_jointpush.yaml').read_text())['env']
    loco = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage2_jointpush.yaml').read_text())['env']
    validate_env(ENV)
    assert ENV['templateRsi'] == stage1['templateRsi']
    assert ENV['jointPush'] == stage1['jointPush']
    assert ENV['ampExpertMapping'] == stage1['ampExpertMapping']
    assert ENV['eval'] == loco['eval']
    assert checkpoint_contract(ENV) != checkpoint_contract(loco)
    import pytest
    with pytest.raises(ValueError):
        check_checkpoint({'stage2_jointpush_contract':checkpoint_contract(loco)},ENV)


def test_coupled_rsi_is_groupwise_half_and_solo_independent():
    torch.manual_seed(42)
    groups = sample_groups(100000,4,.8,'cpu')
    interaction = sample_push_initialization(groups,torch.full_like(groups,5))
    same = groups[:,:,None] == groups[:,None]
    assert ((interaction[:,:,None] == interaction[:,None]) | ~same).all()
    sizes = group_sizes(groups)
    for mask in (sizes > 1,sizes == 1):
        assert abs(float(interaction[mask].float().mean())-.5) < .01
    assert not sample_push_initialization(groups,torch.ones_like(groups)).any()


def test_shared_targets_use_assigned_strip_on_actual_rear_face():
    boxes = torch.zeros(4,13)
    boxes[:,2] = .8
    boxes[:,6] = 1.
    sizes = torch.tensor([8.,2.,1.6]).expand(4,-1)
    goals = torch.tensor([0.,5.,.8]).expand(4,-1)
    targets = hand_targets(boxes,sizes,goals,torch.full((4,),4),torch.arange(4),ENV['jointPush']['contact'])
    torch.testing.assert_close(targets[...,1],torch.full((4,2),-1.))
    torch.testing.assert_close(targets[...,0].mean(-1),torch.tensor([-3.,-1.,1.,3.]))
    solo = hand_targets(boxes,sizes,goals,torch.ones(4,dtype=torch.long),torch.zeros(4,dtype=torch.long),ENV['jointPush']['contact'])
    direct = push_rsi_hand_targets(boxes[:,None],goals[:,None],sizes[:,None],
        dict(push=ENV['jointPush']['contact'],contact_at=ENV['jointPush']['contact']))[:,0]
    torch.testing.assert_close(solo,direct)


def test_body_overlap_detects_people_in_same_rsi_group():
    states = torch.zeros(1,4,1,13)
    states[...,6] = 1.
    states[0,:,0,0] = torch.tensor([0.,.2,4.,4.])
    geometry = dict(ids=torch.tensor([0]),offsets=torch.zeros(1,3),radii=torch.tensor([.3]))
    groups = torch.tensor([[0,0,2,3]])
    assert coupled_body_overlap(states,geometry,groups,torch.ones(1,4,dtype=torch.bool)).item()
    states[0,1,0,0] = 1.
    assert not coupled_body_overlap(states,geometry,groups,torch.ones(1,4,dtype=torch.bool)).item()


def test_legacy_cache_maps_actual_bank_and_reuses_stage1_late_sampling():
    from types import SimpleNamespace
    import numpy as np
    from env.tasks.multi_agent.jointpush_solo_rsi_cache import JointPushSoloCache
    from utils.stage2_jointpush import asset_bank
    from utils.size_rsi import size_key
    sizes,_ = asset_bank(2,ENV,'cpu')
    lib = SimpleNamespace(num_motions=lambda:1,_motion_weights=torch.ones(1),_motion_dt=torch.tensor([.01]))
    cache = JointPushSoloCache.__new__(JointPushSoloCache)
    cache.env = SimpleNamespace(num_envs=2,num_agents=4,device='cpu',_skill=ENV['skill'],
        _motion_lib={k:lib for k in ('sit','climb','pickUp','carryWith','putDown')})
    cache.spec = dict(lateProbability=.7)
    cache.sizes,cache.entries = sizes.tolist(),{}
    valid = np.ones(100,dtype=bool)
    phase = np.arange(100) < 50
    for kind in ('sit','climb','pickUp','carryWith','putDown','putDownOnTop'):
        for row in range(2):
            for owner in range(4):
                cache.entries[cache._key(kind,row,owner)] = [(valid,phase)]
    assert cache._key('sit',0,3) == ('sit',size_key(sizes[0,7]))
    assert cache._key('climb',0,3) == ('climb',size_key(sizes[0,11]))
    assert cache._key('putDownOnTop',0,3) == ('putDownOnTop',size_key(sizes[0,3]),size_key(sizes[0,15]))
    cache._prepare_sampling()
    rows = torch.zeros(20000,dtype=torch.long)
    owners = torch.full_like(rows,3)
    _,times = cache.sample('sit',rows,owners,torch.ones_like(rows))
    frames = (times/.01).round().long()
    fraction = ((frames >= 35)&(frames < 50)).float().mean()
    assert abs(float(fraction)-.745) < .015
