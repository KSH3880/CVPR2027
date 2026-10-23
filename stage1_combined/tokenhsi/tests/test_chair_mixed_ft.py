"""Mixed replay contracts: original box setup and task/RSI isolation."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import yaml
from utils.unified_training import validate_unified_env
from env.tasks.multi_agent.humanoid_chair_mixed_ft import HumanoidChairMixedFT, MixedBoxRsiCache
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry

ROOT = Path(__file__).resolve().parents[1]

def config(name):
    return yaml.safe_load((ROOT/'data/cfg/multi_agent'/f'{name}.yaml').read_text())


def test_original_box_configuration_is_exactly_preserved():
    base=config('stage1_v2_fixed')
    mixed=config('chair_mixed_full_ft')
    options=mixed['env'].pop('chairFT')
    assert mixed==base
    assert options['trainable']=='all_layers'
    assert options['chair_environment_fraction']==.25
    validate_unified_env(config('chair_mixed_full_ft')['env'])
    validate_unified_env(config('chair_sit_climb_full_ft')['env'])


def test_chair_only_probabilities_do_not_replace_box_probabilities():
    task=HumanoidChairMixedFT.__new__(HumanoidChairMixedFT)
    task.cfg=config('chair_mixed_full_ft')
    task._chair_env_mask=torch.tensor([True,False,False])
    probs=torch.tensor([[[0.,.1,.25,.325,.325]]]).expand(3,2,5).clone()
    task._size_template_probabilities=probs.clone()
    with patch.object(HumanoidMACarry,'_init_relation_runtime',return_value=None):
        task._init_relation_runtime()
    assert torch.equal(task._size_template_probabilities[1:],probs[1:])
    assert torch.equal(task._size_template_probabilities[0],torch.tensor([[0.,.5,.5,0.,0.]]).expand(2,-1))


def test_box_cache_profiles_exclude_chairs_even_when_sizes_match():
    cache=MixedBoxRsiCache.__new__(MixedBoxRsiCache)
    cache.env=SimpleNamespace(_chair_env_mask=torch.tensor([True,False]))
    cache.eligible=torch.ones(2,2,5)
    cache.sizes=[[[.45]*3]*4]*2
    cache.entries={}
    result=cache._profiles('sit',0,set())
    assert list(result.values())==[1]
    assert not cache.eligible[0].any()


def test_box_and_chair_loco_reference_slots_both_survive_reset():
    task=HumanoidChairMixedFT.__new__(HumanoidChairMixedFT)
    task._chair_env_mask=torch.tensor([True,False]);task._skill=['loco']
    task._reset_ref_slots={};task._reset_ref_motion_ids={};task._reset_ref_motion_times={}
    def reset(self, rows, owners):
        self._reset_ref_slots['loco']=(rows,owners)
        self._reset_ref_motion_ids['loco']=rows+10
        self._reset_ref_motion_times['loco']=rows.float()+.25
    with patch.object(HumanoidMACarry,'_reset_ref_state_init',reset):
        from env.tasks.multi_agent.humanoid_chair_ft import HumanoidChairFT
        with patch.object(HumanoidChairFT,'_reset_ref_state_init',reset):
            task._reset_ref_state_init(torch.tensor([0,0,1,1]),torch.tensor([0,1,0,1]))
    rows,owners=task._reset_ref_slots['loco']
    assert sorted(zip(rows.tolist(),owners.tolist()))==[(0,0),(0,1),(1,0),(1,1)]
    assert task._reset_ref_motion_ids['loco'].tolist()==[11,11,10,10]


def test_box_chair_table_uses_only_sit_climb_and_preserves_other_box_settings():
    cfg=config('box_chair_table_mixed_full_ft')
    base=config('stage1_v2_fixed')
    options=cfg['env'].pop('chairFT')
    assert cfg['env']['relationGraph']['template_probabilities']==dict(HOLDING=0.,SIT=.5,CLIMB=.5,HOLDING_AT=0.,HOLDING_ON_TOP=0.)
    base['env']['relationGraph']['template_probabilities']=cfg['env']['relationGraph']['template_probabilities']
    base['env']['skillDiscProb']=cfg['env']['skillDiscProb']
    assert cfg==base
    assert options['target_categories']==['chair','table']
    assert options['climb_goal']=='object_comparison'
    assert options['table_input_bbox']=='metadata_top_box_tokenizer'
    assert options['chair_environment_fraction']==2/3
    assert list(options['task_mix'].values())==[1/3]*3
    validate_unified_env(config('box_chair_table_mixed_full_ft')['env'])


def test_mixed_native_assets_have_one_chair_one_table_and_box_sizes_survive():
    task=HumanoidChairMixedFT.__new__(HumanoidChairMixedFT)
    task.cfg=config('box_chair_table_mixed_full_ft');task.num_envs=8;task.num_objects=4;task.num_agents=2;task.device='cpu'
    task.sim=None;task._build_base_size=[.4,.3,.4]
    task.gym=SimpleNamespace(load_asset=lambda *a:object(),get_asset_rigid_body_count=lambda a:1,get_asset_dof_count=lambda a:0)
    def original(self):
        self._box_size=torch.full((8,4,3),.4)
        self._box_assets=[None]*32
    with patch.object(HumanoidMACarry,'_load_box_asset',original):
        task._load_box_asset()
    mask=task._chair_env_mask
    assert int(mask.sum())==round(8*2/3)
    assert (task._chair_kinds[mask,:2].sort(-1).values==torch.tensor([1,2])).all()
    assert (task._chair_features[mask,:2,0]==1).all()
    torch.testing.assert_close(task._box_size[~mask],task._original_box_sizes[~mask])
    assert not task._chair_features[~mask].any()
    table=task._chair_kinds==1
    torch.testing.assert_close(task._chair_seats[...,2][table],task._box_size[...,2][table]/2)


def test_sit_climb_only_overrides_cached_box_task_probabilities():
    task=HumanoidChairMixedFT.__new__(HumanoidChairMixedFT)
    task.cfg=config('box_chair_table_mixed_full_ft')
    task._chair_env_mask=torch.tensor([True,False,False])
    task._size_template_probabilities=torch.tensor([[[0.,.1,.25,.325,.325]]]).expand(3,2,5).clone()
    with patch.object(HumanoidMACarry,'_init_relation_runtime',return_value=None):
        task._init_relation_runtime()
    torch.testing.assert_close(task._size_template_probabilities,torch.tensor([0.,.5,.5,0.,0.]).expand(3,2,5))
