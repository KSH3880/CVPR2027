from pathlib import Path
from types import SimpleNamespace
import torch
import yaml
from env.tasks.multi_agent.humanoid_chair_ft import HumanoidChairFT
from utils.unified_training import validate_unified_env

ROOT=Path(__file__).resolve().parents[1]

class FakeGym:
    def load_asset(self,sim,folder,name,options):
        assert options.fix_base_link
        return folder
    def get_asset_rigid_body_count(self,asset):return 1
    def create_box(self,*args):raise AssertionError('No box should be created')


def test_both_target_categories_use_new_tokenizer_and_correct_support_surface():
    cfg=yaml.safe_load((ROOT/'data/cfg/multi_agent/chair_table_full_ft.yaml').read_text())
    validate_unified_env(cfg['env'])
    task=HumanoidChairFT.__new__(HumanoidChairFT)
    task.cfg=cfg;task.num_envs=16;task.num_objects=4;task.device='cpu'
    task._build_base_size=[.4,.3,.4];task.gym=FakeGym();task.sim=None
    task._load_box_asset()
    assert not (task._chair_kinds==0).any()
    assert torch.equal(task._chair_kinds[:,:2].sort(-1).values,torch.tensor([1,2]).expand(16,-1))
    assert (task._chair_features[...,0]==1).all()
    tables=task._chair_kinds==1
    assert (task._chair_features[tables][:,1:3]==0).all()
    torch.testing.assert_close(task._chair_seats[tables][:,2],task._box_size[tables][:,2]/2)
    torch.testing.assert_close(task._chair_seat_extents[tables],task._box_size[tables][:,:2])
    chairs=task._chair_kinds==2
    assert (task._chair_features[chairs][:,1:3].norm(dim=-1)>0).all()
    assert cfg['env']['chairFT']['trainable']=='all_layers'
    assert list(cfg['env']['relationGraph']['template_probabilities'].values())==[0,.5,.5,0,0]
