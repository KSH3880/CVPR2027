from pathlib import Path
import ast
import torch,yaml
from env.tasks.multi_agent.humanoid_tokenhsi_sit_ft import HumanoidTokenHSISitFT
ROOT=Path(__file__).resolve().parents[2]

def test_dataset_excludes_climb_and_contains_original_sit_motion_sets():
    data=yaml.safe_load((ROOT/'tokenhsi/data/dataset_tokenhsi_sit_only.yaml').read_text())['motions']
    assert set(data)=={'loco','sit'} and len(data['loco'])==12 and len(data['sit'])==20
    assert not any('climb' in row['file'] for rows in data.values() for row in rows)
    cfg=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/tokenhsi_sit_full_ft.yaml').read_text())['env']
    assert cfg['numAgents']==cfg['numObjects']==1 and cfg['episodeLength']==300
    assert cfg['skillInitProb'][:2]==[.5,.5] and not any(cfg['skillInitProb'][2:])
    assert 'native_rsi' not in cfg['chairFT'] and not cfg['chairFT']['chair_front_only']

def test_original_sit_reward_is_ast_identical():
    original=ROOT.parent/'TokenHSI/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py'
    copied=ROOT/'tokenhsi/utils/tokenhsi_sit_reward.py'
    def fn(path):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compute_sit_reward')
    assert ast.dump(fn(original),include_attributes=False)==ast.dump(fn(copied),include_attributes=False)

def test_original_loco_and_sit_chair_placement_rules():
    t=HumanoidTokenHSISitFT.__new__(HumanoidTokenHSISitFT);t.device='cpu'
    t._reset_ref_slots={'loco':(torch.tensor([0,1]),torch.tensor([0,0])),'sit':(torch.tensor([2,3]),torch.tensor([0,0]))}
    t._reset_ref_motion_ids={'sit':torch.tensor([0,0])}
    class Ref:
        def get_obj_motion_state_single_frame(self,ids):return torch.tensor([[2.,3.,7.]]).repeat(len(ids),1),torch.tensor([[0.,0.,0.,1.]]).repeat(len(ids),1)
    t._motion_lib={'sit':Ref()};t._env_origins=torch.tensor([[10.,0.,0.]]).repeat(4,1);t._agent_spawn_offsets=torch.zeros(1,3)
    t._humanoid_root_states=torch.zeros(4,1,13);t._humanoid_root_states[...,0]=10
    t._box_states=torch.zeros(4,1,13);t._box_size=torch.full((4,1,3),.8)
    t._platform_pos=torch.zeros(4,1,3);t._platform_default_pos=torch.zeros_like(t._platform_pos)
    t._reset_boxes(torch.arange(4))
    distance=(t._box_states[:2,0,:2]-t._humanoid_root_states[:2,0,:2]).norm(dim=-1)
    assert ((distance>=1)&(distance<=5)).all()
    torch.testing.assert_close(t._box_states[2:,0,:2],torch.tensor([[12.,3.],[12.,3.]]))
    torch.testing.assert_close(t._box_states[:,0,2],torch.full((4,),.4))
    assert not t._box_states[...,7:].any()
