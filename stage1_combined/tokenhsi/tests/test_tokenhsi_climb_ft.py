"""Guard the original CLIMB reward and reset geometry."""
from pathlib import Path
import ast
import isaacgym
import torch
from env.tasks.multi_agent.humanoid_tokenhsi_climb_ft import HumanoidTokenHSIClimbFT

ROOT=Path(__file__).resolve().parents[2]

def test_original_climb_reward_is_ast_identical():
    paths=[ROOT.parent/'TokenHSI/tokenhsi/env/tasks/basic_interaction_skills/humanoid_climb.py',ROOT/'tokenhsi/utils/tokenhsi_climb_reward.py']
    nodes=[next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compute_climb_reward') for p in paths]
    assert ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False)

def test_loco_distance_and_climb_reference_object_pose():
    t=HumanoidTokenHSIClimbFT.__new__(HumanoidTokenHSIClimbFT);t.device='cpu'
    t._reset_ref_slots={'loco':(torch.arange(1000),torch.zeros(1000,dtype=torch.long)),'climb':(torch.tensor([1000]),torch.tensor([0]))}
    t._reset_ref_motion_ids={'climb':torch.tensor([0])}
    class Ref:
        def get_obj_motion_state_single_frame(self,ids):return torch.tensor([[2.,3.,7.]]),torch.tensor([[0.,0.,0.,1.]])
    t._motion_lib={'climb':Ref()};t._env_origins=torch.tensor([[10.,0.,0.]]).repeat(1001,1);t._agent_spawn_offsets=torch.zeros(1,3)
    t._humanoid_root_states=torch.zeros(1001,1,13);t._humanoid_root_states[...,0]=10
    t._box_states=torch.zeros(1001,1,13);t._box_size=torch.full((1001,1,3),.8)
    t._platform_pos=torch.zeros(1001,1,3);t._platform_default_pos=torch.zeros_like(t._platform_pos)
    t._reset_boxes(torch.arange(1001))
    delta=t._box_states[:1000,0,:2]-t._humanoid_root_states[:1000,0,:2]
    assert ((delta.norm(dim=-1)>=1)&(delta.norm(dim=-1)<=10)).all()
    assert (delta[:,0]<0).any() and (delta[:,0]>0).any()
    torch.testing.assert_close(t._box_states[1000,0,:3],torch.tensor([12.,3.,.4]))
    assert not t._box_states[...,7:].any()

def test_original_iet_and_fall_height_without_contact_requirement():
    t=HumanoidTokenHSIClimbFT.__new__(HumanoidTokenHSIClimbFT)
    t._is_eval=False;t.reset_buf=torch.zeros(3,dtype=torch.long);t.progress_buf=torch.full((3,),10,dtype=torch.long)
    t._terminate_buf=torch.zeros(3,dtype=torch.long);t._contact_forces=torch.zeros(3,1,4,3)
    t._contact_body_ids=torch.tensor([2,3]);t._rigid_body_pos=torch.ones(3,1,4,3)
    t._rigid_body_pos[0,0,0,2]=.01;t.max_episode_length=300;t._enable_early_termination=True
    t._termination_heights=torch.full((4,),.15);t._climb_iet_counts=torch.tensor([0,59,0])
    t._compute_reset()
    assert t.reset_buf.tolist()==[1,1,0] and t._terminate_buf.tolist()==[1,0,0]
    t._is_eval=True;t._compute_reset()
    assert t.reset_buf.tolist()==[1,0,0]
