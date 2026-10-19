"""Short simulator integration check. Evaluation environments, no training."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'tokenhsi')]
from isaacgym import gymapi
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.parse_task import parse_task
from utils.lift_spec import ARM_FEATURES
args=get_args();cfg,train,_=load_cfg(args);set_seed(42,False)
if args.motion_file:cfg['env']['motion_file']=args.motion_file
p=parse_sim_params(args,cfg,train)
task,env=parse_task(args,cfg,train,p)
try:
    ids=torch.arange(task.num_envs,device=task.device)
    task.reset(ids)
    assert task.num_agents==task.num_objects==2
    assert torch.isfinite(task.obs_buf).all()
    if task._dual_amp:
        assert task._amp_obs_buf[...,list(ARM_FEATURES)].abs().sum()>0
        assert torch.isfinite(task._amp_cond_buf).all()
    else:assert not task._amp_obs_buf[...,list(ARM_FEATURES)].any()
    assert (task._tar_pos[:,:,2]==task._lift_cfg['target_height']).all()
    assert (task._box_states[:,:,2]>.6).all()
    assert ((task._humanoid_root_states[:,:,:2]-task._box_states[:,:,:2]).norm(dim=-1)>1.2).all()
    assert not task._lift_achieved.any() and not task._lift_hold_time.any()
    demo=task.fetch_amp_obs_demo(32).reshape(32,10,129)
    assert torch.isfinite(demo).all()
    if task._dual_amp:
        assert demo[...,list(ARM_FEATURES)].abs().sum()>0
        full=task.fetch_amp_obs_demo_full(32).reshape(32,10,129)
        assert torch.isfinite(full).all() and full[...,list(ARM_FEATURES)].abs().sum()>0
        torch.testing.assert_close(task._amp_cond_buf,task._amp_cond_buf[:,:,0,None].expand_as(task._amp_cond_buf))
    else:assert not demo[...,list(ARM_FEATURES)].any()
    for _ in range(8):
        actions=torch.zeros(task.num_envs*2,task.num_dof,device=task.device)
        before_cond=task._amp_cond_buf.clone() if task._dual_amp else None
        task.step(actions)
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
        if task._dual_amp:
            torch.testing.assert_close(task._amp_cond_buf[:,:,1:],before_cond[:,:,:-1])
            torch.testing.assert_close(task.extras['amp_cond'],task._amp_cond_buf.reshape(-1,10))
        else:assert not task._amp_obs_buf[...,list(ARM_FEATURES)].any()
    before=task._humanoid_root_states[1:].clone()
    task._lift_achieved[0]=True;task._lift_hold_time[0]=1
    task.reset(ids[:1])
    assert not task._lift_achieved[0].any() and not task._lift_hold_time[0].any()
    torch.testing.assert_close(before,task._humanoid_root_states[1:])
    if task._dual_amp:
        torch.testing.assert_close(task._amp_cond_buf[0],task._amp_cond_buf[0,:,0,None].expand_as(task._amp_cond_buf[0]))
    print('LIFT_INTEGRATION_PASS',task.obs_buf.shape,task.rew_buf.shape,flush=True)
finally:
    if task.viewer is not None:task.gym.destroy_viewer(task.viewer)
    task.gym.destroy_sim(task.sim)
