"""Audit actual actor topology, RSI, partial resets and PUSH simulation."""
from isaacgym import gymapi
from pathlib import Path
import json
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush

torch.set_num_threads(1)
args=get_args();cfg,train,_=load_cfg(args);cfg['args']=args
cfg['env']['motion_file']=args.motion_file
cfg['_size_audit']=True
set_seed(42)
task=HumanoidMAOpenPush(cfg,parse_sim_params(args,cfg,train),args.physics_engine,args.device,args.device_id,True)
ids=torch.arange(task.num_envs,device=task.device)
try:
    actor_names=[task.gym.get_actor_name(task.envs[0],i) for i in range(task.gym.get_actor_count(task.envs[0]))]
    assert not any('door' in name.lower() for name in actor_names)
    assert sum(name.startswith('push_box') for name in actor_names)==2
    assert task._dof_state.shape[0]//task.num_envs==2*task.num_dof
    assert task._rigid_body_state.shape[0]//task.num_envs==2*task.num_bodies+2+(2 if task._enable_markers else 0)
    assert set(task._motion_lib)=={'loco','push'}
    assert not hasattr(task,'_door_asset') and not hasattr(task,'_door_dof_state') and not hasattr(task,'_hold_spec')
    interaction=late=resets=0
    for _ in range(20):
        task._reset_envs(ids)
        interaction+=int(task._rsi_interaction.sum());late+=int(task._rsi_late.sum());resets+=2*len(ids)
        assert not task.relation_runtime.done.any()
        torch.testing.assert_close(task._physical_push_states,task._box_states[:,2:])
        assert torch.isfinite(task.obs_buf).all() and task.obs_buf.shape==(task.num_envs,570)
    assert .4<interaction/resets<.6
    for _ in range(12):
        task.pre_physics_step(torch.zeros(task.num_envs*2,32,device=task.device))
        task._physics_step();task.post_physics_step()
        torch.testing.assert_close(task._physical_push_states,task._box_states[:,2:])
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all() and torch.isfinite(task._amp_obs_buf).all()
    before=task._physical_push_states[1:].clone();amp=task._amp_obs_buf[1:].clone()
    task._reset_envs(ids[:1])
    torch.testing.assert_close(before,task._physical_push_states[1:]);torch.testing.assert_close(amp,task._amp_obs_buf[1:])
    demo=task.fetch_amp_obs_demo(64)
    assert torch.isfinite(demo).all()
    assert (demo.reshape(64,task._num_amp_obs_steps,-1)[...,-2:].argmax(-1)==1).all()
    metrics=task.consume_relation_diagnostics();assert not any(k.startswith('open/') for k in metrics)
    result=dict(actor_names=actor_names,physical_boxes=2,physical_doors=0,extra_object_dofs=0,
        policy_width=task.obs_buf.shape[-1],amp_width=task.get_num_amp_obs(),motion_libraries=sorted(task._motion_lib),
        interaction_fraction=interaction/resets,late_request_fraction=late/interaction,partial_reset_preserved=True,
        side_count=len(torch.unique(task._box_size[:,2:,0])),simulation_steps=12)
    path=Path(args.output_path);path.mkdir(parents=True,exist_ok=True)
    (path/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
finally:task.gym.destroy_sim(task.sim)
