"""Trace the unmodified public Table player, native observations and simulator."""
from pathlib import Path
import json
import os
import sys

CODE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(CODE/'tokenhsi'),str(CODE)]
import isaacgym
import torch
import numpy as np
import yaml
# The shipped torch2.0 fused attention kernels do not support RTX5090.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
from learning.transformer.trans_players import TransPlayerContinuous

OUT=Path(os.environ['TOKENHSI_NATIVE_PUBLIC_OUTPUT'])
restore_original=TransPlayerContinuous.restore
run_original=TransPlayerContinuous.run_eval


def snapshot(player):
    return {name:{key:value.detach().cpu().clone() for key,value in module.state_dict().items()}
            for name,module in [('model',player.model),('running_mean_std',player.running_mean_std),('amp_input_mean_std',player._amp_input_mean_std)]}


def restore(player,path):
    result=restore_original(player,path)
    source=torch.load(path,map_location='cpu',weights_only=False)
    player._native_frozen=snapshot(player)
    for name,values in player._native_frozen.items():
        assert set(values)==set(source[name]),name
        for key,value in values.items():assert torch.equal(value,source[name][key]),key
    player.model.requires_grad_(False);player.model.eval();player.running_mean_std.eval();player._amp_input_mean_std.eval()
    player._native_epoch=int(source['epoch'])
    assert len(source['model'])==102 and tuple(player.running_mean_std.running_mean.shape)==(265,)
    print('PUBLIC_NATIVE_STRICT',player._native_epoch,len(source['model']),265,flush=True)
    return result


def traced_run(player):
    task=player.env.task
    assert task.max_episode_length==600 and abs(float(task.dt)-1/30)<1e-7
    assert task._is_eval and task._eval_task=='Table_Circle' and player.is_determenistic
    assert not task._enable_task_mask_obs
    assert task.num_envs in [4,512]
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'resolved_env.yaml').write_text(yaml.safe_dump({key:task.cfg[key] for key in ['env','sim']},sort_keys=False))
    original_reset=player.env_reset;original_step=player.env_step
    state=dict(repeat=-1,trace=None,initial=None,frames=0)
    def cpu(value):return value.detach().cpu().numpy().copy()
    def progress(status):
        (OUT/'progress.json').write_text(json.dumps(dict(status=status,repeat=state['repeat'],num_repeats=3,
            frame=state['frames'],max_steps=600,num_envs=task.num_envs,seed=0),indent=2))
    def flush():
        if state['trace'] is not None:
            np.savez_compressed(OUT/f'trace_{state["repeat"]}.npz',**{key:np.stack(value) for key,value in state['trace'].items()})
            np.savez_compressed(OUT/f'initial_{state["repeat"]}.npz',**state['initial'])
            print('PUBLIC_NATIVE_TRACE',state['repeat'],state['frames'],flush=True)
    def reset(ids=None):
        if ids is None:
            flush();state.update(repeat=state['repeat']+1,frames=0,
                trace={key:[] for key in ['objects','goals','success','precision','done','terminate']})
        obs=original_reset(ids)
        if ids is None:
            for field in ['_humanoid_actor_ids','_box_actor_ids','_platform_actor_ids','_tar_platform_actor_ids']:
                indices=getattr(task,field)
                assert indices.dtype==torch.int32 and int(indices.min())>=0 and int(indices.max())<task._root_states.shape[0],field
            state['initial']=dict(root=cpu(task._humanoid_root_states),dof_pos=cpu(task._dof_pos),dof_vel=cpu(task._dof_vel),
                objects=cpu(task._box_states),goals=cpu(task._tar_pos),sizes=cpu(task._obj_lib._every_env_object_bbox_lengths),
                up=cpu(task._obj_lib._every_env_object_up_facings),model_ids=cpu(task._obj_lib._every_env_object_ids))
            assert obs['obs'].shape==(task.num_envs,265) and torch.isfinite(obs['obs']).all()
            if state['repeat']==0:torch.save(obs['obs'].detach().cpu(),OUT/'first_native_observation.pth')
            progress('running');print('PUBLIC_NATIVE_REPEAT_START',state['repeat'],flush=True)
        return obs
    def step(env,action):
        assert torch.isfinite(action).all()
        result=original_step(env,action);obs,reward,done,info=result
        state['frames']+=1;assert state['frames']<=600
        values=dict(objects=task._box_states,goals=task._tar_pos,success=info['success'],precision=info['precision'],done=done,terminate=info['terminate'])
        for key,value in values.items():state['trace'][key].append(cpu(value))
        if state['frames']%100==0:
            progress('running');print('PUBLIC_NATIVE_FRAME',state['repeat'],state['frames'],flush=True)
        return result
    player.env_reset=reset;player.env_step=step
    try:
        with torch.no_grad():run_original(player)
        flush();assert state['repeat']==2
        current=snapshot(player)
        for name,values in player._native_frozen.items():
            for key,value in values.items():assert torch.equal(value,current[name][key]),key
        (OUT/'completed.json').write_text(json.dumps(dict(rc=0,num_envs=task.num_envs,repeats=3,seed=0,
            native_environment=True,original_player_loop=True,original_player_get_action=True,
            native265d_observations=True,model_and_rms_frozen=True,source_epoch=player._native_epoch,strict_model_tensors=102),indent=2))
        progress('complete')
    finally:player.env_reset=original_reset;player.env_step=original_step

TransPlayerContinuous.restore=restore
TransPlayerContinuous.run_eval=traced_run
import run as app
try:app.main()
finally:
    if torch.cuda.is_initialized():torch.cuda.synchronize()
