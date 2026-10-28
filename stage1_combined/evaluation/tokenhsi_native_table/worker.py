"""Run the original TokenHSI environment/player loop with a frozen scene policy."""
from pathlib import Path
import argparse
import ast
import datetime
import json
import os
import random
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tokenhsi'),str(ROOT)]
import isaacgym
from isaacgym import gymapi,gymutil
import torch
import numpy as np
import yaml
from env.tasks.adapt_interaction_skills.humanoid_adapt_carry_box2objs import HumanoidAdaptCarryBox2Objs
from env.tasks.vec_task_wrappers import VecTaskPythonWrapper
from evaluation.tokenhsi_native_table.policy import FrozenScenePolicy,pack_scene

p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--num-envs',type=int,default=512);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);run=ROOT.parent
cfg=yaml.safe_load((run/'native_env.yaml').read_text())
cfg['env']['numEnvs']=a.num_envs
cfg['env']['motion_file']='tokenhsi/data/dataset_loco_sit_carry_climb.yaml'
cfg['args']=SimpleNamespace(eval=True,eval_task='Table_Circle',record=False,output_path=str(out))
cfg['seed']=0;cfg['env']['seed']=0
# Original seed/simulator helpers, extracted without changing their bodies.
module=ast.parse((run/'original/config.py').read_text())
helpers=dict(torch=torch,np=np,random=random,os=os,gymapi=gymapi,gymutil=gymutil,SIM_TIMESTEP=1/60)
for name in ('parse_sim_params','set_seed'):
    node=next(x for x in module.body if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(run/'original/config.py'),'exec'),helpers)
sim_args=SimpleNamespace(slices=0,physics_engine=gymapi.SIM_PHYSX,device='cuda',use_gpu=True,subscenes=0,use_gpu_pipeline=True,num_threads=0)
params=helpers['parse_sim_params'](sim_args,cfg,{})
# Policy construction is outside the native environment RNG stream.
policy=FrozenScenePolicy(run/'checkpoint.pth','cuda:0')
helpers['set_seed'](0)
if os.environ.get('NATIVE_LAYOUT_ONLY')=='1':
    HumanoidAdaptCarryBox2Objs._load_motion=lambda *args:None
task=HumanoidAdaptCarryBox2Objs(cfg,params,gymapi.SIM_PHYSX,'cuda',0,True)
wrapped=VecTaskPythonWrapper(task,'cuda:0',float('inf'),1.)
assert task.max_episode_length==600 and abs(task.dt-1/30)<1e-7
n=task.num_envs
for field in ('_humanoid_actor_ids','_box_actor_ids','_platform_actor_ids','_tar_platform_actor_ids'):
    ids=getattr(task,field)
    print('ACTOR_INDEX',field,str(ids.dtype),ids.tolist(),'root_count',task._root_states.shape[0],flush=True)
    if os.environ.get('NATIVE_LAYOUT_ONLY')=='1':continue
    assert ids.dtype==torch.int32 and int(ids.min())>=0 and int(ids.max())<task._root_states.shape[0],field
print('NATIVE_ACTOR_LAYOUT',n,task._root_states.shape[0],flush=True)
if os.environ.get('NATIVE_LAYOUT_ONLY')=='1':raise SystemExit(0)

class PlayerBridge:
    def __init__(self):
        self.device=task.device;self.num_agents=1;self.is_determenistic=True
        self.env=SimpleNamespace(num_envs=n);self._checkpoint_fn=str(run/'policy/checkpoint.pth');self._eval_task='Table_Circle'
        self.repeat=-1;self.pending=torch.zeros(n,dtype=torch.bool,device=task.device)
        self.trace=None;self.initial=None

    def flush(self):
        if self.trace is not None:
            np.savez_compressed(out/f'trace_{self.repeat}.npz',**{k:np.stack(v) for k,v in self.trace.items()})
            np.savez_compressed(out/f'initial_{self.repeat}.npz',**self.initial)
            print('NATIVE_REPEAT_TRACE',self.repeat,flush=True)

    def env_reset(self,env_ids=None):
        if env_ids is None:
            self.flush();self.repeat+=1
            self.trace={k:[] for k in ('objects','goals','success','precision','done','terminate')}
        obs=wrapped.reset(env_ids)
        if env_ids is None:
            self.pending[:]=True
            self.initial=dict(root=task._humanoid_root_states.cpu().numpy().copy(),
                objects=task._box_states.cpu().numpy().copy(),goals=task._tar_pos.cpu().numpy().copy(),
                sizes=task._obj_lib._every_env_object_bbox_lengths.cpu().numpy().copy(),
                up=task._obj_lib._every_env_object_up_facings.cpu().numpy().copy())
        elif len(env_ids)>0:
            self.pending[env_ids]=True
        return {'obs':obs}

    def get_batch_size(self,obs,batch_size):return n

    def get_action(self,obs,is_determenistic):
        assert is_determenistic
        pos,rot,vel,ang=[v.clone() for v in (task._rigid_body_pos,task._rigid_body_rot,task._rigid_body_vel,task._rigid_body_ang_vel)]
        ids=self.pending.nonzero().flatten()
        if len(ids):
            kin=task._kinematic_humanoid_rigid_body_states[ids]
            for value,part in zip((pos,rot,vel,ang),(slice(0,3),slice(3,7),slice(7,10),slice(10,13))):value[ids]=kin[...,part]
        # Native TokenHSI resets reference positions directly into its shared local origin.
        origins=torch.zeros(n,3,device=task.device)
        packed=pack_scene(pos,rot,vel,ang,task._box_states,task._obj_lib._every_env_object_bps,task._tar_pos,origins)
        action=policy.action(packed);self.pending[:]=False
        if not (out/'first_action.pth').exists():torch.save(dict(obs=packed.cpu(),action=action.cpu()),out/'first_action.pth')
        return action

    def env_step(self,env,action):
        obs,reward,done,info=wrapped.step(action)
        values=dict(objects=task._box_states,goals=task._tar_pos,success=info['success'],precision=info['precision'],done=done,terminate=info['terminate'])
        for key,value in values.items():self.trace[key].append(value.detach().cpu().numpy().copy())
        return {'obs':obs},reward,done,info

source=run/'original/trans_players.py'
module=ast.parse(source.read_text());cls=next(x for x in module.body if isinstance(x,ast.ClassDef) and any(isinstance(f,ast.FunctionDef) and f.name=='run_eval' for f in x.body))
node=next(f for f in cls.body if isinstance(f,ast.FunctionDef) and f.name=='run_eval');node.decorator_list=[]
scope=dict(torch=torch,os=os,json=json,datetime=datetime)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
bridge=PlayerBridge()
try:
    with torch.no_grad():scope['run_eval'](bridge)
    bridge.flush();policy.check_frozen()
    (out/'completed.json').write_text(json.dumps(dict(rc=0,num_envs=n,repeats=3,seed=0,
        native_environment=True,original_player_loop=True,model_and_rms_frozen=True),indent=2))
finally:
    torch.cuda.synchronize()
