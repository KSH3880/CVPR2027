"""Original viewer loop using the SAME local four-person expansion as evaluation."""
import os,json,runpy
from pathlib import Path
# Imports Gym before Torch and installs validated output/count/reset hooks.
import five_task_scaling as base
import torch
from isaacgym import gymapi
from learning.multi_agent.ma_players import MAPlayerContinuous
assert os.environ['PROBE_AGENTS']=='4'
gpu=int(os.environ['TOKENHSI_GUI_GPU_INDEX']);torch.cuda.set_device(gpu)
player_init=MAPlayerContinuous.__init__
def init(self,config):return player_init(self,dict(config,device_name=f'cuda:{gpu}'))
MAPlayerContinuous.__init__=init
player_run=MAPlayerContinuous.run
def run(self):
 self.model.eval();self.running_mean_std.eval();self._reward_debug=False
 action_original=self.get_action;first=True
 def action(*args,**kwargs):
  nonlocal first
  value=action_original(*args,**kwargs)
  if first:
   task=self.env.task;assert task.num_agents==4 and task.num_objects==8
   assert self.model.a2c_network.actor_encoder.entity_counts==[4,8,4]
   if task.viewer is not None:
    points=torch.cat((task._humanoid_root_states[0,:,:3],task._box_states[0,:,:3],task._tar_pos[0]),0).detach().cpu()
    low,high=points.min(0).values,points.max(0).values;center=(low+high)/2;span=float((high[:2]-low[:2]).max())
    task.gym.viewer_camera_look_at(task.viewer,None,gymapi.Vec3(float(center[0]),float(center[1])-max(24.,span*.95),max(14.,span*.72)),gymapi.Vec3(float(center[0]),float(center[1]),.8))
    task._update_camera=lambda:None
   out=Path(task.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
   metadata=dict(checkpoint=task.cfg['args'].checkpoint,agents=4,objects=8,physics_device=str(task.device),actor_device=str(next(self.model.a2c_network.actor_encoder.parameters()).device),attention_intervention=False,camera='initial overview; manual control afterwards')
   (out/'viewer_verification.json').write_text(json.dumps(metadata,indent=2));print('PUSH7000_VIEW_VERIFIED',json.dumps(metadata),flush=True)
   first=False
  return value
 self.get_action=action
 return player_run(self)
MAPlayerContinuous.run=run
runpy.run_path(str(base.SOURCE/'tokenhsi/run.py'),run_name='__main__')
