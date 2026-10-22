"""Attach the validated actor bias hooks to the ORIGINAL viewer loop."""
import json
from pathlib import Path
import torch
from isaacgym import gymapi
from learning.multi_agent.ma_players import MAPlayerContinuous
from bias_correction import BiasCorrection

def install():
 original=MAPlayerContinuous.run
 def run(self):
  task=self.env.task;self.model.eval();self.running_mean_std.eval();self._reward_debug=False
  out=Path(task.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
  correction=BiasCorrection(self,out)
  assert correction.scope=='all_owner' and correction.target=='all_foreign'
  assert correction.layers=={0,1,2,3} and correction.strength==4 and not correction.hard_mask
  action_original=self.get_action;checked=False
  def get_action(*args,**kwargs):
   nonlocal checked
   if checked:return action_original(*args,**kwargs)
   correction.begin(0,torch.ones(task.num_envs,dtype=torch.bool,device=task.device))
   action=action_original(*args,**kwargs);correction.end()
   assert torch.isfinite(action).all()
   # Set an overview after the first real reset. Leave it under user control
   # afterwards; source agent-follow camera would move inside distant boxes.
   if task.viewer is not None:
    points=torch.cat((task._humanoid_root_states[0,:,:3],task._box_states[0,:,:3],task._tar_pos[0]),dim=0).detach().cpu()
    low,high=points.min(0).values,points.max(0).values
    center=(low+high)/2;span=float((high[:2]-low[:2]).max())
    distance=max(20.,span*.95)
    eye=gymapi.Vec3(float(center[0]),float(center[1])-distance,max(14.,span*.72))
    target=gymapi.Vec3(float(center[0]),float(center[1]),.8)
    task.gym.viewer_camera_look_at(task.viewer,None,eye,target)
    task._update_camera=lambda:None
   metadata=dict(agents=task.num_agents,boxes=task.num_objects,layers=[1,2,3,4],scope=correction.scope,target=correction.target,strength=4,hard_mask=False,layer_reconstruction_max_error=max(correction.errors),bias_correction_max_error=max(correction.bias_errors),actor_device=str(next(self.model.a2c_network.actor_encoder.parameters()).device),physics_device=str(task.device),hooks_remain_active=True,checkpoint=task.cfg['args'].checkpoint)
   (out/'viewer_bias_verification.json').write_text(json.dumps(metadata,indent=2))
   print('VIEWER_BIAS_VERIFIED',json.dumps(metadata),flush=True)
   checked=True
   return action
  self.get_action=get_action
  self._viewer_bias_correction=correction
  # Do not call finish(): it removes hooks and would silently disable the bias.
  return original(self)
 MAPlayerContinuous.run=run
