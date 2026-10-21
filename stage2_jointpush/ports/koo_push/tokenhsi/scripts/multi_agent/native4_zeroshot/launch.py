"""Frozen-policy fall comparison. No learning or checkpoint writes."""
import os,sys,runpy,json,hashlib
from pathlib import Path
import isaacgym
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from learning.multi_agent.ma_players import MAPlayerContinuous
from utils.door_reference import rotate,multiply,yaw_from_quat
mode=os.environ['PROBE_SPAWN'];count=int(os.environ['PROBE_AGENTS']);out=Path(os.environ['PROBE_OUT'])
def validate_source(self, checkpoint):
 from utils.relation_task_spec import check_checkpoint_metadata,checkpoint_metadata
 check_checkpoint_metadata(checkpoint,checkpoint_metadata(self._relation_cfg))
 saved=dict(checkpoint['open_push_contract']);actual=self.get_open_push_contract()
 # This loco-only diagnostic uses the installed converted motion manifest;
 # interaction RSI is disabled, so the old screened interaction bank is unused.
 for key in ('manifest_sha256',):
  saved.pop(key,None);actual.pop(key,None)
 if saved!=actual:raise ValueError('Source physics/reward contract differs outside documented diagnostic overrides')
 print('NATIVE_STAGE1_SCALED4: 4 real doors +4 boxes, Start initialization, original Stage1 weights',flush=True)
HumanoidMAOpenPush.validate_inference_checkpoint=validate_source
# Native evaluation never samples the interaction RSI bank: its reset branches
# to loco whenever _is_eval=True. Skip loading missing unused interaction demos.
import utils.open_push_rsi as rsi_module
rsi_module.bank_contract=lambda env: {}
rsi_module.OpenPushRSIBank=lambda *args, **kwargs: None
def digest(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def evaluate(self):
 task=self.env.task
 self.model.eval();self.running_mean_std.eval();self._eval=False;self._reward_debug=False
 before=digest(self.model);trials=[]
 obs=self.env_reset();self.get_batch_size(obs['obs'],1)
 if self.is_rnn:self.init_rnn()
 n=task.num_envs
 active=torch.ones(n,dtype=torch.bool,device=task.device)
 touched=torch.zeros_like(active)
 rows=torch.arange(n,device=task.device)[:,None]
 horizon=int(os.environ.get('PROBE_STEPS','300'))
 for step in range(1,horizon+1):
  with torch.no_grad():action=self.get_action(obs,True)
  obs,reward,done,info=self.env_step(self.env,action)
  if not isinstance(obs,dict):obs=self.obs_to_torch(obs)
  roots=task._humanoid_root_states
  ids=task._agent_box_assignment if count==4 else task._push_binding+2
  boxes=task._box_states[rows,ids]
  sizes=task._box_size[rows,ids] if count==4 else roots.new_tensor(task.cfg['env']['pushBox']['size']).expand(n,count,-1)
  inv=boxes[...,3:7].clone();inv[...,:3]*=-1
  positions=task._rigid_body_pos
  local=rotate(inv[:,:,None].expand(n,count,positions.shape[2],4),positions-boxes[:,:,None,:3])
  body_clearance=(local.abs()-sizes[:,:,None]/2).clamp_min(0).norm(dim=-1).amin(-1)
  touched|=(body_clearance<.2).any(-1)&active
  fall=task._terminate_buf.reshape(n,count).bool()
  finished=active&(done.reshape(n,count).any(-1)|(step==horizon))
  for env in finished.nonzero().flatten().tolist():
   record=dict(trial=env,steps=step,fallen_agents=fall[env].nonzero().flatten().tolist(),scene_fall=bool(fall[env].any()),body_proximity_before_end=bool(touched[env]),root_height=roots[env,:,2].tolist(),body_point_box_clearance=body_clearance[env].tolist())
   trials.append(record);print('PROBE_TRIAL',json.dumps(record),flush=True)
  active&=~finished
  if not active.any():break
 assert len(trials)==n
 assert digest(self.model)==before,'Policy changed during evaluation'
 result=dict(agents=count,objects=count*2,spawn=mode,trials=trials,scene_fall_rate=sum(r['scene_fall'] for r in trials)/len(trials),mean_first_end_steps=sum(r['steps'] for r in trials)/len(trials),far_scene_falls=sum(r['scene_fall'] and not r['body_proximity_before_end'] for r in trials),policy_unchanged=True,horizon_steps=horizon)
 out.write_text(json.dumps(result,indent=2)+'\n');print('PROBE_SUMMARY',json.dumps({k:v for k,v in result.items() if k!='trials'}),flush=True)
if os.environ.get('PROBE_VIEW')!='1':
 MAPlayerContinuous.run=evaluate
 if os.environ.get('NATIVE4_ATTENTION_SCOPE'):
  sys.path.insert(0,str(Path(__file__).parent))
  from attention_probe import evaluate_attention
  MAPlayerContinuous.run=evaluate_attention
 if os.environ.get('NATIVE4_MIXED_ATTENTION') == '1':
  sys.path.insert(0,str(Path(__file__).parent))
  from attention_capture import capture_run
  MAPlayerContinuous.run=capture_run
snapshot=runpy.run_path(str(Path(__file__).with_name('native4_task.py')))
Task=snapshot['HumanoidMAOpenPush']
# Reuse Carry's configured-object-count branch for this diagnostic only.
Task._partition_push=True
Task.validate_inference_checkpoint=validate_source
import utils.parse_task as task_parser
task_parser.HumanoidMAOpenPush=Task
import utils.open_push_spec as spec_module
_original_validation=spec_module.validate_env
_original_compile=spec_module.compile_graph
def validate_scaled(env):
 import copy
 proxy=copy.deepcopy(env);proxy['numAgents']=2;proxy['numObjects']=4
 return _original_validation(proxy)
spec_module.validate_env=validate_scaled
import env.tasks.multi_agent.open_push_reward as reward_module
_original_runtime_step=reward_module.OpenPushRuntime.step
from utils.edge_ontop_spec import expand_graph
def graph4(spec,m,o,device=None):
 g=_original_compile(spec,4,8,device)
 agents=torch.arange(4,device=device)
 g.edge_src[::2]=agents;g.edge_dst[::2]=8+agents
 g.edge_src[1::2]=8+agents;g.edge_dst[1::2]=12+agents
 return g
reward_module.compile_graph=graph4
reward_module.graph_for_tasks=lambda spec,tasks: expand_graph(graph4(spec,4,8,tasks.device),len(tasks))
def push_step(self,roots,hands,handles,angles,boxes,goals):
 assert self.tasks.eq(1).all(), 'Scaled diagnostic supports PUSH evaluation only'
 # The native OPEN branch is inactive in this PUSH-only comparison.
 self.elapsed[:],opened=reward_module.update_hold(self.elapsed,angles,self.contact & self.posture & self.open_eligible,self.config['door']['target_angle']-self.config['door']['angle_tolerance'],self.hold['duration'],self.dt)
 p=reward_module.push_reward(roots,hands,boxes,goals,self.start,self.push_eligible,self.config,self.size,self.push_contact,self.push_credit,self.push_reset_distance)
 local=p['state']+p['progress']+p['bonus']
 self.phi[:,::2]=p['push_phi'] if 'contact_at' in self.config else p['phi']
 self.phi[:,1::2]=p['phi']
 self.own_success[:,::2]=p['push_success'] if 'contact_at' in self.config else p['success']
 self.own_success[:,1::2]=p['success']
 self.done.copy_(p['success']);self.achieved|=self.own_success;self.error.copy_(p['error'])
 return dict(state=p['state'],progress=p['progress'],bonus=p['bonus'],reward=.9*local+.1*local.flip(-1),success=p['success'])
reward_module.OpenPushRuntime.step=push_step
if os.environ.get('NATIVE4_MIXED_ATTENTION') == '1':
 from mixed4 import install
 install(Task,reward_module,_original_runtime_step,graph4)
runpy.run_path('tokenhsi/scripts/multi_agent/partition_push_launch.py',run_name='__main__')
