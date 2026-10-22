"""Frozen original Y2m policy: first-episode fall and goal-reaching audit."""
import os,sys,runpy,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PORT=ROOT/'ports/koo_native_y2m'
sys.path.insert(0,str(PORT/'tokenhsi'))
import isaacgym
import torch
if os.environ.get('PROBE_AGENTS')=='4':
 from native4_y2m import install
 install()
from learning.multi_agent.ma_players import MAPlayerContinuous

if '--rl_device' in sys.argv and sys.argv[sys.argv.index('--rl_device')+1]=='cpu':
 original_player_init=MAPlayerContinuous.__init__
 def cpu_player_init(self,config):
  config=dict(config,device_name='cpu')
  original_player_init(self,config)
 MAPlayerContinuous.__init__=cpu_player_init

def digest(self):
 h=hashlib.sha256()
 for d in [self.model.state_dict(),self.running_mean_std.state_dict()]:
  for k,v in sorted(d.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def evaluate(self):
 task=self.env.task;n,m=task.num_envs,task.num_agents
 o=task.num_objects
 assert m in (2,4) and o in (m,8)
 assert task.obs_buf.shape[1]==248*m+37*o
 assert torch.equal(task._agent_box_assignment.sort(-1).values,torch.arange(m,device=task.device).expand(n,-1))
 assert self.model.a2c_network.actor_encoder.entity_counts==[m,o,m]
 assert task.relation_runtime.graph.num_agents==m and task.relation_runtime.graph.num_objects==o
 self.model.eval();self.running_mean_std.eval();self._eval=False;self._reward_debug=False
 out=Path(task.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
 before=digest(self)
 cpu_validation=None
 if m==4 and '--rl_device' in sys.argv and sys.argv[sys.argv.index('--rl_device')+1]=='cpu':
  from native4_y2m import validate_cpu
  cpu_validation=validate_cpu(task)
 obs=self.env_reset();self.get_batch_size(obs['obs'],1)
 if cpu_validation is not None and o==m:
  from native4_y2m import compare_pair_readout
  cpu_validation['attention_readout']=compare_pair_readout(self,obs)
 recorder=None
 if os.environ.get('JOINTPUSH_BIAS_CORRECTION')=='1':
  from bias_correction import BiasCorrection
  recorder=BiasCorrection(self,out)
 elif os.environ.get('JOINTPUSH_ATTENTION_ABLATION')=='1':
  from attention_ablation import AttentionAblation
  recorder=AttentionAblation(self,out)
 boxes=task._owned_push_states()[...,:3].clone();goals=task._tar_pos.clone()
 start_distance=(boxes[...,:2]-goals[...,:2]).norm(dim=-1)
 active=torch.ones(n,dtype=torch.bool,device=task.device)
 ever=torch.zeros(n,m,dtype=torch.bool,device=task.device);contact=torch.zeros_like(ever)
 first=torch.full((n,m),-1,device=task.device,dtype=torch.long)
 minimum=start_distance.clone();trials=[];traces=[]
 horizon=int(os.environ.get('PROBE_STEPS','600'))
 for step in range(1,horizon+1):
  if recorder is not None:recorder.begin(step-1,active)
  with torch.no_grad():action=self.get_action(obs,True)
  if recorder is not None:recorder.end()
  assert torch.isfinite(action).all()
  obs,_,done,_=self.env_step(self.env,action)
  if not isinstance(obs,dict):obs=self.obs_to_torch(obs)
  assert torch.isfinite(obs['obs']).all()
  distance=(task._owned_push_states()[...,:2]-goals[...,:2]).norm(dim=-1)
  success=task.relation_runtime.done.bool()
  first=torch.where(active[:,None]&success&~ever,torch.full_like(first,step),first)
  ever |= active[:,None]&success
  contact |= active[:,None]&task.relation_runtime.push_contact.bool()
  minimum=torch.where(active[:,None],torch.minimum(minimum,distance),minimum)
  fall=task._terminate_buf.reshape(n,m).bool()
  ended=active&(done.reshape(n,m).any(-1)|(step==horizon))
  for env in ended.nonzero().flatten().tolist():
   trials.append(dict(scene=env,steps=step,scene_fall=bool(fall[env].any()),fallen_agents=fall[env].nonzero().flatten().tolist(),initial_goal_distance=start_distance[env].tolist(),final_goal_distance=distance[env].tolist(),min_goal_distance=minimum[env].tolist(),ever_success=ever[env].tolist(),final_success=success[env].tolist(),first_success_step=first[env].tolist(),ever_hand_contact=contact[env].tolist(),root_heights=task._humanoid_root_states[env,:,2].tolist()))
  active &= ~ended
  if step in [1,30,60,120,300,599]:
   traces.append(dict(step=step,active=active.cpu().tolist(),distance=distance.cpu().tolist(),root_positions=task._humanoid_root_states[...,:3].cpu().tolist()))
  if step%60==0:print('EVAL_STEP',step,'active',int(active.sum()),'fallen scenes',sum(r['scene_fall'] for r in trials),flush=True)
  if not active.any():break
 assert len(trials)==n and before==digest(self)
 mean=lambda key:sum(sum(t[key]) for t in trials)/(n*m)
 summary=dict(cpu_validation=cpu_validation,agents=m,boxes=o,scenes=n,horizon=horizon,seed=task.cfg['args'].seed,checkpoint=task.cfg['args'].checkpoint,initialization='original eval loco only',model_and_rms_unchanged=True,scene_falls=sum(t['scene_fall'] for t in trials),scene_fall_rate=sum(t['scene_fall'] for t in trials)/n,agent_ever_goal_success_rate=mean('ever_success'),agent_final_goal_success_rate=mean('final_success'),scene_all_agents_ever_goal_success_rate=sum(all(t['ever_success']) for t in trials)/n,agent_ever_hand_contact_rate=mean('ever_hand_contact'),mean_initial_goal_distance_m=mean('initial_goal_distance'),mean_final_goal_distance_m=mean('final_goal_distance'),mean_min_goal_distance_m=mean('min_goal_distance'),success_definition='Original runtime.done/CONTACT+AT: exp(-10*3D_distance^2)>=0.9 plus source z/tilt conditions',fall_definition='Original per-agent terminate flag; scene counted at first termination, remaining agents censored',trials=trials)
 if recorder is not None:summary['attention_ablation']=recorder.finish()
 (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'traces.json').write_text(json.dumps(traces))
 print('SCALING_EVAL_RESULT',json.dumps({k:v for k,v in summary.items() if k!='trials'}),flush=True)
MAPlayerContinuous.run=evaluate
runpy.run_path(str(PORT/'tokenhsi/run.py'),run_name='__main__')
