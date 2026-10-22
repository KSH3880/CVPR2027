"""Saved-config five-task checkpoint scaling audit. External source stays unchanged."""
import os,sys,json,runpy,hashlib,copy
from pathlib import Path
sys.dont_write_bytecode=True
SOURCE=Path('/home/hwanhee/koo_cvpr/stage1_fix/ports/ksh_stage1_stage2_base/push')
WORK=Path.cwd()
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(SOURCE/'tokenhsi'))
import isaacgym
import torch
# Retain the original output_etc/check enforcement under our local workspace.
import utils.output_paths as paths
resolve_original=paths.resolve_output_path
def resolve_local(experiment,mode,requested='',check=False,max_iterations=0,root=WORK):
 return resolve_original(experiment,mode,requested,check,max_iterations,root=WORK)
paths.resolve_output_path=resolve_local
# Explicit evaluation-only cardinality extension. Other contracts stay strict.
if int(os.environ.get('PROBE_AGENTS','2'))==4:
 from utils import five_task,stage1_push
 validate_original=five_task.validate_env
 def validate_four(env):
  assert env['numAgents']==4 and env['numObjects']==8
  source_env=copy.deepcopy(env);source_env['numAgents']=2;source_env['numObjects']=4
  return validate_original(source_env)
 five_task.validate_env=validate_four
 check_original=stage1_push.check_checkpoint
 def check_four(weights,env):
  assert env['numAgents']==4 and env['numObjects']==8
  source_env=copy.deepcopy(env);source_env['numAgents']=2;source_env['numObjects']=4
  return check_original(weights,source_env)
 stage1_push.check_checkpoint=check_four
 sample_original=stage1_push.sample_boxes_and_goals
 def sample_four(count,device,origins,env,sizes=None,epoch=0):
  if env['numAgents']==2:return sample_original(count,device,origins,env,sizes,epoch)
  assert sizes is not None and sizes.shape==(count,8,3)
  states=torch.zeros(count,8,13,device=device)
  # Original square footprint: the two original PUSH stations remain, and the
  # other two corners become the new agents' stations. Extra actors retain the
  # unchanged base reset (the source caller only assigns first M reset states).
  centers=states.new_tensor([[-8.,-8.],[8.,8.],[-8.,8.],[8.,-8.],[0.,-16.],[0.,16.],[-16.,0.],[16.,0.]])
  states[...,:2]=centers+(torch.rand(count,8,2,device=device)-.5)+origins[:,None,:2]
  states[...,2]=sizes[...,2]/2
  yaw=torch.rand(count,8,device=device)*2*3.141592653589793
  states[...,5]=(yaw/2).sin();states[...,6]=(yaw/2).cos()
  from utils.push_goal_curriculum import distance_range
  lo,hi=distance_range(env,epoch);distances=lo+(hi-lo)*torch.rand(count,4,device=device)
  forward=stage1_push.rotate(states[:,:4,3:7],states.new_tensor([0.,1.,0.]).expand(count,4,3))
  goals=states[:,:4,:3]+forward*distances[...,None]
  return states,goals,forward,distances
 stage1_push.sample_boxes_and_goals=sample_four

from learning.multi_agent.ma_players import MAPlayerContinuous
if '--rl_device' in sys.argv and sys.argv[sys.argv.index('--rl_device')+1]=='cpu':
 original_init=MAPlayerContinuous.__init__
 def init(self,config):original_init(self,dict(config,device_name='cpu'))
 MAPlayerContinuous.__init__=init

def digest(self):
 h=hashlib.sha256()
 for d in (self.model.state_dict(),self.running_mean_std.state_dict()):
  for key,v in sorted(d.items()):h.update(key.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def run(self):
 self.model.eval();self.running_mean_std.eval();self._eval=False;self._reward_debug=False
 task=self.env.task;n,m,o=task.num_envs,task.num_agents,task.num_objects
 assert o==2*m and m in (2,4)
 assert self.model.a2c_network.actor_encoder.entity_counts==[m,o,m]
 assert task.gym.get_env_dof_count(task.envs[0])==m*task.num_dof
 for ids in (task._humanoid_actor_ids,task._box_actor_ids):
  assert int(ids.min())>=0 and int(ids.max())<len(task._root_states)
  assert len(torch.unique(ids))==ids.numel()
 actor_count=task.gym.get_actor_count(task.envs[0])
 assert task.obs_buf.shape[1]==317*m
 before=digest(self);out=Path(task.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
 obs=self.env_reset();self.get_batch_size(obs['obs'],1)
 packet=obs['obs'][:,-5*m:].reshape(n,m,5)
 assert (packet[:,:,0]==1).all()
 tasks=packet[:,:,1].long().clone()
 preset=os.environ.get('TASK_GRAPH','push_at')
 if preset=='push_at':assert (tasks==4).all()
 else:assert preset=='random_scenario' and ((tasks>=0)&(tasks<=4)).all()
 assert torch.equal(packet[:,:,2].long(),torch.arange(m,device=task.device).expand(n,-1))
 assert torch.equal(packet[:,:,3].long()[tasks>=2],(m+torch.arange(m,device=task.device).expand(n,-1))[tasks>=2])
 assert torch.equal(packet[:,:,4].long()[(tasks==2)|(tasks==4)],(m+o+torch.arange(m,device=task.device).expand(n,-1))[(tasks==2)|(tasks==4)])
 assert task._box_states.shape[1]==o
 assert not task._push_rsi_interaction.any(),'Loco-only eval required'
 physical=[]
 for eid in range(min(n,2)):
  for i in range(m):
   handle=task._box_handles[eid*o+i]
   mass=sum(p.mass for p in task.gym.get_actor_rigid_body_properties(task.envs[eid],handle))
   friction=[p.friction for p in task.gym.get_actor_rigid_shape_properties(task.envs[eid],handle)]
   if int(tasks[eid,i])==4:
    assert abs(mass-20)<1e-3
    assert all(abs(f-task.cfg['env']['pushBox']['friction'])<1e-5 for f in friction)
   else:assert mass>0
   physical.append(dict(env=eid,object=i,task=int(tasks[eid,i]),mass=mass,friction=friction))
 active=torch.ones(n,dtype=torch.bool,device=task.device);trials=[]
 initial=dict(humans=task._humanoid_root_states.detach().cpu().tolist(),boxes=task._box_states.detach().cpu().tolist(),goals=task._tar_pos.detach().cpu().tolist(),sizes=task._box_size.detach().cpu().tolist())
 (out/'initial.json').write_text(json.dumps(initial))
 initial_xy=task._box_states[:,:m,:2].clone();goals=task._tar_pos.clone()
 initial_distance=(initial_xy-goals[...,:2]).norm(dim=-1)
 minimum_distance=initial_distance.clone()
 ever_success=torch.zeros(n,m,dtype=torch.bool,device=task.device)
 ever_runtime=torch.zeros_like(ever_success);ever_at=torch.zeros_like(ever_success);ever_contact=torch.zeros_like(ever_success);ever_force=torch.zeros_like(ever_success)
 from env.tasks.multi_agent.edge_context_reward import owner_sum
 from utils.edge_context_spec import AT
 from utils.edge_context_spec import PUSH_CONTACT
 horizon=int(os.environ.get('PROBE_STEPS','600'))
 for step in range(1,horizon+1):
  with torch.no_grad():action=self.get_action(obs,True)
  assert torch.isfinite(action).all()
  obs,_,done,_=self.env_step(self.env,action)
  if not isinstance(obs,dict):obs=self.obs_to_torch(obs)
  assert torch.isfinite(obs['obs']).all()
  distance=(task._box_states[:,:m,:2]-goals[...,:2]).norm(dim=-1)
  minimum_distance=torch.where(active[:,None],torch.minimum(minimum_distance,distance),minimum_distance)
  runtime=task.relation_runtime;graph=runtime.graph
  at=owner_sum((runtime.own_success & graph.edge_valid & (graph.edge_relation==AT)).float(),graph)>0
  contact=owner_sum((runtime.own_success & graph.edge_valid & (graph.edge_relation==PUSH_CONTACT)).float(),graph)>0
  ever_success |= active[:,None]&(at&contact)
  ever_runtime |= active[:,None]&runtime.done.bool()
  ever_at |= active[:,None]&at;ever_contact |= active[:,None]&contact
  ever_force |= active[:,None]&task._push_hits.bool()
  fall=task._terminate_buf.reshape(n,m).bool()
  ended=active&(done.reshape(n,m).any(-1)|(step==horizon))
  for eid in ended.nonzero().flatten().tolist():
   trials.append(dict(scene=eid,tasks=tasks[eid].cpu().tolist(),steps=step,seconds=step*task.dt,scene_fall=bool(fall[eid].any()),fallen_agents=fall[eid].nonzero().flatten().tolist(),root_heights=task._humanoid_root_states[eid,:,2].tolist(),ever_success=ever_success[eid].tolist(),ever_runtime_success=ever_runtime[eid].tolist(),ever_at=ever_at[eid].tolist(),ever_contact=ever_contact[eid].tolist(),ever_force=ever_force[eid].tolist(),initial_goal_distance=initial_distance[eid].tolist(),final_goal_distance=distance[eid].tolist(),minimum_goal_distance=minimum_distance[eid].tolist(),box_displacement_m=(task._box_states[eid,:m,:2]-initial_xy[eid]).norm(dim=-1).tolist(),net_goal_progress_m=(initial_distance[eid]-distance[eid]).tolist()))
  active &= ~ended
  if step%60==0:print('FALL_STEP',step,'active',int(active.sum()),'falls',sum(r['scene_fall'] for r in trials),flush=True)
  if not active.any():break
 assert len(trials)==n and before==digest(self)
 mean=lambda key:sum(sum(t[key]) for t in trials)/(n*m)
 summary=dict(agent_original_runtime_success_rate=mean('ever_runtime_success'),agent_goal_success_rate=mean('ever_at'),agent_joint_contact_goal_success_rate=mean('ever_success'),agent_contact_success_rate=mean('ever_contact'),agent_force_contact_rate=mean('ever_force'),mean_initial_goal_distance_m=mean('initial_goal_distance'),mean_final_goal_distance_m=mean('final_goal_distance'),mean_box_displacement_m=mean('box_displacement_m'),mean_net_goal_progress_m=mean('net_goal_progress_m'),all_agents_goal_scene_rate=sum(all(t['ever_at']) for t in trials)/n,attention_intervention=False,checkpoint=task.cfg['args'].checkpoint,checkpoint_epoch=int(os.environ["EXPECTED_EPOCH"]),agents=m,objects=o,assigned_boxes=m,extra_boxes=o-m,scenes=n,seed=task.cfg['args'].seed,horizon=horizon,dt=task.dt,scene_falls=sum(t['scene_fall'] for t in trials),scene_fall_rate=sum(t['scene_fall'] for t in trials)/n,model_rms_unchanged=True,finite=True,actor_count=actor_count,dof_count=m*task.num_dof,physics_sample=physical,task=preset+'; loco-only reset; no masks or training',task_counts={name:int((tasks==i).sum()) for i,name in enumerate(('SIT','CLIMB','CARRY_AT','CARRY_ONTOP','PUSH_AT'))},fall_definition='Original per-agent terminate flag, first scene termination. Other agents censored at first scene end.',expansion='source 2-person environment' if m==2 else 'local evaluation-only4 extension; four corners of original square footprint, native extra-box reset',trials=trials)
 (out/'result.json').write_text(json.dumps(summary,indent=2));print('FIVE_TASK_FALL_RESULT',json.dumps({k:v for k,v in summary.items() if k!='trials'}),flush=True)
if __name__=='__main__':
 MAPlayerContinuous.run=run
 runpy.run_path(str(SOURCE/'tokenhsi/run.py'),run_name='__main__')
