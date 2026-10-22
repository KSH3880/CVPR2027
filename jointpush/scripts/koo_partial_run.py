"""Process-local Stage1BasePush Partial Stage2 adapter; source files remain intact."""
import os,sys,runpy,json,copy,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'jointpush/ports/koo_1021/ports/stage1_base_push'
WORK=Path.cwd()
sys.path[:0]=[str(SOURCE/'tokenhsi'),str(SOURCE),str(ROOT/'stage2_jointpush/scripts')]
import isaacgym
import torch
import utils.output_paths as output_paths
_original_output=output_paths.resolve_output_path
output_paths.resolve_output_path=lambda experiment,mode,requested='',check=False,max_iterations=0,root=WORK:_original_output(experiment,mode,requested,check,max_iterations,root=WORK)
import utils.unified_training as unified
_original_validate=unified.validate_unified_env
def validate(env):
 if 'partitionPush' not in env:return _original_validate(env)
 assert (env['numAgents'],env['numObjects'])==(4,8)
 assert env['partitionPush']['rsi']=='loco_only'
 assert env['templateRsi']['PUSH_AT']==[1.]+[0.]*(len(env['skill'])-1)
 assert env['ampExpertMapping']['push']==[.5]+[0.]*7+[.5]
unified.validate_unified_env=validate
import utils.edge_context_spec as context
context.PUSH_CONTACT=12
from utils.edge_context_spec import EdgeContextGraph
import utils.edge_stage1_spec as stage1_spec
_original_compile=stage1_spec.compile_stage1_graph
def compile_graph(spec,m,o,device=None):
 if not spec.get('partial_push_adapter'):return _original_compile(spec,m,o,device)
 assert o==2*m
 src=torch.tensor([v for a in range(m) for v in (a,m+a)],device=device)
 dst=torch.tensor([v for a in range(m) for v in (m+a,m+o+a)],device=device)
 relation=torch.tensor([12,7]*m,device=device);owner=torch.arange(m,device=device).repeat_interleave(2)
 return EdgeContextGraph(tuple('edge'+str(i) for i in range(2*m)),m,o,src,dst,relation,owner,torch.ones(2*m,device=device,dtype=torch.bool),relation==7,torch.zeros(2*m,2*m,device=device,dtype=torch.bool),torch.full_like(src,-1))
stage1_spec.compile_stage1_graph=compile_graph
import utils.task_role_spec as roles
_original_sizes=roles.task_size_probabilities
def sizes(shape,ranges):
 result=torch.zeros(*shape.shape[:-1],6,device=shape.device);result[...,5]=1;return result
roles.task_size_probabilities=sizes
from ksh_partition_scene import HumanoidMAKshPartitionPush
from env.tasks.multi_agent.humanoid_ma_shared_support import LocoRsi
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
class HumanoidMAKooPartialPush(HumanoidMAKshPartitionPush):
 _amp_family_count=4
 _solo_cache_class=LocoRsi
 _load_motion=HumanoidMAStage1Push._load_motion
 _fetch_conditioned_amp_demo=HumanoidMAStage1Push._fetch_conditioned_amp_demo
 def _reset_actors(self,ids):
  # Reject only ambiguous positions; never reassign a sampled physical group.
  from ksh_partition_helpers import group_layout,sample_surround_spawn
  expected,_,_=group_layout(self._group_leaders[ids])
  rows=torch.arange(len(ids),device=self.device)[:,None]
  active=torch.cat((expected,self._ordinary_keep[ids]),-1)
  candidates=self._box_states[ids][rows,active]
  selected=self._box_states[ids][rows,expected]
  size=self._box_size[ids][rows,expected]
  for attempt in range(128):
   positions=self._spawn_roots[ids]
   nearest=torch.cdist(positions,candidates[...,:2]).argmin(-1)
   bad=(active.gather(1,nearest)!=expected).any(-1)
   if not bad.any():break
   pending=bad.nonzero().flatten()
   pos,yaw=sample_surround_spawn(selected[pending],size[pending],self._joint_push_spec['spawn_distance'],self._joint_push_spec.get('spawn_min_agent_distance',1.2))
   self._spawn_roots[ids[pending]]=pos;self._spawn_yaw[ids[pending]]=yaw
  else:raise RuntimeError('Cannot sample unambiguous assigned-box surround spawns')
  return super()._reset_actors(ids)
import utils.parse_task as parser
parser.HumanoidMAKooPartialPush=HumanoidMAKooPartialPush
# Validate and strictly copy Stage1 tensors, initializing only new coordination.
from learning.multi_agent import stage2_network_builder as stage2
from utils.stage1_push_spec import check_checkpoint as check_base
import yaml
base_env=yaml.safe_load((SOURCE/'tokenhsi/data/cfg/multi_agent/stage1_base_push.yaml').read_text())['env']
base_env['motion_file']='tokenhsi/data/dataset_stage1_base_push.yaml'
def transfer(model,weights):
 check_base(weights,base_env,for_training=True)
 source=weights['model'];target=model.state_dict();copied=[];added=[]
 for name,value in target.items():
  if name.startswith(('a2c_network.coordination.','a2c_network.critic_coordination.')):added.append(name);continue
  if name not in source or source[name].shape!=value.shape:raise ValueError('Incompatible Stage1 tensor: '+name)
  value.copy_(source[name]);copied.append(name)
 if set(source)-set(target):raise ValueError('Discarded Stage1 tensor')
 model.load_state_dict(target,strict=True);stage2.freeze_actor(model)
 print('PARTIAL_STAGE2_TRANSFER',len(copied),'copied',len(added),'new coordination tensors; Stage1 epoch',weights['epoch'],flush=True)
 return dict(source_epoch=weights['epoch'],copied=copied,added=added,stage1_optimizer_restored=False)
stage2.transfer_stage1=transfer
# New checkpoint identity includes physical, reward, graph and initialization contracts.
import utils.shared_support as shared
_original_contract=shared.checkpoint_contract
def contract(env):
 if 'partitionPush' not in env:return _original_contract(env)
 return dict(adapter='koo_partial_v1',source_commit='ec09d71994ef48e9d7cd4bf17264e8183274da51',partition=env['partitionPush'],template_rsi=env['templateRsi'],graph=env['relationGraph'],agents=4,objects=8)
shared.checkpoint_contract=contract
import utils.stage2_spec as stage2_spec
_original_check=stage2_spec.check_checkpoint
def check_partial(weights,env):
 if 'partitionPush' not in env:return _original_check(weights,env)
 if weights.get('shared_support_contract')!=contract(env):raise ValueError('Partial Joint Push Stage2 contract differs')
stage2_spec.check_checkpoint=check_partial
from learning.multi_agent.ma_agent import MAAgent
def digest(module):
 h=hashlib.sha256()
 for name,value in sorted(module.state_dict().items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
_init=MAAgent.__init__
def init(self,base_name,config):
 _init(self,base_name,config)
 task=self.vec_env.env.task
 self._relation_experiment_config['sim']=copy.deepcopy(task.cfg['sim'])
 Path(self.experiment_dir,'relation_config.yaml').write_text(yaml.safe_dump(self._relation_experiment_config,sort_keys=False))
 self._partial_actor_digest=digest(self.model.a2c_network.actor_encoder)
 self._partial_coord_digest=digest(self.model.a2c_network.coordination)
 self._partial_rms_digest=digest(self.running_mean_std)
 assert not any(p.requires_grad for p in self.model.a2c_network.actor_encoder.parameters())
 assert any(p.requires_grad for p in self.model.a2c_network.coordination.parameters())
 print('PARTIAL_STAGE2_TRAIN_READY',json.dumps(dict(agents=task.num_agents,objects=task.num_objects,envs=task.num_envs,observation=task.get_obs_size(),amp=task.get_num_amp_obs(),frozen_stage1_actor=True,coordination_trainable=True,rsi='loco_only',optimizer_resumed=task.cfg['args'].resume>0)),flush=True)
MAAgent.__init__=init
# Record trainable coordination gradients and verify the frozen encoder is unchanged.
_calc=MAAgent.calc_gradients
_seen=False
def gradients(self,*args,**kwargs):
 global _seen
 result=_calc(self,*args,**kwargs)
 if not _seen:
  branch=self.model.a2c_network.coordination
  values=[p.grad.detach().abs().sum() for p in branch.parameters() if p.grad is not None]
  norm=float(torch.stack(values).sum()) if values else 0.
  if not norm>0 or not torch.isfinite(torch.tensor(norm)):raise ValueError('Coordination gradient invalid')
  print('PARTIAL_STAGE2_COORDINATION_GRAD',norm,flush=True);_seen=True
 return result
MAAgent.calc_gradients=gradients
_epoch=MAAgent.train_epoch
def epoch(self):
 result=_epoch(self)
 if self.epoch_num==2:
  actor_unchanged=digest(self.model.a2c_network.actor_encoder)==self._partial_actor_digest
  rms_unchanged=digest(self.running_mean_std)==self._partial_rms_digest
  coord_changed=digest(self.model.a2c_network.coordination)!=self._partial_coord_digest
  assert actor_unchanged and rms_unchanged and coord_changed
  report=dict(epoch=self.epoch_num,actor_encoder_unchanged=actor_unchanged,normalizer_unchanged=rms_unchanged,coordination_updated=coord_changed,stage1_optimizer_restored=False)
  Path(self.experiment_dir,'startup_verification.json').write_text(json.dumps(report,indent=2))
  self.save(str(Path(self.experiment_dir,'nn','KooPartialJointPush_startup')))
  print('PARTIAL_STAGE2_UPDATES_VERIFIED',json.dumps(report),flush=True)
 return result
MAAgent.train_epoch=epoch
if '--test' in sys.argv:
 from learning.multi_agent.ma_players import MAPlayerContinuous
 _player_init=MAPlayerContinuous.__init__
 def player_init(self,config):
  if os.environ.get('PARTIAL_CPU')=='1':config=dict(config,device_name='cpu')
  elif os.environ.get('HEADLESS')=='0':config=dict(config,device_name='cuda:'+os.environ.get('TOKENHSI_GUI_GPU_INDEX','0'))
  return _player_init(self,config)
 MAPlayerContinuous.__init__=player_init
 def restore(self,path):
  weights=torch.load(path,map_location=self.device,weights_only=False)
  if 'shared_support_contract' in weights:
   check_partial(weights,self.env.task.cfg['env']);self.model.load_state_dict(weights['model'],strict=True)
  else:transfer(self.model,weights)
  self.running_mean_std.load_state_dict(weights['running_mean_std'],strict=True);self.running_mean_std.eval()
  if hasattr(self,'_amp_input_mean_std'):self._amp_input_mean_std.load_state_dict(weights['amp_input_mean_std'],strict=True)
  self.model.eval()
  print('PARTIAL_STAGE2_PLAYER_RESTORED',flush=True)
 MAPlayerContinuous.restore=restore
 _run=MAPlayerContinuous.run
 def run(self):
  original=self.get_action;steps=0;limit=int(os.environ.get('PARTIAL_CHECK_STEPS','0'))
  def action(*args,**kwargs):
   nonlocal steps
   value=original(*args,**kwargs)
   assert torch.isfinite(value).all() and torch.isfinite(self.env.task.obs_buf).all()
   steps+=1
   if limit and steps>limit:
    print('PARTIAL_STAGE2_CHECK_PASSED',limit,'finite physics/action steps',flush=True);raise SystemExit(0)
   return value
  self.get_action=action
  return _run(self)
 MAPlayerContinuous.run=run
runpy.run_path(str(SOURCE/'tokenhsi/run.py'),run_name='__main__')
