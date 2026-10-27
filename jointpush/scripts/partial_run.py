"""RawCuboid Partial Stage2 bootstrap. All external sources remain read-only."""
import os,sys,runpy,json,copy,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'jointpush/ports/jhh_raw_cuboid'
WORK=Path.cwd()
sys.path[:0]=[str(SOURCE/'tokenhsi'),str(SOURCE),str(ROOT)]
import isaacgym
import torch
if os.environ.get('HEADLESS')=='0':
 if os.environ.get('CUDA_MPS_PIPE_DIRECTORY'):raise ValueError('Native viewer must bypass training MPS')
 gui_index=int(os.environ['TOKENHSI_GUI_GPU_INDEX'])
 if gui_index>=torch.cuda.device_count():raise ValueError('Native viewer requires unfiltered CUDA enumeration')
 torch.cuda.set_device(gui_index)
 print('PARTIAL_NATIVE_GPU',gui_index,'CUDA devices',torch.cuda.device_count(),'MPS bypassed',flush=True)
import utils.output_paths as output_paths
_original_output=output_paths.resolve_output_path
output_paths.resolve_output_path=lambda experiment,mode,requested='',check=False,max_iterations=0,root=WORK:_original_output(experiment,mode,requested,check,max_iterations,root=WORK)
import utils.unified_training as unified
_original_validate=unified.validate_unified_env
def validate(env):
 if 'partitionPush' not in env:return _original_validate(env)
 assert (env['numAgents'],env['numObjects'])==(4,8)
 assert env['partitionPush']['rsi']=='loco_only'
 if env['partitionPush'].get('version')!=4 or env['partitionPush'].get('spawn_mode')!='goal_opposite_180':raise ValueError('Prepare a new rear180/random-assignment workspace before running')
 assert all(row==[1.]+[0.]*(len(env['skill'])-1) for row in env['templateRsi'].values())
 assert env['partialReward']['progress_reference']=='physical_shared_box_center'
unified.validate_unified_env=validate
import utils.edge_context_spec as context
context.PUSH_CONTACT=11
from utils.edge_context_spec import EdgeContextGraph
import utils.edge_stage1_spec as stage1_spec
_original_compile=stage1_spec.compile_stage1_graph
def compile_graph(spec,m,o,device=None):
 if not spec.get('partial_push_adapter'):return _original_compile(spec,m,o,device)
 assert o==2*m
 src=torch.tensor([v for a in range(m) for v in (a,m+a)],device=device)
 dst=torch.tensor([v for a in range(m) for v in (m+a,m+o+a)],device=device)
 relation=torch.tensor([11,7]*m,device=device);owner=torch.arange(m,device=device).repeat_interleave(2)
 return EdgeContextGraph(tuple('edge'+str(i) for i in range(2*m)),m,o,src,dst,relation,owner,torch.ones(2*m,device=device,dtype=torch.bool),relation==7,torch.zeros(2*m,2*m,device=device,dtype=torch.bool),torch.full_like(src,-1))
stage1_spec.compile_stage1_graph=compile_graph
import utils.task_role_spec as roles
_original_sizes=roles.task_size_probabilities
def sizes(shape,ranges):
 result=torch.zeros(*shape.shape[:-1],5,device=shape.device);result[...,3]=1;return result
roles.task_size_probabilities=sizes
from jointpush.partial.scene import PartialPushScene
from learning.multi_agent.shared_support_network import SharedSupportBuilder
_build_shared=SharedSupportBuilder.Network.__init__
def build_shared(self,*args,**kwargs):
 _build_shared(self,*args,**kwargs)
 for encoder in (self.actor_encoder,self.critic_encoder):
  encoder.edge_encoder.category_embed=torch.nn.Embedding(5,64)
  torch.nn.init.trunc_normal_(encoder.edge_encoder.category_embed.weight,std=.02)
SharedSupportBuilder.Network.__init__=build_shared
import utils.parse_task as parser
parser.PartialPushScene=PartialPushScene
# Validate and strictly copy Stage1 tensors, initializing only new coordination.
from learning.multi_agent import stage2_network_builder as stage2
from utils.stage1_push import check_contract as check_base
import yaml
base_env=yaml.safe_load((WORK/'configs/baseline.yaml').read_text())
def transfer(model,weights):
 check_base(weights,base_env)
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
 result=dict(adapter='raw_cuboid_partial_v1',source_manifest_sha256=hashlib.sha256((ROOT/'jointpush/jhh_source_manifest.json').read_bytes()).hexdigest(),partition=env['partitionPush'],reward=env['partialReward'],template_rsi=env['templateRsi'],graph=env['relationGraph'],agents=4,objects=8)
 if env['pushBox'].get('surface_mode'):result['box_surfaces']=env['pushBox']
 return result
shared.checkpoint_contract=contract
import utils.stage2_spec as stage2_spec
_original_check=stage2_spec.check_checkpoint
def check_partial(weights,env):
 if 'partitionPush' not in env:return _original_check(weights,env)
 if weights.get('shared_support_contract')!=contract(env):raise ValueError('Partial Joint Push Stage2 contract differs')
stage2_spec.check_checkpoint=check_partial
from learning.multi_agent.ma_agent import MAAgent
# Optional continuation directory contains a filtered copy of prior events.
if os.environ.get('PARTIAL_TB_LOGDIR'):
 from tensorboardX import SummaryWriter
 from rl_games.common import a2c_common
 _tb_path=Path(os.environ['PARTIAL_TB_LOGDIR']).resolve()
 _tb_offsets=json.loads((_tb_path/'continuation.json').read_text()).get('time_offset',0)
 class ContinuedSummaryWriter(SummaryWriter):
  def __init__(self,*args,**kwargs):
   super().__init__(str(_tb_path))
  def add_scalar(self,tag,scalar_value,global_step=None,*args,**kwargs):
   if tag.endswith('/time') and global_step is not None:global_step+=_tb_offsets
   return super().add_scalar(tag,scalar_value,global_step,*args,**kwargs)
 a2c_common.SummaryWriter=ContinuedSummaryWriter
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
_restore_full=MAAgent.set_full_state_weights
def restore_full(self,weights):
 result=_restore_full(self,weights)
 assert self.epoch_num==weights['epoch'] and self.frame==weights.get('frame',0)
 assert all(torch.equal(v,weights['model'][k].to(v.device)) for k,v in self.model.state_dict().items())
 def same(a,b):
  if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b.to(a.device))
  if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
  if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
  return a==b
 assert same(self.optimizer.state_dict(),weights['optimizer'])
 self._partial_start_epoch=self.epoch_num
 self._partial_actor_digest=digest(self.model.a2c_network.actor_encoder)
 self._partial_coord_digest=digest(self.model.a2c_network.coordination)
 self._partial_rms_digest=digest(self.running_mean_std)
 report=dict(epoch=self.epoch_num,frame=self.frame,model_restored=True,optimizer_restored=True,optimizer_states=len(self.optimizer.state),tensorboard_logdir=os.environ.get('PARTIAL_TB_LOGDIR',self.summaries_dir))
 Path(self.experiment_dir,'resume_verification.json').write_text(json.dumps(report,indent=2))
 print('PARTIAL_STAGE2_RESUME_VERIFIED',json.dumps(report),flush=True)
 return result
MAAgent.set_full_state_weights=restore_full
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
 if self.epoch_num==getattr(self,'_partial_start_epoch',0)+2:
  actor_unchanged=digest(self.model.a2c_network.actor_encoder)==self._partial_actor_digest
  rms_unchanged=digest(self.running_mean_std)==self._partial_rms_digest
  coord_changed=digest(self.model.a2c_network.coordination)!=self._partial_coord_digest
  assert actor_unchanged and rms_unchanged and coord_changed
  report=dict(epoch=self.epoch_num,actor_encoder_unchanged=actor_unchanged,normalizer_unchanged=rms_unchanged,coordination_updated=coord_changed,stage1_optimizer_restored=False,stage2_optimizer_resumed=hasattr(self,'_partial_start_epoch'))
  Path(self.experiment_dir,'startup_verification.json').write_text(json.dumps(report,indent=2))
  self.save(str(Path(self.experiment_dir,'nn','PartialRawCuboid_startup')))
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
  self._restored_weights=weights
  print('PARTIAL_STAGE2_PLAYER_RESTORED',flush=True)
 MAPlayerContinuous.restore=restore
 _run=MAPlayerContinuous.run
 def run(self):
  original=self.get_action;steps=0;limit=int(os.environ.get('PARTIAL_CHECK_STEPS','0'))
  def action(*args,**kwargs):
   nonlocal steps
   value=original(*args,**kwargs)
   if steps==0 and os.environ.get('PARTIAL_GRAD_CHECK')=='1':
    assert os.environ.get('PARTIAL_CPU')=='1'
    actor_hash=digest(self.model.a2c_network.actor_encoder);rms_hash=digest(self.running_mean_std)
    coord=self.model.a2c_network.coordination;state=copy.deepcopy(coord.state_dict());before=digest(coord)
    with torch.enable_grad():
     obs=self.running_mean_std(self.env.task.obs_buf)
     mu,_=self.model.a2c_network.eval_actor(obs)
     optimizer=torch.optim.Adam(coord.parameters(),lr=2e-5);optimizer.zero_grad();mu.square().mean().backward()
     norm=sum(float(p.grad.abs().sum()) for p in coord.parameters() if p.grad is not None)
     assert norm>0 and torch.isfinite(torch.tensor(norm))
     optimizer.step()
    assert digest(coord)!=before and digest(self.model.a2c_network.actor_encoder)==actor_hash and digest(self.running_mean_std)==rms_hash
    coord.load_state_dict(state,strict=True)
    task=self.env.task;out=Path(task.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
    verification=dict(agents=4,objects=8,observations=task.get_obs_size(),amp=task.get_num_amp_obs(),coordination_gradient_l1=norm,frozen_actor_unchanged=True,scene_normalizer_unchanged=True,coordination_update_verified=True,probe_weights_restored=True)
    (out/'gradient_verification.json').write_text(json.dumps(verification,indent=2))
    fixture=dict(model=self.model.state_dict(),running_mean_std=self.running_mean_std.state_dict(),amp_input_mean_std=self._amp_input_mean_std.state_dict(),shared_support_contract=contract(task.cfg['env']),validation_only=True,epoch=0)
    torch.save(fixture,out/'validation_transfer.pth')
    print('PARTIAL_CPU_GRADIENT_VERIFIED',json.dumps(verification),flush=True)
   if steps==0 and os.environ.get('PARTIAL_CPU')=='1':
    task=self.env.task;assignment=task._agent_box_assignment;rows=torch.arange(task.num_envs,device=task.device)[:,None]
    centers=task._box_states[rows,assignment,:2];root=task._humanoid_root_states[...,:2]
    assert (((root-centers)*(task._tar_pos[...,:2]-centers)).sum(-1)<=1e-4).all()
    regions=task._agent_region_assignment.clone()
    assert torch.equal(regions.sort(-1).values,torch.arange(4,device=task.device).expand(task.num_envs,4))
    assert torch.equal(assignment.gather(1,regions),assignment)
    task._compute_observations();assert torch.equal(task._agent_region_assignment,regions)
    print('PARTIAL_REAR180_RANDOM_ASSIGNMENT_VERIFIED',flush=True)
   assert torch.isfinite(value).all() and torch.isfinite(self.env.task.obs_buf).all()
   steps+=1
   if limit and steps>limit:
    print('PARTIAL_STAGE2_CHECK_PASSED',limit,'finite physics/action steps',flush=True);raise SystemExit(0)
   return value
  self.get_action=action
  return _run(self)
 MAPlayerContinuous.run=run
runpy.run_path(str(SOURCE/'tokenhsi/run.py'),run_name='__main__')
