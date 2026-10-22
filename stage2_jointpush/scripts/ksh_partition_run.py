"""Read-only KSH Stage1 -> current partial scene, identity Stage2, evaluation only."""
import os,sys,runpy,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
SOURCE=Path(os.environ.get('KSH_PUSH_SOURCE','/home/hwanhee/koo_cvpr/stage1_fix/ports/ksh_stage1_stage2_base/push'))
WORK=Path.cwd()
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(SOURCE/'tokenhsi'))
import isaacgym
import torch
# Output and transient builds always belong to the local workspace.
import utils.output_paths as output_paths
resolve_original=output_paths.resolve_output_path
output_paths.resolve_output_path=lambda experiment,mode,requested='',check=False,max_iterations=0,root=WORK:resolve_original(experiment,mode,requested,check,max_iterations,root=WORK)
# The base source Stage1 validator describes a different physical scene. Limit
# the exception to the named partial adapter and retain all original callers.
import utils.unified_training as unified
original_validation=unified.validate_unified_env
def validate(env):
 if 'partitionPush' not in env:return original_validation(env)
 assert (env['numAgents'],env['numObjects'])==(4,8)
 assert env['partitionPush']['size_min'][1]==env['partitionPush']['size_max'][1]==1.8
 assert env['partitionPush']['rsi']=='loco_only'
unified.validate_unified_env=validate
import utils.five_task as five_task
original_probabilities=five_task.size_probabilities
def probabilities(sizes,env):
 if 'partitionPush' not in env:return original_probabilities(sizes,env)
 result=torch.zeros(*sizes.shape[:-1],6,device=sizes.device);result[...,5]=1
 return result
five_task.size_probabilities=probabilities
from ksh_partition_scene import HumanoidMAKshPartitionPush
import utils.parse_task as parser
parser.HumanoidMAKshPartitionPush=HumanoidMAKshPartitionPush
from env.tasks.base_task import BaseTask
create_original=BaseTask.create_sim
def create(self,compute,graphics,engine,params):
 if graphics>=0:graphics=int(os.environ.get('TOKENHSI_GRAPHICS_DEVICE_ID','0'))
 return create_original(self,compute,graphics,engine,params)
BaseTask.create_sim=create
from learning.multi_agent.ma_players import MAPlayerContinuous
from learning.multi_agent.stage2_network_builder import transfer_stage1
player_init=MAPlayerContinuous.__init__
def init(self,config):return player_init(self,dict(config,device_name='cuda:0' if '--headless' not in sys.argv else config.get('device_name','cuda:0')))
MAPlayerContinuous.__init__=init

def digest(module):
 h=hashlib.sha256()
 for key,value in sorted(module.state_dict().items()):h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def restore(self,path):
 source=torch.load(path,map_location=self.device,weights_only=False)
 report=transfer_stage1(self.model,source)
 self.running_mean_std.load_state_dict(source['running_mean_std'],strict=True)
 self.running_mean_std.eval();self.model.eval()
 if hasattr(self,'_amp_input_mean_std'):
  self._amp_input_mean_std.load_state_dict(source['amp_input_mean_std'],strict=True)
 self._source_digest=digest(self.model)
 task=self.env.task;out=Path(task.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
 report.update(checkpoint=path,optimizer_steps=0,trained=False,source_unchanged=True)
 (out/'transfer_report.json').write_text(json.dumps(report,indent=2))
 print('KSH_PARTITION_TRANSFER',len(report['copied']),'copied',len(report['added']),'identity-initialized',flush=True)
MAPlayerContinuous.restore=restore
original_run=MAPlayerContinuous.run
def run(self):
 self._reward_debug=False
 action_original=self.get_action;steps=0
 limit=int(os.environ.get('VIEW_CHECK_STEPS','0'))
 def action(*args,**kwargs):
  nonlocal steps
  value=action_original(*args,**kwargs)
  task=self.env.task
  assert torch.isfinite(value).all() and torch.isfinite(task.obs_buf).all()
  assert task.num_agents==4 and task.num_objects==8
  assert torch.allclose(task._box_size[...,1],torch.full_like(task._box_size[...,1],1.8))
  if steps==0:
   out=Path(task.cfg['args'].output_path)
   report=dict(checkpoint=task.cfg['args'].checkpoint,agents=4,object_tokens=8,bank_actors=24,box_y_min=float(task._box_size[...,1].min()),box_y_max=float(task._box_size[...,1].max()),group_counts=task._group_counts.cpu().tolist(),physical_assignments=task._agent_box_assignment.cpu().tolist(),region_assignments=task._agent_region_assignment.cpu().tolist(),obs_width=task.obs_buf.shape[-1],physics_device=str(task.device),actor_device=str(next(self.model.parameters()).device),optimizer_steps=0,identity_stage2=True,goal_distance=[2.,6.])
   (out/'viewer_verification.json').write_text(json.dumps(report,indent=2))
   print('KSH_PARTITION_VIEW_VERIFIED',json.dumps(report),flush=True)
  steps+=1
  if limit and steps>limit:
   assert digest(self.model)==self._source_digest
   print('KSH_PARTITION_CHECK_FINISHED',steps-1,'steps; weights unchanged',flush=True)
   raise SystemExit(0)
  return value
 self.get_action=action
 return original_run(self)
MAPlayerContinuous.run=run
if '--test' not in sys.argv or '--eval' not in sys.argv:raise ValueError('This entry point is viewer/evaluation only, never training')
runpy.run_path(str(SOURCE/'tokenhsi/run.py'),run_name='__main__')
