"""Read-only external five-task viewer, with output redirected locally."""
import sys,runpy,os
from pathlib import Path
sys.dont_write_bytecode=True
SOURCE=Path('/home/hwanhee/koo_cvpr/stage1_fix/ports/ksh_stage1_stage2_base/push')
WORK=Path.cwd()
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(SOURCE/'tokenhsi'))
import isaacgym
import torch
gpu=int(os.environ["TOKENHSI_GUI_GPU_INDEX"])
torch.cuda.set_device(gpu)
from learning.multi_agent.ma_players import MAPlayerContinuous
player_init=MAPlayerContinuous.__init__
def same_gpu(self,config):
 return player_init(self,dict(config,device_name=f"cuda:{gpu}"))
MAPlayerContinuous.__init__=same_gpu
import utils.output_paths as paths
original=paths.resolve_output_path
def local(experiment,mode,requested='',check=False,max_iterations=0,root=WORK):
 return original(experiment,mode,requested,check,max_iterations,root=WORK)
paths.resolve_output_path=local
runpy.run_path(str(SOURCE/'tokenhsi/run.py'),run_name='__main__')
