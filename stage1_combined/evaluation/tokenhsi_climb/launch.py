"""Evaluate the finetuned CLIMB policy with original TokenHSI point metrics/player loop."""
from pathlib import Path
import sys,importlib.util,ast,textwrap,datetime,json,os
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tokenhsi'),str(ROOT)]
import isaacgym
import torch
import numpy as np
from env.tasks.multi_agent.humanoid_tokenhsi_climb_ft import HumanoidTokenHSIClimbFT
from learning.multi_agent.chair_ft_agent import ChairFTPlayer

if os.environ.get('TOKENHSI_CLIMB_30D')=='1':
    from env.tasks.multi_agent.humanoid_tokenhsi_climb_30d import HumanoidTokenHSIClimb30FT as HumanoidTokenHSIClimbFT
    from learning.multi_agent.ma_players import MAPlayerContinuous as ChairFTPlayer

# Evaluation-only patches: preserve training code and weights.
reset_original=HumanoidTokenHSIClimbFT._reset_envs
def reset(self,ids):
    result=reset_original(self,ids)
    self._precision_buf[ids]=float('inf')
    return result

def metrics(self):
    obj=self._box_states[:,0]
    target=obj[:,:3].clone()
    target[:,2]+=self._climb_metadata_heights[:,0]+self._char_h
    error=(target-self._humanoid_root_states[:,0,:3]).norm(dim=-1)
    success=error<=self._success_threshold
    self._success_buf[success]+=1
    self._precision_buf[success]=torch.minimum(error[success],self._precision_buf[success])

HumanoidTokenHSIClimbFT._reset_envs=reset
HumanoidTokenHSIClimbFT._compute_metrics_evaluation=metrics

# Copy the original evaluation loop verbatim except artifact destination and trace recording.
source=ROOT.parent/'TokenHSI/tokenhsi/learning/transformer/trans_players.py'
tree=ast.parse(source.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run_eval')
code=textwrap.dedent(''.join(source.read_text().splitlines(keepends=True)[fn.lineno-1:fn.end_lineno]))
a=code.index('    folder_name =');b=code.index('    os.makedirs(save_dir',a)
code=code[:a]+"    save_dir = self.env.task.cfg['args'].output_path\n"+code[b:]
code=code.replace('        has_collected = torch.zeros', '        trace=[]\n        has_collected = torch.zeros')
needle='            normal_all_done_indices = done.nonzero(as_tuple=False)'
insert='''            task=self.env.task
            obj=task._box_states[:,0]
            target=obj[:,:3].clone();target[:,2]+=task._climb_metadata_heights[:,0]+task._char_h
            trace.append((task._humanoid_root_states[:,0,:3].cpu().numpy().copy(),target.cpu().numpy().copy(),(has_collected==0).cpu().numpy().copy(),done.cpu().numpy().copy()))
'''
code=code.replace(needle,insert+needle)
code=code.replace('        print(curr_exp_name)', '''        directory=Path(self.env.task.cfg['args'].output_path)
        directory.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(directory/f'trace_{i}.npz',root=np.stack([r[0] for r in trace]),target=np.stack([r[1] for r in trace]),alive=np.stack([r[2] for r in trace]),done=np.stack([r[3] for r in trace]))
        print(curr_exp_name)''')
namespace=dict(torch=torch,np=np,datetime=datetime,json=json,os=os,Path=Path)
exec(compile(code,str(source)+'[eval-adapter]','exec'),namespace)
ChairFTPlayer.run_eval=torch.no_grad()(namespace['run_eval'])
spec=importlib.util.spec_from_file_location('evaluation_run',ROOT/'tokenhsi/run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
if os.environ.get('TOKENHSI_CLIMB_30D')=='1':
    import utils.parse_task as parser
    parser.HumanoidTokenHSIClimb30FT=HumanoidTokenHSIClimbFT
    original_builder=run.build_alg_runner
    def build30(observer):
        runner=original_builder(observer)
        runner.player_factory.register_builder('climb30_ft',lambda **kw:ChairFTPlayer(**kw))
        return runner
    run.build_alg_runner=build30
run.main()
