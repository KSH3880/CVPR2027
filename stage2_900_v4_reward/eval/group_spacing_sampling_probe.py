import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import textwrap
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tokenhsi'),str(ROOT/'eval')]
import isaacgym
import torch
from env.tasks.multi_agent.humanoid_ma_coupled_push import HumanoidMACoupledPush

OUTPUT=Path(sys.argv[sys.argv.index('--output_path')+1])
OUTPUT.mkdir(parents=True,exist_ok=True)
LIMIT=int(os.environ.get('SPACING_PROBE_ATTEMPTS','4'))
ROWS=[]
TIMING={}


def stage(name):
    torch.cuda.synchronize()
    TIMING[name]=time.monotonic()


def record(task,attempt,ids,rejected,push_bad,scene_bad):
    torch.cuda.synchronize()
    now=time.monotonic()
    roots=task._humanoid_root_states[ids,:,:2]
    gap=(roots[:,:,None]-roots[:,None]).norm(dim=-1)
    eye=torch.eye(task.num_agents,dtype=torch.bool,device=task.device)
    close=(((gap<.7)&~eye).any((1,2)))&(task._scene_domains[ids]!=1)
    row=dict(attempt=attempt+1,input_scenes=len(ids),accepted=int((~rejected).sum()),
        remaining=int(rejected.sum()),push_rejected=int(push_bad.sum()),scene_rejected=int(scene_bad.sum()),
        human_spacing_below_07=int(close.sum()),push_seconds=TIMING['distractors']-TIMING['push'],
        distractor_seconds=TIMING['validation']-TIMING['distractors'],
        validation_seconds=now-TIMING['validation'],pending_scene_ids=ids[rejected].cpu().tolist())
    ROWS.append(row)
    (OUTPUT/'sampling_probe.json').write_text(json.dumps(dict(diagnostic_only=True,max_attempts=LIMIT,
        spacing=os.environ.get('SPACING_METERS'),group_radius=os.environ.get('GROUP_SPAWN_RADIUS'),rows=ROWS),indent=2)+'\n')
    print('SAMPLING_PROBE',json.dumps(row),flush=True)
    if attempt+1==LIMIT:
        torch.save(dict(ids=ids.cpu(),rejected=rejected.cpu(),push_rejected=push_bad.cpu(),
            scene_rejected=scene_bad.cpu(),roots=task._humanoid_root_states[ids].cpu(),
            groups=task._groups[ids].cpu(),templates=task._templates[ids].cpu(),
            boxes=task._box_states[ids].cpu(),sizes=task._box_size[ids].cpu()),OUTPUT/'rejected_layouts.pt')


original=HumanoidMACoupledPush._reset_ontop_context_envs
source=textwrap.dedent(inspect.getsource(original))
replacements={
    "range(self.cfg['env']['stage2JointPush']['layout_attempts'])":"range(min(self.cfg['env']['stage2JointPush']['layout_attempts'],_probe_limit))",
    '        rejected = scene.sample_push(self,pending)':'        _probe_stage("push")\n        rejected = scene.sample_push(self,pending)\n        push_bad = rejected.clone()\n        _probe_stage("distractors")',
    '        rejected |= self._ontop_scene_infeasible(pending)':'        _probe_stage("validation")\n        scene_bad = self._ontop_scene_infeasible(pending)\n        rejected |= scene_bad\n        _probe_record(self,attempt,pending,rejected,push_bad,scene_bad)',
}
for before,after in replacements.items():
    if source.count(before)!=1:
        raise ValueError('Sampling probe source changed: '+before)
    source=source.replace(before,after)
namespace=dict(original.__globals__,_probe_limit=LIMIT,_probe_stage=stage,_probe_record=record)
exec(compile(source,__file__,'exec'),namespace)
HumanoidMACoupledPush._reset_ontop_context_envs=namespace['_reset_ontop_context_envs']
runpy.run_path(str(ROOT/'eval/group_spacing_eval.py'),run_name='__main__')
