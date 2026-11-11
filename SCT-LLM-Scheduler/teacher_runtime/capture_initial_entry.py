"""Capture a fresh reset state; no policy rollout, FSM, or LLM."""
import os
from pathlib import Path
import sys
import json
import runpy

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True

def read_only_ksh(event, args):
    paths = []
    if event == 'open':
        path, mode, flags = args
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
            paths = [path]
    elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.truncate'):
        paths = [args[0]]
    elif event in ('os.rename', 'os.link', 'os.symlink'):
        paths = args[:2]
    for path in paths:
        if isinstance(path, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(path)).resolve()
            if str(resolved).startswith('/home/hwanhee/ksh/'):
                raise PermissionError('Recording process forbids writes under ksh')

sys.addaudithook(read_only_ksh)
sys.path[:0] = [str(ROOT/'tokenhsi'), str(ROOT/'eval')]
sys.path.insert(0, str(ROOT.parent))
from sct_scheduler.stage1 import policy_manifest
import isaacgym
import torch
from learning.multi_agent import formal_eval
from utils.task_role_spec import task_graph_packet
from utils.evaluation_protocol import predecessors
import stage2_protocol_sampling
import stage2_scalability_layout
stage2_scalability_layout.install()

@torch.no_grad()
def capture_initial(player):
    task = player.env.task
    destination = Path(task.cfg['args'].output_path)
    player.env_reset()
    stage2_protocol_sampling.capture(task)
    n, m = task.num_envs, task.num_agents
    values = dict(
        packet=task_graph_packet(task.relation_runtime.graph,n).reshape(n,m,5).long(),
        predecessor=predecessors(task.relation_runtime.graph,n,task.device),
        roots=task._humanoid_root_states, physical_boxes=task._box_states,
        physical_sizes=task._box_size, logical_boxes=task._evaluation_box_values(task._box_states),
        logical_sizes=task._evaluation_box_values(task._box_size),
        logical_box_order=task._logical_box_order, assignment=task._agent_box_assignment,
        goals=task._tar_pos, center_goals=task._center_tar_pos,
        groups=task._groups, templates=task._templates,
        hands=task._rigid_body_pos[:,:,task._key_body_ids[:2]])
    torch.save({k:v.detach().cpu().clone() for k,v in values.items()},destination/'initial_scene.pt')
    torch.cuda.synchronize(task.device)
    (destination/'capture.json').write_text(json.dumps(dict(
        capture='immediately after env_reset; no policy inference or env_step',
        environments=n,agents=m,seed=task.cfg['args'].seed,control_dt_s=task.dt,
        scene_carrier_checkpoint=task.cfg['args'].checkpoint,
        policy_controller_id=policy_manifest()['controller_id'],
        policy_checkpoint=policy_manifest()['checkpoint_local'],completed=True),indent=2))
    print('[INITIAL_SCENE_CAPTURED]', n, m, flush=True)
    return None

formal_eval.run_formal_eval = capture_initial
runpy.run_path(str(ROOT/'tokenhsi/run.py'),run_name='__main__')
