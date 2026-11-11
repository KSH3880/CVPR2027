"""Local-only dense trajectory collection; no FSM, no LLM, no Stage1 replacement."""
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
import isaacgym
import numpy as np
import torch
from learning.multi_agent import formal_eval
from utils.task_role_spec import task_graph_packet
from utils.evaluation_protocol import spatial_success, shared_center_success, predecessors
import stage2_scalability_layout
import stage2_xy100
import stage2_full_observation
import table2_sampling
from table2_metrics import surface_distance

stage2_scalability_layout.install()
stage2_xy100.install()
SETTING = os.environ['SCT_RECORD_SETTING']
table2_sampling.install(SETTING)
base_eval = formal_eval.run_formal_eval


@torch.no_grad()
def evaluate(player):
    task = player.env.task
    destination = Path(task.cfg['args'].output_path)
    reset, step = player.env_reset, player.env_step
    dense = {k: [] for k in ('roots', 'physical_boxes', 'logical_boxes', 'hands', 'surface_distances',
                              'task_success', 'group_success', 'alive')}
    packet = alive = None

    def capture():
        roots = task._humanoid_root_states
        logical = task._evaluation_box_values(task._box_states)
        sizes = task._evaluation_box_values(task._box_size)
        assigned = task._assigned_box_values(task._box_states)
        assigned_sizes = task._assigned_box_values(task._box_size)
        hands = task._rigid_body_pos[:, :, task._key_body_ids[:2]]
        n, m = task.num_envs, task.num_agents
        distances = surface_distance(hands.reshape(n*m, 1, 2, 3).double(),
                                     assigned.reshape(n*m, 13)[:, :7].double(),
                                     assigned_sizes.reshape(n*m, 3).double()).reshape(n, m, 2)
        current = spatial_success(packet, roots[..., :3], logical[..., :7], sizes, task._tar_pos)[..., 1] & alive
        if SETTING.startswith('enable_'):
            success = current.all(-1)
        else:
            success = shared_center_success(assigned[..., :3], task._center_tar_pos, task._groups, alive).any(-1)
        for key, value in dict(roots=roots, physical_boxes=task._box_states, logical_boxes=logical,
                               hands=hands, surface_distances=distances, task_success=current,
                               group_success=success, alive=alive).items():
            dense[key].append(value.cpu().numpy().copy())

    def reset_record(*args, **kwargs):
        nonlocal packet, alive
        obs = reset(*args, **kwargs)
        packet = task_graph_packet(task.relation_runtime.graph, task.num_envs).reshape(task.num_envs, task.num_agents, 5).long()
        alive = packet[..., 0].bool().clone()
        initial = {key: value.cpu().clone() for key, value in dict(
            packet=packet, predecessor=predecessors(task.relation_runtime.graph,task.num_envs,task.device),
            physical_sizes=task._box_size, logical_sizes=task._evaluation_box_values(task._box_size),
            logical_box_order=task._logical_box_order, assignment=task._agent_box_assignment,
            goals=task._tar_pos, center_goals=task._center_tar_pos,
            groups=task._groups, templates=task._templates).items()}
        torch.save(initial, destination/'record_initial.pt')
        capture()
        return obs

    def step_record(*args, **kwargs):
        nonlocal alive
        value = step(*args, **kwargs)
        capture()
        alive &= ~task._terminate_buf.reshape(task.num_envs, task.num_agents).bool()
        return value

    player.env_reset, player.env_step = reset_record, step_record
    try:
        result = base_eval(player)
        torch.cuda.synchronize(task.device)
        np.savez_compressed(destination/'dense_trajectories.npz', **{k: np.stack(v) for k, v in dense.items()})
        (destination/'recording.json').write_text(json.dumps(dict(
            setting=SETTING, checkpoint=task.cfg['args'].checkpoint, control_dt_s=task.dt,
            frames=len(dense['roots']), initial_sample_included=True, initial_release_times_s=0,
            external_freeze=False, stage1_replacement=False, distractors='physically retained',
            success_rule='existing formal evaluator XY100; enabling simultaneous',
            completed=True), indent=2))
        print('[RECORDING_COMPLETE]', SETTING, 'successes', int(np.stack(dense['group_success']).any(0).sum()), flush=True)
        return result
    finally:
        player.env_reset, player.env_step = reset, step
        torch.cuda.synchronize(task.device)


formal_eval.run_formal_eval = evaluate
runpy.run_path(str(ROOT/'tokenhsi/run.py'), run_name='__main__')
