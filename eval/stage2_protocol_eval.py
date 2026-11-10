import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tokenhsi'))
import isaacgym
import torch
from learning.multi_agent import formal_eval
from utils.evaluation_protocol import predecessors
from utils.stage2_spec import evaluation_groups
from utils.task_role_spec import task_graph_packet


def capture_initial(player):
    task = player.env.task
    path = Path(task.cfg['args'].output_path)
    path.mkdir(parents=True, exist_ok=True)
    obs = player.env_reset()
    n, m = task.num_envs, task.num_agents
    packet = task_graph_packet(task.relation_runtime.graph, n).reshape(n, m, 5).long().clone()
    groups = evaluation_groups(packet, obs['obs'][:, -m:].long().clone())
    values = task._evaluation_box_values
    initial = dict(groups=groups.cpu(), packet=packet.cpu(),
        predecessor=predecessors(task.relation_runtime.graph, n, task.device).cpu(),
        sizes=values(task._box_size).cpu().clone(), goals=task._tar_pos.cpu().clone(),
        roots=task._humanoid_root_states.cpu().clone(), boxes=values(task._box_states).cpu().clone(),
        dof_pos=task._dof_pos.cpu().clone(), dof_vel=task._dof_vel.cpu().clone(),
        logical_box_order=task._logical_box_order.cpu().clone(),
        shared_center_goals=task._center_tar_pos.cpu().clone(),
        shared_centers=task._assigned_box_values(task._box_states)[..., :3].cpu().clone())
    torch.save(initial, path / 'initial.pt')
    torch.save(obs['obs'].cpu().clone(), path / 'observed_scene.pt')
    (path / 'reference_info.json').write_text(json.dumps(dict(seed=task.cfg['args'].seed,
        envs=n, agents=m, objects=task.num_objects, simulated_steps=0,
        purpose='Canonical initial snapshot only; no rollout or success score'), indent=2))
    return initial


if __name__ == '__main__':
    if os.environ.get('REFERENCE_ONLY') == '1':
        formal_eval.run_formal_eval = capture_initial
    else:
        import stage2_distractor_physics

    runpy.run_path(str(ROOT / 'output_etc/check/stage2_protocol_compare_1500_env64_20261108/new_driver.py'),
                   run_name='__main__')
