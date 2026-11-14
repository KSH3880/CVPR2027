from pathlib import Path
import json
import runpy

import isaacgym
import torch
from utils.before_chain_viewer import install, goal_aligned_scene, human_relations

install()
from learning.multi_agent import viewer_diagnostics
from utils.task_role_spec import task_graph_packet

original_run = viewer_diagnostics.run_viewer_diagnostics

def checked_viewer(player):
    task = player.env.task
    if task.num_agents != 3 or task.num_objects != 6:
        raise ValueError('The chain viewer requires 3 humans and 6 object tokens')
    original_reset, original_step = player.env_reset, player.env_step
    record = dict(task='A: AT -> B: ON_TOP -> C: ON_TOP', steps=0, resets=0,
                  checkpoint=task.cfg['args'].checkpoint)
    def reset(ids=None):
        obs = original_reset(ids)
        if ids is None or len(ids):
            graph = task.relation_runtime.graph
            packet = task_graph_packet(graph, task.num_envs).reshape(task.num_envs, 3, 5)
            scene = task._compute_clean_scene_obs()[:, :-3]
            virtual = goal_aligned_scene(scene, 3, 6)
            poses = virtual[:, -(15+12*7):-15].reshape(task.num_envs, 12, 7)
            real = scene[:, -(15+12*7):-15].reshape(task.num_envs, 12, 7)
            expected = torch.tensor([[1,0,0],[2,1,0],[0,2,1]],device=task.device)
            if not (human_relations(packet, 3)==expected).all():
                raise ValueError('Human dependency chain is incorrect')
            target = task._tar_pos[:, :2]-task._env_origins[:, None]
            if not torch.allclose(poses[:, 6:8, :3], target, atol=1e-5, rtol=0):
                raise ValueError('Virtual support XYZ is incorrect')
            if not torch.equal(poses[:, 6:8, 3:7], real[:, 3:5, 3:7]):
                raise ValueError('Virtual support quaternion changed')
            if not torch.equal(poses[:, :6], real[:, :6]):
                raise ValueError('Original entity poses changed')
            record.update(resets=record['resets']+1, packet=packet[0].tolist(),
                virtual_support_xyz=poses[0, 6:8, :3].tolist(),
                physical_sizes=task._box_size[0, :3].tolist(),
                human_relations=expected.tolist())
            print('[chain viewer] verified A AT -> B ON_TOP -> C ON_TOP', flush=True)
        return obs
    def step(env, action):
        if not torch.isfinite(action).all():
            raise ValueError('Nonfinite chain action')
        result = original_step(env, action)
        record['steps'] += 1
        return result
    player.env_reset, player.env_step = reset, step
    try:
        result = original_run(player)
    finally:
        player.env_reset, player.env_step = original_reset, original_step
        output=Path(task.cfg['args'].output_path)
        output.mkdir(parents=True, exist_ok=True)
        (output/'chain_viewer.json').write_text(json.dumps(record, indent=2))
    return result

viewer_diagnostics.run_viewer_diagnostics = checked_viewer
runpy.run_path(str(Path(__file__).with_name('run.py')), run_name='__main__')
