import json
import runpy
import sys
from pathlib import Path

import isaacgym
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.stage1_grasp_shift import initial_grasp_sides, shared_payload_views

context = runpy.run_path(str(ROOT/'tests/check_joint_carry_goal360_runtime.py'))
task = context['task']
args = context['context']['args']
joint = task._joint_env_mask.nonzero().flatten()
solo = (~task._joint_env_mask).nonzero().flatten()
weights = task._template_rsi_weights.clone()
preset = task._task_graph_preset
report = {'observation_size': task.get_obs_size(), 'resets': []}
assert task.get_obs_size() == 1268

try:
    for graph in ('joint_carry_at', 'joint_carry_ontop'):
        task._task_graph_preset = graph
        for skill in ('loco', 'pickUp', 'carryWith', 'putDown'):
            if graph == 'joint_carry_ontop' and skill == 'putDown':
                continue
            task._template_rsi_weights.zero_()
            task._template_rsi_weights[:, task._skill.index(skill)] = 1.
            untouched = task.obs_buf[solo].clone()
            old_sides = task._grasp_sides[solo].clone()
            task._reset_envs(joint)
            torch.testing.assert_close(task.obs_buf[solo], untouched, atol=0, rtol=0)
            torch.testing.assert_close(task._grasp_sides[solo], old_sides, atol=0, rtol=0)
            raw = task._compute_clean_scene_obs(joint, grasp_view=False)
            torch.testing.assert_close(task._grasp_sides[joint], initial_grasp_sides(raw))
            expected = shared_payload_views(raw, task._grasp_sides[joint], 1., .15)
            torch.testing.assert_close(task.obs_buf[joint].reshape(-1, 634), expected, atol=2e-6, rtol=1e-5)
            raw_solo = task._compute_clean_scene_obs(grasp_view=False)[solo]
            torch.testing.assert_close(task.obs_buf[solo].reshape(-1,2,634),
                raw_solo[:,None].expand(-1,2,-1), atol=0, rtol=0)
            report['resets'].append({'task': graph, 'skill': skill,
                'joint': len(joint), 'solo': len(solo),
                'platform_active': int(task._joint_loco_platform.active[joint].sum())})

    ids = joint[::2]
    untouched = task.obs_buf[joint[1::2]].clone()
    old_sides = task._grasp_sides[joint[1::2]].clone()
    task._reset_envs(ids)
    torch.testing.assert_close(task.obs_buf[joint[1::2]], untouched, atol=0, rtol=0)
    torch.testing.assert_close(task._grasp_sides[joint[1::2]], old_sides, atol=0, rtol=0)
    old_sides = task._grasp_sides.clone()
    for _ in range(8):
        task.step(torch.zeros(task.num_envs * 2, task.num_actions, device=task.device))
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
        torch.testing.assert_close(task._grasp_sides, old_sides, atol=0, rtol=0)
        raw = task._compute_clean_scene_obs(grasp_view=False)
        expected = shared_payload_views(raw[joint], task._grasp_sides[joint], 1., .15)
        torch.testing.assert_close(task.obs_buf[joint].reshape(-1,634), expected, atol=2e-6, rtol=1e-5)
    report['partial_reset_and_persistent_assignment'] = True
    report['finite_physics_steps'] = 8
finally:
    task._template_rsi_weights.copy_(weights)
    task._task_graph_preset = preset

out = Path(args.output_path)
(out/'grasp_view_runtime.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
