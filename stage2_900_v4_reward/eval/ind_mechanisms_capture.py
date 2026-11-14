import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tokenhsi'), str(ROOT/'eval')]
import isaacgym
import torch
import stage2_protocol_eval

original = stage2_protocol_eval.capture_initial


def capture(player):
    initial = original(player)
    task = player.env.task
    out = Path(task.cfg['args'].output_path)
    source = Path(os.environ['IND_ORIGINAL'])
    expected = torch.load(source/'initial.pt', map_location='cpu', weights_only=False)
    for key, value in expected.items():
        if not torch.equal(value, initial[key]):
            raise ValueError('Original snapshot mismatch: '+key)
    obs = torch.load(out/'observed_scene.pt', weights_only=False)
    if not torch.equal(obs, torch.load(source/'observed_scene.pt', weights_only=False)):
        raise ValueError('Original first observation mismatch')
    initial.update(kinematic=task._kinematic_humanoid_rigid_body_states.cpu().clone(),
        origins=task._env_origins.cpu().clone(), physical_sizes=task._box_size.cpu().clone(),
        density=task._box_density.cpu().reshape(task.num_envs, -1).clone(),
        observation=obs)
    torch.save(initial, out/'extended.pt')
    (out/'capture_verification.json').write_text(json.dumps(dict(
        original=str(source), initial_exact=True, observation_exact=True), indent=2))
    return initial


stage2_protocol_eval.capture_initial = capture
runpy.run_path(str(ROOT/'eval/stage2_scalability_eval.py'), run_name='__main__')
