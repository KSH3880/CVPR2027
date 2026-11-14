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

capture_initial = stage2_protocol_eval.capture_initial


def capture(player):
    initial = capture_initial(player)
    task = player.env.task
    groups, roots = initial['groups'].clone(), initial['roots'][..., :2]
    rows, followers = (initial['predecessor'] >= 0).nonzero(as_tuple=True)
    groups[rows, followers] = groups[rows, initial['predecessor'][rows, followers]]
    same = groups[:, :, None] == groups[:, None, :]
    centers = torch.bmm(same.float(), roots)/same.sum(-1, keepdim=True)
    origins = task._env_origins.detach().cpu()[:, None, :2]
    radius = float(os.environ['GROUP_SPAWN_RADIUS'])
    error = float(((centers-origins).norm(dim=-1)-radius).abs().max())
    gap = (centers[:, :, None]-centers[:, None]).norm(dim=-1).masked_fill(same, float('inf'))
    proof = dict(group_center_radius_m=radius, maximum_radius_error_m=error,
        minimum_group_center_gap_m=float(gap.min()), group_center_definition='I singleton; E pair; C group',
        envs=task.num_envs, agents=task.num_agents, simulated_steps=0, radius_verified=error < .002)
    path = Path(task.cfg['args'].output_path)
    (path/'radius_verification.json').write_text(json.dumps(proof, indent=2)+'\n')
    if not proof['radius_verified']:
        raise ValueError('Initial group centers differ from configured radius')
    return initial


stage2_protocol_eval.capture_initial = capture
runpy.run_path(str(ROOT/'eval/group_spacing_sampling_probe.py'), run_name='__main__')
