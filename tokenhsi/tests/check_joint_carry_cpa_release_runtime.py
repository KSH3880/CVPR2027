import json
import runpy
import sys
from copy import deepcopy
from pathlib import Path

import isaacgym
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.multi_agent.ma_players import MAPlayerContinuous
from env.tasks.multi_agent.joint_carry_reward import JointCarryRuntime, cpa_collision_penalty
from utils.edge_ontop_spec import ON_TOP


@torch.no_grad()
def check(player):
    task = player.env.task
    config = deepcopy(task._relation_cfg)
    cpa = config['joint_carry']['collision']
    assert cpa.pop('release_buffer') == 1.
    obs = player.env_reset();player.get_batch_size(obs['obs'], 1)
    baseline = JointCarryRuntime(task.num_envs, task.relation_runtime.graph, config, task.device)
    counts = dict(released_scene_steps=0, nonzero_penalty_removed=0, retained_penalty_scene_steps=0,
                  at_removed=0, ontop_removed=0, independent_scene_steps=0)
    max_removed = 0.
    for step in range(64):
        obs, reward, _, info = player.env_step(player.env, player.get_action(obs, True))
        obs = player.obs_to_torch(obs)
        roots = task._humanoid_root_states
        objects = task._logical_box_values(task._box_states)
        sizes = task._logical_box_values(task._box_size)
        radius = sizes[:, 0, :2].norm(dim=-1)/2+1.
        release = task._joint_env_mask & ((roots[..., :2]-objects[:, 0, None, :2]).norm(dim=-1)
                                         <= radius[:, None]).all(-1)
        raw = -task._agent_collision_coeff*cpa_collision_penalty(
            roots[..., :3], roots[..., 7:10], task._agent_collision_dist, cpa['ttc_discount'], task.dt)
        expected = raw.masked_fill(release[:, None], 0.)
        terms = info['reward_terms'].reshape(task.num_envs, 2, -1)
        collision = terms[..., task.REWARD_TERM_NAMES.index('collision')]
        torch.testing.assert_close(collision, expected, rtol=0, atol=0)
        torch.testing.assert_close(collision[~task._joint_env_mask], raw[~task._joint_env_mask], rtol=0, atol=0)
        runtime = task.relation_runtime
        baseline.graph = runtime.graph
        baseline.coupled_holding.copy_(runtime.coupled_holding)
        result = runtime.last_result;diag = task._ontop_last_diag
        before = baseline.step(result['phi_raw'], result['progress_raw'], diag['z_error'],
                               diag['feet_height_error'], diag['region_error'])
        for key in ('state_component', 'progress_component', 'success_component', 'own_success', 'agent_task_reward'):
            torch.testing.assert_close(result[key], before[key], rtol=0, atol=0)
        torch.testing.assert_close(terms[..., :-1].sum(-1), terms[..., -1], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(reward.reshape(task.num_envs, 2), terms[..., -1])
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
        assert torch.isfinite(reward).all()
        nonzero = raw[:, 0] < -1e-6
        ontop = ((runtime.graph.edge_relation == ON_TOP) & runtime.graph.edge_valid).any(-1)
        counts['released_scene_steps'] += int(release.sum())
        counts['nonzero_penalty_removed'] += int((release & nonzero).sum())
        counts['retained_penalty_scene_steps'] += int((~release & nonzero).sum())
        counts['at_removed'] += int((release & nonzero & ~ontop).sum())
        counts['ontop_removed'] += int((release & nonzero & ontop).sum())
        counts['independent_scene_steps'] += int((~task._joint_env_mask).sum())
        max_removed = max(max_removed, float((collision-raw).max()))
    assert min(counts['at_removed'], counts['ontop_removed'], counts['retained_penalty_scene_steps']) > 0
    output = Path(task.cfg['args'].output_path);output.mkdir(parents=True, exist_ok=True)
    report = dict(environments=task.num_envs, steps=64, counts=counts, max_removed=max_removed,
                  radius=float(radius[task._joint_env_mask][0]),
                  original_reward_components_preserved=True, independent_cpa_preserved=True,
                  finite=True, checkpoint=task.cfg['args'].checkpoint)
    (output/'cpa_release_runtime_check.json').write_text(json.dumps(report, indent=2)+'\n')
    print('CPA_RELEASE_RUNTIME', json.dumps(report), flush=True)


MAPlayerContinuous.run_eval = check
runpy.run_path(str(ROOT/'run.py'), run_name='__main__')
