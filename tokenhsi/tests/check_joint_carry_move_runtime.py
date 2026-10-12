import json
import runpy
import sys
from pathlib import Path

import isaacgym
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.multi_agent.ma_players import MAPlayerContinuous
from env.tasks.multi_agent.edge_context_reward import owner_sum


@torch.no_grad()
def check(player):
    task = player.env.task
    assert task._relation_cfg['joint_carry']['alignment']['direction'] == 'min_root_velocity_xy'
    obs = player.env_reset(); player.get_batch_size(obs['obs'], 1)
    count = 0
    for _ in range(32):
        obs, _, done, info = player.env_step(player.env, player.get_action(obs, True))
        obs = player.obs_to_torch(obs)
        diag = info['joint_alignment']; reward = diag['reward']
        terms = info['reward_terms'].reshape(task.num_envs, task.num_agents, -1)
        index = task.REWARD_TERM_NAMES.index('joint_alignment')
        expected = reward[:, None].expand(-1, task.num_agents)
        torch.testing.assert_close(terms[..., index], expected)
        torch.testing.assert_close(terms[..., :-1].sum(-1), terms[..., -1])
        result = task.relation_runtime.last_result
        torch.testing.assert_close(result['agent_task_reward'],
            owner_sum(result['total'], task.relation_runtime.graph) + expected)
        assert not reward[~task._joint_env_mask].any()
        assert not reward[~task.relation_runtime.coupled_holding].any()
        assert (reward >= 0).all() and (reward <= .200001).all()
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
        assert torch.isfinite(task.rew_buf).all()
        count += int((reward > 0).sum())
    assert count > 0
    graph = task.relation_runtime.graph
    ontop = ((graph.edge_relation == 8) & graph.edge_valid).any(-1)
    report = dict(steps=32, joint_at=int((task._joint_env_mask & ~ontop).sum()),
        joint_ontop=int((task._joint_env_mask & ontop).sum()),
        independent=int((~task._joint_env_mask).sum()), nonzero_scene_steps=count,
        both_agents_receive_identical_alignment=True, exactly_once_in_task_total=True,
        no_independent_or_gate_off_alignment=True, component_sum_equals_total=True,
        observation_amp_reward_finite=True, reward_terms=list(task.REWARD_TERM_NAMES))
    out = Path(task.cfg['args'].output_path); out.mkdir(parents=True, exist_ok=True)
    (out/'runtime_check.json').write_text(json.dumps(report, indent=2))
    print('MOVE_RUNTIME', json.dumps(report), flush=True)
    return report


MAPlayerContinuous.run_eval = check
runpy.run_path(str(ROOT/'run.py'), run_name='__main__')
