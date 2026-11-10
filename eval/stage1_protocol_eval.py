import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tokenhsi'))
import isaacgym
import torch
from learning.multi_agent import formal_eval
from stage1_protocol_policy import Stage1Policy

original_eval = formal_eval.run_formal_eval


def evaluate(player):
    original_action = player.get_action
    policy = None
    calls = 0

    def action(obs, is_determenistic=False):
        nonlocal policy, calls
        if policy is None:
            policy = Stage1Policy(player, os.environ['STAGE1_POLICY_CHECKPOINT'], obs['obs'])
        calls += 1
        return original_action(obs, is_determenistic)

    player.get_action = action
    result = original_eval(player)
    player.get_action = original_action
    metadata = policy.finish()
    metadata['action_calls'] = calls
    metadata['scene_carrier_checkpoint'] = player.env.task.cfg['args'].checkpoint
    output = Path(player.env.task.cfg['args'].output_path)
    (output/'stage1_policy.json').write_text(json.dumps(metadata, indent=2))
    result.update(policy='stage1_26k', experiment='stage1_push_26k_protocol', stage1_policy=metadata)
    (output/'result.json').write_text(json.dumps(result, indent=2))
    return result


formal_eval.run_formal_eval = torch.no_grad()(evaluate)
runpy.run_path(str(ROOT/'eval/stage2_scalability_eval.py'), run_name='__main__')
