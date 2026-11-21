import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tokenhsi'), str(ROOT/'eval')]
import isaacgym
import torch
from learning.multi_agent import formal_eval

original_eval = formal_eval.run_formal_eval


@torch.no_grad()
def checked_eval(player):
    task = player.env.task
    path = Path(task.cfg['args'].output_path)
    network = player.model.a2c_network
    if task.cfg['env']['relationGraph'].get('human_relation_bias') is not False:
        raise ValueError('Expected a trained no-bias student')
    if network.actor_encoder.edge_encoder.bias_projection is None:
        raise ValueError('SCT task bias must remain enabled')
    calls, hooks = {}, []
    for name, branch in (('actor', network.coordination), ('critic', network.critic_coordination)):
        if branch.relation_fusion is None:
            raise ValueError('Dependency feature fusion must remain enabled')
        for index, block in enumerate(branch.blocks):
            key = f'{name}_{index}'
            calls[key] = 0

            def check(module, args, key=key):
                if module.use_relation_bias or module.bias_projection is not None:
                    raise ValueError('Dependency attention bias is present: '+key)
                calls[key] += 1

            if block.use_relation_bias or block.bias_projection is not None:
                raise ValueError('Expected projection-free no-bias layers')
            hooks.append(block.register_forward_pre_hook(check))
    action = player.get_action
    first = True

    def checked_action(observation, *args, **kwargs):
        nonlocal first
        result = action(observation, *args, **kwargs)
        if first:
            proof = json.loads((path/'first_policy_input.json').read_text())
            if not proof['exact']:
                raise ValueError('Initial policy input differs from the common snapshot')
            (path/'evaluation_ready.json').write_text(json.dumps(dict(
                dependency_bias=False, feature_fusion=True, sct_task_bias=True,
                first_policy_input_exact=True), indent=2)+'\n')
            first = False
        return result

    player.get_action = checked_action
    try:
        result = original_eval(player)
    finally:
        player.get_action = action
        for hook in hooks:
            hook.remove()
    if first or any(value != result['simulated_steps'] for value in calls.values()):
        raise ValueError('No-bias layer call counts differ from the rollout')
    (path/'nobias_verification.json').write_text(json.dumps(dict(
        trained_without_dependency_bias=True, dependency_projection_parameters=0,
        dependency_fusion_preserved=True, sct_task_bias_preserved=True,
        layer_calls=calls), indent=2)+'\n')
    return result


formal_eval.run_formal_eval = checked_eval
runpy.run_path(str(ROOT/'eval/v4_formal_eval.py'), run_name='__main__')
