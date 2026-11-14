from copy import deepcopy
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
from bias_ablation import BiasAblation, state_digest
from stage1_protocol_policy import Stage1Policy
import stage2_scalability_layout
from utils.coupled_push_spec import physical_box_count


original_validate = stage2_scalability_layout.validate_env


def validate_single(env):
    if env['numAgents'] != 1:
        return original_validate(env)
    if env['numObjects'] != 2 or os.environ['STAGE2_SCENE_MODE'] != 'independent':
        raise ValueError('One-person ablation requires Independent and two object tokens')
    equivalent = deepcopy(env)
    equivalent.update(numAgents=4, numObjects=8)
    equivalent['stage2JointPush']['physical_boxes'] = physical_box_count(4, 8)
    stage2_scalability_layout.original_validate(equivalent)


stage2_scalability_layout.validate_env = validate_single
original_eval = formal_eval.run_formal_eval


@torch.no_grad()
def evaluate(player):
    task = player.env.task
    path = Path(task.cfg['args'].output_path)
    mode = os.environ.get('BIAS_ABLATION', 'none')
    native = handler = None
    proof = {}
    original_action = player.get_action

    def normalization():
        state = dict(player.running_mean_std.state_dict())
        if player._normalize_amp_input:
            state.update({'amp/'+key: value for key, value in player._amp_input_mean_std.state_dict().items()})
        return state_digest(state)

    def action(obs, is_determenistic=False):
        nonlocal native, handler, proof
        if handler is None:
            if os.environ.get('STAGE1_POLICY_CHECKPOINT'):
                native = Stage1Policy(player, os.environ['STAGE1_POLICY_CHECKPOINT'], obs['obs'])
            handler = BiasAblation(player.model.a2c_network, mode)
            proof['model_sha256_before'] = state_digest(player.model.state_dict())
            proof['normalization_sha256_before'] = normalization()
            first = obs['obs'].clone()
            parked = torch.load(path/'physics_only_reset.pt', map_location=task.device, weights_only=False)
            if not torch.equal(first, parked['observed_scene']):
                raise ValueError('First policy input differs from canonical reset observation')
            cpu_rng = torch.get_rng_state()
            gpu_rng = torch.cuda.get_rng_state(player.device)

            def restore_rng():
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(gpu_rng, player.device)

            baseline = original_action(obs, True).clone()
            restore_rng()
            unchanged = original_action(obs, True).clone()
            if not torch.equal(baseline, unchanged):
                raise ValueError('Disabled ablation changed baseline actions')
            handler.enabled = mode != 'none'
            restore_rng()
            ablated = original_action(obs, True).clone()
            restore_rng()
            if not torch.equal(first, obs['obs']):
                raise ValueError('Ablation probe modified observations')
            torch.save(first.cpu(), path/'first_policy_input.pt')
            proof.update(first_policy_input_exact=True, no_ablation_action_exact=True,
                observation_unchanged_by_probe=True, rng_restored_after_probe=True,
                action_max_abs_difference=float((baseline-ablated).abs().max()),
                diagnostic_forward_calls=3, native_verification_before_ablation=native is not None)
            handler.reset_counts()
            (path/'first_policy_input_proof.json').write_text(json.dumps(proof, indent=2))
            if os.environ.get('BIAS_ABLATION_PROBE_ONLY') == '1':
                proof.update(mode=mode, zero_output_verified=handler.zero_checks,
                             original_bias_max_abs=handler.original_max)
                (path/'first_policy_input_proof.json').write_text(json.dumps(proof, indent=2))
                print('BIAS_ABLATION_PROBE '+json.dumps(proof), flush=True)
                raise SystemExit(0)
        return original_action(obs, is_determenistic)

    player.get_action = action
    try:
        result = original_eval(player)
    finally:
        player.get_action = original_action
    proof.update(handler.finish(result['simulated_steps']))
    proof['model_weights_unchanged'] = proof['model_sha256_before'] == state_digest(player.model.state_dict())
    proof['normalization_unchanged'] = proof['normalization_sha256_before'] == normalization()
    if not proof['model_weights_unchanged'] or not proof['normalization_unchanged']:
        raise ValueError('Ablation changed checkpoint weights or normalization')
    if native is not None:
        metadata = dict(native.finish(), action_calls=result['simulated_steps'],
                        native_actor_check_scope='before inference ablation')
        (path/'stage1_policy.json').write_text(json.dumps(metadata, indent=2))
        result.update(policy='stage1_26k', experiment='stage1_push_26k_protocol', stage1_policy=metadata)
    result['bias_ablation'] = proof
    (path/'bias_ablation.json').write_text(json.dumps(proof, indent=2))
    (path/'result.json').write_text(json.dumps(result, indent=2))
    return result


if os.environ.get('REFERENCE_ONLY') != '1':
    formal_eval.run_formal_eval = evaluate
runpy.run_path(str(ROOT/'eval/stage2_scalability_eval.py'), run_name='__main__')
