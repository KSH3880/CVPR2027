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
from bias_ablation import state_digest
from ind_dct_ablation import DctAblation

original_eval = formal_eval.run_formal_eval


@torch.no_grad()
def evaluate(player):
    task = player.env.task
    path = Path(task.cfg['args'].output_path)
    checkpoint = torch.load(task.cfg['args'].checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['epoch'] != 2000:
        raise ValueError('Requires V4 epoch2000')

    def check_checkpoint():
        for live, saved in ((player.model.state_dict(), checkpoint['model']),
                            (player.running_mean_std.state_dict(), checkpoint['running_mean_std'])):
            if live.keys() != saved.keys() or any(not torch.equal(live[k].cpu(), v) for k, v in saved.items()):
                raise ValueError('Weights or normalization differ from checkpoint')

    check_checkpoint()
    handler = DctAblation(player.model.a2c_network, os.environ['IND_DCT_ABLATION'])
    original_action = player.get_action
    proof = {}
    amp_before = state_digest(player._amp_input_mean_std.state_dict())

    def action(obs, is_determenistic=False):
        if not proof:
            first = obs['obs'].clone()
            reference = Path(os.environ['STAGE2_REFERENCE_ROOT'])/'general_seed1'/'observed_scene.pt'
            if not torch.equal(first.cpu(), torch.load(reference, map_location='cpu', weights_only=False)):
                raise ValueError('First input differs from shared reference')
            cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state(player.device)

            def restore():
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(gpu_rng, player.device)

            baseline = original_action(obs, True).clone()
            restore()
            repeated = original_action(obs, True).clone()
            if not torch.equal(baseline, repeated):
                raise ValueError('Baseline action is not reproducible')
            handler.enabled = handler.mode != 'none'
            restore()
            changed = original_action(obs, True).clone()
            restore()
            if not torch.equal(first, obs['obs']):
                raise ValueError('Probe modified observation')
            torch.save(first.cpu(), path/'first_policy_input.pt')
            proof.update(first_policy_input_exact=True, disabled_action_exact=True,
                         observation_unchanged=True, rng_restored=True,
                         action_max_abs_difference=float((baseline-changed).abs().max()))
            handler.reset_counts()
        return original_action(obs, is_determenistic)

    player.get_action = action
    try:
        result = original_eval(player)
    finally:
        player.get_action = original_action
    proof.update(handler.finish(result['simulated_steps']))
    check_checkpoint()
    if amp_before != state_digest(player._amp_input_mean_std.state_dict()):
        raise ValueError('AMP normalization changed')
    proof.update(weights_exact_before_after=True, normalization_exact_before_after=True)
    result['dct_ablation'] = proof
    (path/'dct_ablation.json').write_text(json.dumps(proof, indent=2))
    (path/'result.json').write_text(json.dumps(result, indent=2))
    return result


if os.environ.get('REFERENCE_ONLY') != '1':
    formal_eval.run_formal_eval = evaluate
runpy.run_path(str(ROOT/'eval/stage2_scalability_eval.py'), run_name='__main__')
