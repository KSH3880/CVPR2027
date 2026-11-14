import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tokenhsi'), str(ROOT/'eval')]
import isaacgym
import torch
from learning.multi_agent import formal_eval


original_eval = formal_eval.run_formal_eval


def check_weights(player, checkpoint):
    for live, saved in ((player.model.state_dict(), checkpoint['model']),
                        (player.running_mean_std.state_dict(), checkpoint['running_mean_std'])):
        if live.keys() != saved.keys() or any(not torch.equal(tensor, live[key].detach().cpu())
                                             for key, tensor in saved.items()):
            raise ValueError('V4 weights or normalization differ from checkpoint')


@torch.no_grad()
def checked_eval(player):
    task = player.env.task
    path = Path(task.cfg['args'].output_path)
    checkpoint = torch.load(task.cfg['args'].checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['epoch'] != 2000:
        raise ValueError('V4 formal evaluation requires epoch2000')
    check_weights(player, checkpoint)
    get_action = player.get_action
    checked = False

    def action(observation, *args, **kwargs):
        nonlocal checked
        if not checked:
            reference = Path(os.environ['STAGE2_REFERENCE_ROOT'])/f'general_seed{task.cfg["args"].seed}'/'observed_scene.pt'
            expected = torch.load(reference, map_location='cpu', weights_only=False)
            actual = observation['obs'].detach().cpu()
            if not torch.equal(actual, expected):
                raise ValueError('Actual first V4 input differs from shared reference')
            proof = dict(exact=True, epoch=2000, reference=str(reference),
                sha256=hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest())
            (path/'first_policy_input.json').write_text(json.dumps(proof, indent=2)+'\n')
            checked = True
        return get_action(observation, *args, **kwargs)

    player.get_action = action
    try:
        result = original_eval(player)
        check_weights(player, checkpoint)
    finally:
        player.get_action = get_action
    if not checked:
        raise ValueError('Actual first policy input was not checked')
    proof = dict(epoch=2000, weights_exact_before_after=True, normalization_exact_before_after=True)
    (path/'policy_preservation.json').write_text(json.dumps(proof, indent=2)+'\n')
    return result


formal_eval.run_formal_eval = checked_eval
runpy.run_path(str(ROOT/'eval'/('table2_eval.py' if 'TABLE2_SETTING' in os.environ
                               else 'stage2_scalability_eval.py')), run_name='__main__')
