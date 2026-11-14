import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/'sct_coopft'
ORIGINAL = SOURCE/'output_etc/check/sct_coopft_4000_tables12_seeds167_20261111/entry.py'
sys.path[:0] = [str(SOURCE/'eval'), str(SOURCE/'tokenhsi')]
import isaacgym
import torch
from utils import output_paths
from learning.multi_agent import formal_eval


resolve = output_paths.resolve_output_path


def local_output(*args, **kwargs):
    kwargs['root'] = ROOT
    path = Path(resolve(*args, **kwargs))
    return str(path if path.is_absolute() else ROOT/path)


output_paths.resolve_output_path = local_output
original_eval = formal_eval.run_formal_eval


def check_weights(player, checkpoint):
    live = player.model.state_dict()
    if any(key not in live or not torch.equal(value, live[key].detach().cpu())
           for key, value in checkpoint['model'].items()):
        raise ValueError('Coop. FT policy differs from checkpoint')
    normalizer = player.running_mean_std.state_dict()
    if any(not torch.equal(value, normalizer[key].detach().cpu())
           for key, value in checkpoint['running_mean_std'].items()):
        raise ValueError('Coop. FT normalization differs from checkpoint')


@torch.no_grad()
def checked_eval(player):
    task = player.env.task
    path = Path(task.cfg['args'].output_path)
    checkpoint = torch.load(task.cfg['args'].checkpoint, map_location='cpu', weights_only=False)
    check_weights(player, checkpoint)
    get_action = player.get_action
    checked = False

    def action(observation, *args, **kwargs):
        nonlocal checked
        if not checked:
            canonical = Path(os.environ['STAGE2_REFERENCE_ROOT'])/f'general_seed{task.cfg["args"].seed}'/'observed_scene.pt'
            expected = torch.load(canonical, map_location='cpu', weights_only=False)
            actual = observation['obs'].detach().cpu()
            if not torch.equal(actual, expected):
                raise ValueError('Actual first policy input differs from canonical observation')
            proof = dict(exact=True, reference=str(canonical), epoch=checkpoint['epoch'],
                         observation_sha256=hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest())
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
        raise ValueError('No actual policy input was checked')
    proof = dict(epoch=checkpoint['epoch'], model_tensors=len(checkpoint['model']),
                 first_policy_input_exact=True, weights_exact_before_after=True, normalization_exact_before_after=True)
    (path/'coopft_policy_verification.json').write_text(json.dumps(proof, indent=2)+'\n')
    result['coopft_policy_verification'] = proof
    return result


formal_eval.run_formal_eval = checked_eval
source = ORIGINAL.read_text().replace("'sct_coopft_ppo_4k'", "'sct_coopft_ppo_checkpoint'")
exec(compile(source, str(ORIGINAL), 'exec'), dict(__name__='__main__', __file__=str(ORIGINAL)))
