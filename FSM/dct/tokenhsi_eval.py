import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
TOKEN = ROOT/'table89_assets/tokenhsi_push'
sys.path[:0] = [str(Path(__file__).parent), str(ROOT/'FSM'), str(ROOT/'eval'),
               str(ROOT/'tokenhsi'), str(TOKEN), str(TOKEN/'scripts')]
import isaacgym
import torch

from common import MODELS, read, write
from tokenhsi_table_policy import TokenHSIPolicy, project_scene, LIMITATION
from learning.amp_players import AMPPlayerContinuous
import policy

MODELS['tokenhsi'] = TOKEN/'checkpoints/TokenHSIPushKD_00040500.pth'


def load_tokenhsi(player, model, observation):
    if model != 'tokenhsi':
        raise ValueError('Expected TokenHSI condition')
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        native = TokenHSIPolicy(MODELS[model], player.env.task.device)
    if native.saved['epoch'] != 40500:
        raise ValueError('Expected TokenHSI epoch 40500')
    player._tokenhsi = native

    def reject_carrier(*args, **kwargs):
        raise ValueError('Scene carrier policy must not generate actions')

    player.model.a2c_network.eval_actor = reject_carrier
    return native, None


@torch.no_grad()
def token_action(player, obs, is_determenistic=False):
    task, native = player.env.task, player._tokenhsi
    raw = project_scene(obs['obs'], task.num_agents, task.num_objects)
    first = native.calls == 0
    action = native.action(raw)
    if first:
        expected = project_scene(task.obs_buf, task.num_agents, task.num_objects)
        if not torch.equal(raw, expected):
            raise ValueError('Native input differs from current scene')
        torch.save(dict(scene=obs['obs'].cpu(), raw=raw.cpu(), normalized=native.normalize(raw).cpu(),
                        action=action.cpu()), Path(task.cfg['args'].output_path)/'tokenhsi_first_input.pt')
    return action


policy.load = load_tokenhsi
AMPPlayerContinuous.get_action = token_action
runpy.run_path(str(ROOT/'FSM/dct/evaluate.py'), run_name='__main__')
path = Path(os.environ['DCT_OUTPUT'])
result = read(path/'result.json')
result.update(distractor_attention='unsupported: fixed397D self/assigned-task input',
    protocol_limitations=[LIMITATION], actual_policy_input='tokenhsi_first_input.pt',
    first_policy_input_file='common scene before native397D conversion')
write(path/'result.json', result)
