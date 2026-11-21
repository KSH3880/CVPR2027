import importlib.util
import sys

import torch

from common import COOP_ROOT, MODELS
from stage1_protocol_policy import Stage1Policy


def load_coop(player):
    for name in ('mean_cooperation', 'sct_coopft_network'):
        full = 'learning.multi_agent.'+name
        spec = importlib.util.spec_from_file_location(full, COOP_ROOT/f'tokenhsi/learning/multi_agent/{name}.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
    builder_class = sys.modules['learning.multi_agent.sct_coopft_network'].SCTCoopFTBuilder
    saved = torch.load(MODELS['coopft'], map_location='cpu', weights_only=False)
    cfg = saved['relation_experiment_config']
    task = player.env.task
    builder = builder_class()
    builder.load(cfg['train']['params']['network'])
    devices = [torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        network = builder.build('amp', actions_num=task.get_action_size(), input_shape=(task.get_obs_size(),),
            amp_input_shape=(task.get_num_amp_obs(),), value_size=1, num_agents=task.num_agents,
            num_objects=task.num_objects, observation_mode=task.get_policy_obs_mode(),
            scene_entity_sizes=task.get_scene_entity_sizes(), scene_kinematic_size=task.get_scene_kinematic_size(),
            scene_arena_scale=task.get_scene_arena_scale(), relation_reward_mode=cfg['env']['relationReward']['mode'],
            relation_graph_spec=cfg['env']['relationGraph'], device=task.device).to(task.device)
    player.model.a2c_network = network
    player.model.load_state_dict(saved['model'], strict=True)
    player.running_mean_std.load_state_dict(saved['running_mean_std'], strict=True)
    if player._normalize_amp_input:
        player._amp_input_mean_std.load_state_dict(saved['amp_input_mean_std'], strict=True)
    player.model.eval().requires_grad_(False)
    return saved


def check_weights(player, saved):
    for live, expected in ((player.model.state_dict(), saved['model']),
                           (player.running_mean_std.state_dict(), saved['running_mean_std'])):
        if any(k not in live or not torch.equal(v, live[k].detach().cpu()) for k, v in expected.items()):
            raise ValueError('Policy/normalization changed')


def load(player, model, obs):
    if model in ('sct', 'fsm'):
        return Stage1Policy(player, MODELS[model], obs), None
    saved = load_coop(player) if model == 'coopft' else torch.load(MODELS[model], map_location='cpu', weights_only=False)
    check_weights(player, saved)
    return None, saved
