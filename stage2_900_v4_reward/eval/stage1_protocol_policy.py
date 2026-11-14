from copy import deepcopy
import hashlib
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from utils.before_object_alias import encoder_packet, goal_aligned_scene


def stage1_scene(observation, people, objects):
    scene = observation[:, :-people]
    packet = encoder_packet(scene[:, -5*people:], people, objects)
    scene = goal_aligned_scene(scene, people, objects)
    return torch.cat((scene[:, :-5*people], packet), -1)


class Stage1Policy:
    def __init__(self, player, path, observation):
        from learning.multi_agent.stage2_push_network import transfer_stage1
        self.player = player
        self.task = player.env.task
        self.path = Path(path).resolve(strict=True)
        self.checkpoint = torch.load(self.path, map_location=player.device, weights_only=False)
        if self.checkpoint['epoch'] != 26000:
            raise ValueError('Expected Stage1 epoch 26000')
        self.saved = self.checkpoint['relation_experiment_config']
        self.network = player.model.a2c_network
        transfer_stage1(player.model, self.checkpoint)
        player.running_mean_std.load_state_dict(self.checkpoint['running_mean_std'], strict=True)
        player.running_mean_std.eval()
        if player._normalize_amp_input:
            player._amp_input_mean_std.load_state_dict(self.checkpoint['amp_input_mean_std'], strict=True)
        for encoder in (self.network.actor_encoder, self.network.critic_encoder):
            encoder.context_fusion.merge_shared = False
            encoder.set_task_graph(deepcopy(self.saved['env']['relationGraph']))

        def encode(network, encoder, coordination, obs):
            return encoder(stage1_scene(obs, network.num_agents, network.num_objects))

        self.network._encode = MethodType(encode, self.network)
        player.model.eval().requires_grad_(False)
        self.coordination_calls = 0

        def reject_coordination(module, args):
            self.coordination_calls += 1
            raise ValueError('Stage1 evaluation invoked Stage2 coordination')

        self.hooks = [module.register_forward_pre_hook(reject_coordination) for module in
                      (self.network.coordination, self.network.critic_coordination)]
        self.check_weights()
        self.verify_native(observation)

    def check_weights(self):
        live = self.player.model.state_dict()
        if not all(torch.equal(value, live[key]) for key, value in self.checkpoint['model'].items()):
            raise ValueError('Stage1 checkpoint weights changed')
        for key, value in self.checkpoint['running_mean_std'].items():
            if not torch.equal(value, self.player.running_mean_std.state_dict()[key]):
                raise ValueError('Stage1 normalization statistics changed')

    def verify_native(self, observation):
        from learning.multi_agent.stage1_push_network import Stage1PushBuilder
        from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        builder = Stage1PushBuilder()
        builder.load(self.saved['train']['params']['network'])
        task, m, o = self.task, self.task.num_agents, self.task.num_objects
        native = nn.Module()
        native.a2c_network = builder.build('amp', actions_num=task.get_action_size(),
            input_shape=(task.get_obs_size()-m,), amp_input_shape=(task.get_num_amp_obs(),),
            value_size=1, num_agents=m, num_objects=o, observation_mode=task.get_policy_obs_mode(),
            scene_entity_sizes=task.get_scene_entity_sizes(), scene_kinematic_size=task.get_scene_kinematic_size(),
            scene_arena_scale=task.get_scene_arena_scale(), relation_reward_mode=self.saved['env']['relationReward']['mode'],
            relation_graph_spec=self.saved['env']['relationGraph'], device=task.device).to(task.device)
        native.load_state_dict(self.checkpoint['model'], strict=True)
        native.eval().requires_grad_(False)
        normalizer = SceneRunningMeanStd(task.get_scene_entity_sizes(), [m, o, m],
            normalized_sizes=task.get_scene_normalized_entity_sizes(),
            kinematic_size=task.get_scene_kinematic_size(), extra_passthrough_size=5*m).to(task.device)
        normalizer.load_state_dict(self.checkpoint['running_mean_std'], strict=True)
        normalizer.eval()
        converted = stage1_scene(observation, m, o)
        normalized = normalizer(converted)
        if not torch.equal(normalized, stage1_scene(self.player._preproc_obs(observation), m, o)):
            raise ValueError('Stage1 normalization differs after input conversion')
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng)
        expected = native.a2c_network.eval_actor(normalized)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng)
        actual = self.network.eval_actor(self.player._preproc_obs(observation))
        if not all(torch.equal(a, b) for a, b in zip(actual, expected)):
            raise ValueError('Native Stage1 mean/sigma differs')
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng)
        self.metadata = dict(checkpoint=str(self.path), epoch=26000, frame=self.checkpoint.get('frame'),
            sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(), source_tensors=len(self.checkpoint['model']),
            native_actor_exact=True, normalization_exact=True, merge_shared=False,
            coordination_used=False, external_wait_gate=False, input='goal-aligned enable alias; individual PUSH regions',
            physical_scene_changed=False, distractor_attention='all_observed_tokens')

    def finish(self):
        self.check_weights()
        for hook in self.hooks:
            hook.remove()
        return dict(self.metadata, weights_unchanged=True, normalization_unchanged=True,
                    coordination_calls=self.coordination_calls)
