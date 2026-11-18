import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from learning.multi_agent.stage2_push_network import Stage2PushBuilder


class TeacherPolicy(nn.Module):
    def __init__(self, task, path, fusion):
        super().__init__()
        path = Path(path).resolve(strict=True)
        checkpoint = torch.load(path, map_location=task.device, weights_only=False)
        saved = checkpoint['relation_experiment_config']
        graph = saved['env']['relationGraph']
        if graph.get('human_relation_fusion') != fusion:
            raise ValueError('Wrong teacher relation fusion: ' + str(path))
        builder = Stage2PushBuilder()
        builder.load(saved['train']['params']['network'])
        self.a2c_network = builder.build('amp', actions_num=task.get_action_size(),
            input_shape=(task.get_obs_size(),), amp_input_shape=(task.get_num_amp_obs(),),
            value_size=1, num_agents=task.num_agents, num_objects=task.num_objects,
            observation_mode=task.get_policy_obs_mode(), scene_entity_sizes=task.get_scene_entity_sizes(),
            scene_kinematic_size=task.get_scene_kinematic_size(), scene_arena_scale=task.get_scene_arena_scale(),
            relation_reward_mode=saved['env']['relationReward']['mode'], relation_graph_spec=graph,
            device=task.device).to(task.device)
        self.load_state_dict(checkpoint['model'], strict=True)
        self.normalizer = SceneRunningMeanStd(task.get_scene_entity_sizes(),
            [task.num_agents, task.num_objects, task.num_agents],
            normalized_sizes=task.get_scene_normalized_entity_sizes(),
            kinematic_size=task.get_scene_kinematic_size(),
            extra_passthrough_size=task.get_relation_suffix_size()).to(task.device)
        self.normalizer.load_state_dict(checkpoint['running_mean_std'], strict=True)
        self.eval().requires_grad_(False)
        self.metadata = dict(path=str(path), epoch=checkpoint.get('epoch'),
            frame=checkpoint.get('frame'), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            fusion=fusion, state_tensors=len(checkpoint['model']))

    def means(self, obs, object_valid=None):
        self.a2c_network.object_valid = object_valid
        return self.a2c_network.eval_actor(self.normalizer(obs))[0]


class Stage2Teachers(nn.Module):
    def __init__(self, task, before_path, coupled_path, output_path=None):
        super().__init__()
        self.before = TeacherPolicy(task, before_path, 'before_zerobg')
        self.coupled = TeacherPolicy(task, coupled_path, 'zerobg')
        self.eval().requires_grad_(False)
        self.metadata = dict(before=self.before.metadata, coupled=self.coupled.metadata,
            solo_teacher='before', routing='shared PUSH members use coupled; all others use before',
            full_scene_observation=True, agents=task.num_agents, objects=task.num_objects,
            group_spawn_gap=5. if task.num_agents > 4 else None,
            group_sampler='sequential_solo_before_coupled' if task.num_agents > 4 else 'training_scene_mixture')
        directory = Path(output_path or task.cfg['args'].output_path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'teachers.json').write_text(json.dumps(self.metadata, indent=2))
        print('[two teachers] ' + json.dumps(self.metadata), flush=True)

    @torch.no_grad()
    def means(self, obs, coupled_members, object_valid=None):
        before = self.before.means(obs, object_valid)
        coupled = self.coupled.means(obs, object_valid)
        return torch.where(coupled_members.reshape(-1, 1), coupled, before)
