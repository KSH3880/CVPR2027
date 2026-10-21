"""Opt-in four-person OPEN/PUSH capture; retain original two-person reward math."""
from types import SimpleNamespace
import torch
from utils.edge_ontop_spec import expand_graph
from utils.edge_door_spec import OPEN
from utils.open_push_spec import PUSH


def mixed_graph(spec, tasks, graph_builder):
    graph = expand_graph(graph_builder(spec, 4, 8, tasks.device), len(tasks))
    agents = torch.arange(4, device=tasks.device)[None]
    push = tasks == 1
    graph.edge_dst[:, ::2] = 4 + agents + 4 * push.long()
    graph.edge_relation[:, ::2] = torch.where(push, PUSH, OPEN)
    graph.edge_valid[:, 1::2] = push
    graph.required_goal[:, 1::2] = push
    return graph


def pair_view(runtime, start):
    fields = {}
    for key, value in vars(runtime).items():
        if torch.is_tensor(value) and value.ndim >= 2:
            if value.shape[1] == 4:
                value = value[:, start:start + 2]
            elif value.shape[1] == 8:
                value = value[:, 2 * start:2 * (start + 2)]
        fields[key] = value
    return SimpleNamespace(**fields)


def mixed_step(runtime, original_step, roots, hands, handles, angles, boxes, goals):
    results = []
    for start in (0, 2):
        values = [value[:, start:start + 2] for value in (roots, hands, handles, angles, boxes, goals)]
        results.append(original_step(pair_view(runtime, start), *values))
    return {key: torch.cat([result[key] for result in results], dim=1) for key in results[0]}


def install(task_type, reward_module, original_step, graph_builder):
    def sample_tasks(self, env_ids):
        preset = self._task_graph_preset
        if preset == 'random_open_push':
            return (torch.rand(len(env_ids), 4, device=self.device) >=
                    self._relation_graph_spec['task_probabilities']['OPEN']).long()
        patterns = {'push': [1] * 4, 'open': [0] * 4,
                    'open_push': [0, 1, 0, 1], 'push_open': [1, 0, 1, 0]}
        return torch.tensor(patterns[preset], device=self.device).expand(len(env_ids), -1).clone()

    task_type._sample_tasks = sample_tasks
    reward_module.graph_for_tasks = lambda spec, tasks: mixed_graph(spec, tasks, graph_builder)
    reward_module.OpenPushRuntime.step = lambda self, *args: mixed_step(self, original_step, *args)
