import ast
import copy
import importlib.util
from pathlib import Path

import torch
import yaml
from utils.open_push_spec import compile_graph, PUSH
from utils.edge_door_spec import OPEN
from env.tasks.multi_agent.open_push_reward import OpenPushRuntime

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'scripts/multi_agent/native4_zeroshot'
spec = importlib.util.spec_from_file_location('mixed4', PACKAGE / 'mixed4.py')
mixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mixed)


def graph_builder():
    namespace = dict(torch=torch, _original_compile=compile_graph)
    for node in ast.parse((PACKAGE / 'launch.py').read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == 'graph4':
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'graph4', 'exec'), namespace)
    return namespace['graph4']


def test_all_sixteen_mixed_task_patterns_bind_correct_doors_boxes_and_goals():
    tasks = (torch.arange(16)[:, None] >> torch.arange(4)[None]) % 2
    config = dict(sampler='two_agent_open_push', semantic_only=True,
                  task_probabilities={'OPEN': .5, 'PUSH': .5})
    graph = mixed.mixed_graph(config, tasks, graph_builder())
    actors = torch.arange(4)[None].expand(16, -1)
    assert torch.equal(graph.edge_src[:, ::2], actors)
    assert torch.equal(graph.edge_dst[:, ::2], 4 + actors + 4 * tasks)
    assert torch.equal(graph.edge_relation[:, ::2], torch.where(tasks.bool(), PUSH, OPEN))
    assert torch.equal(graph.edge_src[:, 1::2], 8 + actors)
    assert torch.equal(graph.edge_dst[:, 1::2], 12 + actors)
    assert torch.equal(graph.edge_valid[:, 1::2], tasks.bool())


def test_mixed_reward_matches_original_two_person_steps_and_updates_runtime_views():
    env = yaml.safe_load((ROOT / 'data/cfg/multi_agent/open_push_push180_contact_at_warmstart.yaml').read_text())['env']
    runtime = OpenPushRuntime(1, env['relationGraph'], env['relationReward'], env['doorHold'], 1 / 30, 'cpu', env['pushBox']['size'], num_agents=4)
    runtime.tasks[:] = torch.tensor([[0, 1, 1, 0]])
    roots = torch.zeros(1, 4, 3); roots[..., 2] = .9
    hands = torch.zeros(1, 4, 2, 3); hands[..., 2] = .8
    handles = torch.ones(1, 4, 3)
    angles = torch.zeros(1, 4)
    boxes = torch.zeros(1, 4, 13); boxes[..., 2] = .65; boxes[..., 6] = 1
    goals = boxes[..., :3].clone(); goals[..., 1] = 2
    runtime.start.copy_(boxes[..., :3])
    expected = []
    references = []
    values = (roots, hands, handles, angles, boxes, goals)
    for start in (0, 2):
        ref = copy.deepcopy(mixed.pair_view(runtime, start))
        references.append(ref)
        expected.append(OpenPushRuntime.step(ref, *[value[:, start:start + 2] for value in values]))
    actual = mixed.mixed_step(runtime, OpenPushRuntime.step, *values)
    for key in actual:
        assert torch.equal(actual[key], torch.cat([value[key] for value in expected], 1))
    assert torch.equal(runtime.phi, torch.cat([ref.phi for ref in references], 1))
    assert torch.equal(runtime.done, torch.cat([ref.done for ref in references], 1))
