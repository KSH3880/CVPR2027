"""Portable native4 diagnostic: real graph indices, geometry and source setup."""
import ast
import importlib.util
from pathlib import Path
import torch
import yaml
import pytest
from utils.open_push_spec import compile_graph
from utils.open_push_scene import sample_boxes
from utils.edge_ontop_spec import expand_graph
from utils.edge_stage1_spec import semantic_graph_packet

PACKAGE = Path(__file__).resolve().parents[1]/'scripts/multi_agent/native4_zeroshot'


def functions(path, namespace):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_four_physical_box_sampler_and_owner_bijection():
    ns = functions(PACKAGE/'native4_task.py', dict(torch=torch, _native_pair_boxes=sample_boxes))
    spec = dict(arena_center=[-4.,0.], arena_radius=4.5, min_station_distance=6., owner_loco_distance_range=[1.,2.])
    boxes, direction = ns['sample_boxes'](256, 'cpu', spec, [1.3]*3)
    assert boxes.shape == (256,4,13) and direction.shape == (256,4,3)
    distances = torch.cdist(boxes[...,:2], boxes[...,:2])
    distances.diagonal(dim1=1,dim2=2).fill_(float('inf'))
    assert distances.min() >= 6.
    assert (boxes[:,:2,1] < -5.5).all() and (boxes[:,2:,1] > 5.5).all()
    for mode in ('random','fixed','swapped'):
        lanes = ns['sample_lanes'](256, 'cpu', True, mode)
        assert lanes.sort(-1).values.eq(torch.arange(4)).all()


def test_all_push_edges_bind_to_four_distinct_boxes_and_goals():
    ns = functions(PACKAGE/'launch.py', dict(torch=torch, _original_compile=compile_graph))
    spec = dict(sampler='two_agent_open_push', semantic_only=True, task_probabilities={'OPEN':.5,'PUSH':.5})
    graph = ns['graph4'](spec,4,8)
    assert graph.edge_src.tolist() == [0,8,1,9,2,10,3,11]
    assert graph.edge_dst.tolist() == [8,12,9,13,10,14,11,15]
    assert semantic_graph_packet(expand_graph(graph,3),3).shape == (3,40)


@pytest.mark.parametrize('state_init', ['Start','Random'])
@pytest.mark.parametrize('agents', [2, 4])
def test_setup_restores_saved_source_and_rejects_wrong_initialization(tmp_path, state_init, agents):
    path = PACKAGE/'setup.py'
    spec = importlib.util.spec_from_file_location('native4_setup', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = tmp_path/'source.pth'
    saved = dict(relation_metadata={'schema_version':12}, open_push_contract={}, epoch=29000,
                 relation_experiment_config=dict(num_agents=2,num_objects=4,
                     env=dict(stateInit=state_init,openPushLayout={'lane_spacing':10},numAgents=2,numObjects=4),
                     train={'params':{'seed':123}}))
    torch.save(saved, source)
    before = source.read_bytes()
    out = tmp_path/'output'
    if state_init != 'Start':
        with pytest.raises(ValueError, match='Start'): module.prepare(out, str(source), agents)
        return
    assert module.prepare(out, str(source), agents) == source
    target = yaml.safe_load((out/'env.yaml').read_text())
    assert (target['env']['numAgents'],target['env']['numObjects']) == (agents, 2 * agents)
    assert target['env']['stateInit'] == 'Start'
    assert source.read_bytes() == before
