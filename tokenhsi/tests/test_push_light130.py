from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from env.tasks.multi_agent.push_reward import PushRuntime
from utils.edge_scenario_spec import compose_canonical_graph
from utils.push_navigation import sample_box_properties, sample_spawns
from utils.stage1_push import PUSH, check_contract, check_navigation_transfer, contract, load_bank

ROOT = Path(__file__).resolve().parents[2]
NAME = 'stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130'
ENV = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent'/f'{NAME}.yaml').read_text())['env']
OLD = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform.yaml').read_text())['env']


def test_variant_only_changes_requested_settings():
    expected = deepcopy(OLD)
    spec = expected['stage1Push']
    spec['box']['mass_range'] = [10.,15.]
    spec['teacher_kl_coef'] = 0.
    spec['success_radius'] = .2
    spec['rsi']['bank'] = ENV['stage1Push']['rsi']['bank']
    spec['navigation']['curriculum'][0]['half_angle_degrees'] = 65.
    assert ENV == expected
    bank = load_bank(ENV['stage1Push'], ROOT/'tokenhsi/data/dataset_stage1_push.yaml')
    previous = load_bank(OLD['stage1Push'], ROOT/'tokenhsi/data/dataset_stage1_push.yaml')
    assert bank['candidates'] == previous['candidates']
    assert bank['contract']['motion_hashes'] == previous['contract']['motion_hashes']
    size,mass = sample_box_properties(8192, ENV['stage1Push']['box'], 'cpu')
    torch.testing.assert_close(mass, 10+5*(size.prod(-1)-2.523)/(4.913-2.523))
    assert mass.min() >= 10 and mass.max() <= 15
    with pytest.raises(ValueError, match='contract mismatch'):
        check_contract({'stage1_push_contract':contract(OLD)}, ENV)
    check_navigation_transfer({'stage1_push_contract':contract(OLD)}, ENV)
    check_contract({'stage1_push_contract':contract(ENV)}, ENV)
    changed = deepcopy(ENV)
    changed['stage1Push']['success_radius'] = .3
    with pytest.raises(ValueError, match='contract mismatch'):
        check_contract({'stage1_push_contract':contract(ENV)}, changed)


def test_spawn_sector_uses_pushing_face_and_keeps_rsi_positions():
    boxes = torch.zeros(4096,4,13)
    boxes[...,6] = 1.
    boxes[:,1,0] = 20.
    boxes[:,2:,0] = 50.
    sizes = torch.full((4096,4,3),1.7)
    normals = torch.zeros(4096,2,3)
    normals[...,1] = 1.
    roots = torch.zeros(4096,2,3)
    roots[:,1,0] = 20.
    interaction = torch.zeros(4096,2,dtype=torch.bool)
    interaction[:256] = True
    result = sample_spawns(boxes,sizes,normals,roots,interaction,[1.65,6.],65.,.45)
    delta = result[...,:2]-boxes[:,:2,:2]
    angles = torch.atan2(delta[...,0],-delta[...,1])*180/torch.pi
    assert angles[~interaction].abs().max() <= 65.0001
    assert (angles[~interaction]>64).any() and (angles[~interaction]<-64).any()
    torch.testing.assert_close(result[interaction],roots[interaction])


@pytest.mark.parametrize('distance,z,success',[(.15,0.,True),(.2,.001,True),(.20001,0.,False),(.15,.00101,False)])
def test_success_radius_drives_reward_and_evaluation_without_changing_old_tasks(distance,z,success):
    graph = compose_canonical_graph(torch.full((2,2),3))
    graph.edge_relation[0,graph.edge_relation[0]==6] = PUSH
    data = dict(distance_xy=torch.full((2,2),distance),z_error=torch.full((2,2),z),
                phi_at=torch.full((2,2),torch.exp(torch.tensor(-10*(distance**2+z**2))).item()))
    task = SimpleNamespace(_push_env=torch.tensor([True,False]),_push_options=ENV['stage1Push'],
        _push_hits=torch.zeros(2,2,dtype=torch.bool),_push_state=lambda ids:{k:v[ids] for k,v in data.items()})
    runtime = PushRuntime(task,Stage1ContextRuntime(2,graph,ENV['relationReward'],'cpu'))
    phi = torch.full((2,4),data['phi_at'][0,0].item())
    zeros = torch.zeros_like(phi)
    original_runtime = Stage1ContextRuntime(2,graph,ENV['relationReward'],'cpu')
    original = original_runtime.step(phi,phi,zeros,zeros,zeros)
    reward = runtime.step(phi,phi,zeros,zeros,zeros)
    assert runtime.done[0].eq(success).all()
    assert reward['reward_saturated'][0].eq(success).all()
    if success:torch.testing.assert_close(reward['agent_task_reward'][0],torch.full((2,),1.2))
    torch.testing.assert_close(reward['agent_task_reward'][1],original['agent_task_reward'][1])
    torch.testing.assert_close(runtime.done[1],original_runtime.done[1])
