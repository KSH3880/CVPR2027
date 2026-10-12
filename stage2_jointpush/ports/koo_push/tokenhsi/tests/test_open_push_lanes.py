"""Lane ownership invariants, independent of Isaac Gym."""
import copy
from pathlib import Path
import sys
import torch
import pytest
import yaml
from utils.open_push_lanes import sample_lanes,owner_values,logical_order
from utils.open_push_spec import graph_for_tasks,validate_env

ROOT=Path(__file__).resolve().parents[1]


def test_sampling_bijection_and_frequency():
    torch.manual_seed(91)
    lanes=sample_lanes(10000,'cpu',True)
    assert torch.equal(lanes.sort(-1).values,torch.arange(2).expand_as(lanes))
    assert .48 < lanes[:,0].float().mean() < .52
    assert torch.equal(sample_lanes(8,'cpu'),torch.arange(2).expand(8,2))
    assert sample_lanes(8,'cpu',True,'swapped').tolist()==[[1,0]]*8
    with pytest.raises(ValueError):sample_lanes(1,'cpu','true')
    with pytest.raises(ValueError):sample_lanes(1,'cpu',True,'invalid')


@pytest.mark.parametrize('tasks',[[0,0],[0,1],[1,0],[1,1]])
def test_graph_and_physical_binding_agree(tasks):
    env=yaml.safe_load((ROOT/'data/cfg/multi_agent/open_push_push180_random_lanes.yaml').read_text())['env']
    validate_env(env)
    lanes=torch.tensor([[0,1],[1,0]])
    graph=graph_for_tasks(env['relationGraph'],torch.tensor([tasks,tasks]))
    physical=logical_order(lanes).gather(1,graph.edge_dst[:,::2]-2)
    assert torch.equal(physical,lanes+2*torch.tensor(tasks))
    forces=torch.tensor([[[11.,12.,13.],[21.,22.,23.]]]*2)
    assert owner_values(forces,lanes)[1].tolist()==forces[1].flip(0).tolist()


def test_only_lane_option_changes_and_bank_contract_preserved():
    from utils.open_push_rsi import bank_contract
    base=yaml.safe_load((ROOT/'data/cfg/multi_agent/open_push_late_rsi_hinge_friction_push2m_push180.yaml').read_text())
    new=yaml.safe_load((ROOT/'data/cfg/multi_agent/open_push_push180_random_lanes.yaml').read_text())
    expected=copy.deepcopy(base);expected['env']['openPushRandomizeLanes']=True
    assert new==expected
    for cfg in (base,new):cfg['env']['motion_file']='tokenhsi/data/dataset_open_push_push180.yaml'
    assert bank_contract(base['env'])==bank_contract(new['env'])


@pytest.mark.parametrize('tasks',[[0,0],[0,1],[1,0],[1,1]])
def test_only_owner_of_moving_physical_box_gets_credit(tasks):
    from env.tasks.multi_agent.open_push_reward import OpenPushRuntime
    env=yaml.safe_load((ROOT/'data/cfg/multi_agent/open_push_push180_random_lanes.yaml').read_text())['env']
    lanes=torch.tensor([[0,1],[1,0]])
    boxes=torch.zeros(2,2,13);boxes[:,:,2]=.65;boxes[:,:,6]=1
    boxes[:,:,1]=torch.tensor([-5.,5.])
    owner=owner_values(boxes,lanes)
    goals=owner[...,:3].clone();goals[...,1]+=2
    runtime=OpenPushRuntime(2,env['relationGraph'],env['relationReward'],env['doorHold'],1/30,'cpu',env['pushBox']['size'])
    runtime.reset(torch.arange(2),torch.tensor([tasks,tasks]),owner)
    runtime.reset_push_credit(torch.arange(2),owner,goals)
    boxes[:,0,1]+=.1
    runtime.accumulate_push_credit(owner_values(boxes,lanes),goals,lanes==0)
    expected=((lanes==0)&(torch.tensor(tasks)==1)).float()*.1
    torch.testing.assert_close(runtime.push_credit,expected,atol=1e-6,rtol=1e-5)


def test_transfer_preserves_state_and_rejects_other_changes():
    sys.path.insert(0,str(ROOT/'scripts/multi_agent'))
    from transfer_open_push_random_lanes import prepare,SOURCE_CONFIG,TARGET_CONFIG,TRAIN_CONFIG,TARGET_MANIFEST
    from transfer_open_push_push180 import expected_contract
    from utils.relation_task_spec import checkpoint_metadata
    source=yaml.safe_load(SOURCE_CONFIG.read_text())['env'];target=yaml.safe_load(TARGET_CONFIG.read_text())['env']
    train=yaml.safe_load(TRAIN_CONFIG.read_text())
    state={k:{'test':torch.tensor([1.,2.])} for k in ('model','optimizer','running_mean_std','reward_mean_std','amp_input_mean_std')}
    state.update(amp_enabled=True,epoch=3500,frame=123,open_push_contract=expected_contract(source,TARGET_MANIFEST,'bankhash'),relation_metadata=checkpoint_metadata(source['relationReward']),relation_experiment_config=dict(num_envs=2048,num_agents=2,num_objects=4,observation_size=684,env=source,train=train))
    result=prepare(state,source,target,'bankhash',Path('/tmp/test-lanes.pth'))
    for k in ('model','optimizer','running_mean_std','reward_mean_std','amp_input_mean_std','epoch','frame'):
        assert result[k] is state[k]
    assert 'lane_randomization' not in state['open_push_contract']
    for key in ('pushBox','relationReward','openPushRSI'):
        bad=copy.deepcopy(target)
        if key=='pushBox':bad[key]['mass']+=1
        elif key=='relationReward':bad[key]['push']['target_distance']+=1
        else:bad[key]['bank_file']+='invalid'
        with pytest.raises(ValueError):prepare(state,source,bad,'bankhash',Path('/tmp/test-lanes.pth'))
    bad=dict(state);bad.pop('optimizer')
    with pytest.raises(ValueError):prepare(bad,source,target,'bankhash',Path('/tmp/test-lanes.pth'))
