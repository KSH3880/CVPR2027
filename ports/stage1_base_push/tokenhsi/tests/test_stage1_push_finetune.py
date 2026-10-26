from copy import deepcopy
from pathlib import Path
import math
import pytest
import torch
import yaml
from utils.stage1_push_spec import validate_env, sample_asset_sizes
from utils.stage1_push_scene import sample_four_box_scene, advance_rsi_box
from utils.push_scene import random_at_goal_valid, forward_xy

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT/'data/cfg/multi_agent/stage1_push_finetune.yaml').read_text())['env']
SOURCE = yaml.safe_load((ROOT/'data/cfg/multi_agent/stage1_push_finetune_source.yaml').read_text())['env']


def test_finetune_asset_task_distribution_and_contact_weights():
    validate_env(CFG)
    assert CFG['pushExtension']['contact_weights'] == [.2,.2,.2]
    torch.manual_seed(42)
    sizes, posterior = sample_asset_sizes(100000,CFG,'cpu')
    expected = torch.tensor([0.,.05,.125,.1625,.1625,.5])
    torch.testing.assert_close(posterior.mean((0,1)),expected,atol=.003,rtol=0)
    push = posterior[...,5] == 1
    payload = sizes[:,:2][push]
    spare = sizes[:,2:][push]
    assert set((payload[:,2]*20).round().int().tolist()) == set(range(29,35))
    for boxes in (payload,spare):
        assert (boxes[:,:2]>=1.6-1e-6).all() and (boxes[:,:2]<=2.2+1e-6).all()
        assert (boxes[:,2]>=1.45-1e-6).all() and (boxes[:,2]<=1.7+1e-6).all()
        assert (boxes[:,:2].amin(-1)>=1.1*boxes[:,2]-1e-6).all()
    assert len(CFG['pushBox']['size_profiles']) == 167
    assert CFG['pushBox']['mass'] == 20 and CFG['pushBox']['friction'] == .4
    old = yaml.safe_load((ROOT/'data/cfg/multi_agent/stage1_v2_fixed.yaml').read_text())['env']
    assert CFG['relationReward'] == old['relationReward']
    assert all(row[4] == 0 for row in CFG['templateRsi'].values())  # no OMOMO RSI
    original = yaml.safe_load((ROOT/'data/dataset_loco_sit_carry_climb.yaml').read_text())['motions']
    new = yaml.safe_load((ROOT/'data/dataset_stage1_push_finetune.yaml').read_text())['motions']
    assert all(new[key] == rows for key,rows in original.items())
    assert len(new['push_loco']) == 4 and len(new['push']) == 21


@pytest.mark.parametrize('seed',[42,20261023,7])
def test_stage1_square_cone_start_bounds_and_full_paths(seed):
    xy,q,goals = sample_four_box_scene(2048,SOURCE,'cpu',torch.Generator().manual_seed(seed))
    assert random_at_goal_valid(xy[:,:2],q[:,:2],goals,SOURCE).all()
    anchors = torch.tensor(SOURCE['pushScene']['owner_anchor_xy'])
    distances = (xy[:,:2]-anchors).norm(dim=-1)
    assert distances.min() >= 1-1e-5 and distances.max() <= 10+1e-5
    assert (goals.abs() <= 4.5).all()
    delta = goals-xy[:,:2]
    cosine = (delta/delta.norm(dim=-1,keepdim=True)*forward_xy(q[:,:2])).sum(-1)
    assert cosine.min() >= math.cos(math.pi/4)-1e-6
    assert ((goals-xy[:,:2]).norm(dim=-1)>=1-1e-5).all()
    assert ((goals-anchors).norm(dim=-1)>=1-1e-5).all()
    angles = cosine.acos()
    assert (angles > .7).any() and (angles < .05).any()


def test_rsi_10m_goal_4m_progress_leaves_6m_and_goal_is_fixed():
    start = torch.tensor([[0.,0.,.7],[3.,-2.,.65]])
    goal = start+torch.tensor([[0.,10.,0.],[6.,8.,0.]])
    before = goal.clone()
    placed = advance_rsi_box(start,goal,torch.tensor([.4,.4]))
    torch.testing.assert_close((placed-start).norm(dim=-1),torch.full((2,),4.))
    torch.testing.assert_close((goal-placed).norm(dim=-1),torch.full((2,),6.))
    assert torch.equal(goal,before)
    with pytest.raises(ValueError):
        advance_rsi_box(start,goal,torch.tensor([1.1,0.]))


def test_contact_and_at_completed_reward_is_1_point_2():
    from utils.stage1_push_spec import compose_graph
    from env.tasks.multi_agent.stage1_push_reward import Stage1PushRuntime
    graph = compose_graph(torch.full((1,2),5))
    runtime = Stage1PushRuntime(1,graph,CFG['relationReward'],'cpu',contact_weights=[.2,.2,.2])
    runtime.contact[:] = True
    phi = torch.ones(1,4)
    result = runtime.step(phi,phi,torch.zeros_like(phi),torch.zeros_like(phi),torch.zeros_like(phi))
    torch.testing.assert_close(result['agent_task_reward'],torch.full((1,2),1.2))


def test_amp_transfer_preserves_original_family_logits():
    import importlib.util
    path = ROOT.parent/'prepare_finetune.py'
    spec = importlib.util.spec_from_file_location('prepare_finetune_test',path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model = {'a2c_network._disc_mlp.0.weight':torch.randn(1024,1320)}
    for side in ('actor','critic'):
        model[f'a2c_network.{side}_encoder.edge_encoder.category_embed.weight'] = torch.randn(4,64)
    expanded = module.expand_model(model)
    old = torch.randn(3,10,132)
    # Each of the original family labels must produce the original hidden activation.
    old[...,129:] = torch.eye(3)[:,None]
    new = torch.cat((old,torch.zeros(3,10,1)),-1)
    key = 'a2c_network._disc_mlp.0.weight'
    torch.testing.assert_close(old.flatten(1)@model[key].T,new.flatten(1)@expanded[key].T,rtol=1e-4,atol=1e-4)
