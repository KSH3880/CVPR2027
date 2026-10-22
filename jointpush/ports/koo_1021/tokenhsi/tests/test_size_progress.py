from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges, interaction_own_success
from env.tasks.multi_agent.edge_ontop_reward import inner_xy_region_error
from utils.edge_scenario_spec import sample_graph, classify_templates
from utils.unified_training import (validate_unified_env, EXPERT_PROBS, FAMILY_PROBS,
    family_from_templates, append_family, amp_family_ids, sample_family_matched)
from utils.edge_context_spec import HOLDING, AT
from utils.edge_ontop_spec import ON_TOP
from utils.edge_interaction_spec import SIT, CLIMB
from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata

ROOT=Path(__file__).resolve().parents[1]
BASE=yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_scenario_stage1_unified_shared_random_binding.yaml').read_text())['env']
ENV=yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_scenario_stage1_size_progress.yaml').read_text())['env']
CFG=ENV['relationReward']


def scene(preset):
    graph=sample_graph(1,ENV['relationGraph'],preset=preset,generator=torch.Generator().manual_seed(8))
    objects=torch.zeros(1,4,13);objects[...,6]=1;objects[...,2]=.2
    sizes=torch.full((1,4,3),.8);sizes[...,2]=.4
    return [torch.zeros(1,2,2,3),torch.zeros(1,2,2,3),torch.zeros(1,2,3),objects,sizes,torch.zeros(1,2,3)],graph


def evaluate(s,g,cfg=CFG):
    return evaluate_interaction_edges(*s,g,cfg,.94)


def test_config_changes_progress_and_enables_task_amp_with_checkpoint_contract():
    validate_relation_config(CFG)
    validate_unified_env(ENV)
    original=deepcopy(BASE)
    original['relationReward']['progress']=CFG['progress']
    original['ampTaskConditioning']=True
    original['ampConditionedResume']=True
    original['skillDiscProb']=ENV['skillDiscProb']
    assert original==ENV
    nominal=torch.tensor(FAMILY_PROBS)@torch.tensor(EXPERT_PROBS)
    torch.testing.assert_close(nominal,torch.tensor(ENV['skillDiscProb']))
    with pytest.raises(ValueError,match='reward config differs'):
        check_checkpoint_metadata({'relation_metadata':checkpoint_metadata(BASE['relationReward'])},checkpoint_metadata(CFG))


@pytest.mark.parametrize('preset,expected',[
    ('holding',0),('sit',1),('climb',2),('holding_at',0),('holding_ontop',0)])
def test_size_progress_task_family_matches_expert_and_replay(preset,expected):
    graph=sample_graph(4,ENV['relationGraph'],preset=preset,
                       generator=torch.Generator().manual_seed(21))
    envs=torch.arange(4).repeat_interleave(2)
    agents=torch.arange(2).repeat(4)
    family=family_from_templates(classify_templates(graph,envs,agents,with_climb=True))
    assert (family==expected).all()
    # Marker motion features verify the actual source family, not just a relabel.
    steps=ENV['numAMPObsSteps']
    pool_family=torch.arange(3).repeat_interleave(3)
    pool=append_family(pool_family[:,None,None].expand(-1,steps,4).float(),
                       pool_family[:,None].expand(-1,steps)).flatten(1)
    reference=append_family(torch.zeros(8,steps,4),
                           family[:,None].expand(-1,steps)).flatten(1)
    for _ in range(2):
        matched=sample_family_matched(pool,reference,steps)
        assert torch.equal(amp_family_ids(matched,steps),family)
        assert torch.equal(matched[:,0],family.float())


@pytest.mark.parametrize('preset,rel,margin',[('sit',SIT,.1),('climb',CLIMB,.05)])
def test_interaction_progress_matches_success_rectangle(preset,rel,margin):
    s,g=scene(preset);s[2][...,0]=.5
    _,d=evaluate(s,g);mask=(g.edge_relation==rel)&g.edge_valid
    expected_error=.5-.8*(.5-margin)
    torch.testing.assert_close(d['progress'][mask],torch.full_like(d['progress'][mask],1/(1+expected_error)))
    assert (d['region_error'][mask]>0).all()
    s[2][...,0]=.1
    _,d=evaluate(s,g)
    assert (d['progress'][mask]==1).all()
    # XY progress saturation must not silently imply correct height/feet.
    success=interaction_own_success(torch.ones_like(d['progress']),d['z_error'],d['feet_height_error'],g,CFG,d['region_error'])
    assert not success[mask].any()


def test_climb_no_longer_saturates_outside_box_but_legacy_unchanged():
    s,g=scene('climb');s[2][...,0]=.5
    _,new=evaluate(s,g);_,old=evaluate(s,g,BASE['relationReward']);mask=(g.edge_relation==CLIMB)&g.edge_valid
    assert (old['progress'][mask]==1).all()
    assert (new['progress'][mask]<1).all()


def test_ontop_progress_uses_support_rectangle():
    s,g=scene('holding_ontop');mask=(g.edge_relation==ON_TOP)&g.edge_valid
    for index in mask.nonzero():
        batch,edge=index.tolist();source=int(g.edge_src[batch,edge])-2
        s[3][batch,source,0]=.5
    _,d=evaluate(s,g)
    torch.testing.assert_close(d['progress'][mask],1/(1+d['region_error'][mask]))
    assert (d['progress'][mask]<1).all()


def test_holding_tracks_box_size_while_at_does_not():
    s,g=scene('holding');s[2][...,0]=1.
    mask=(g.edge_relation==HOLDING)&g.edge_valid
    _,small=evaluate(s,g);s[4][...,:2]=1.2;_,large=evaluate(s,g)
    assert (large['progress'][mask]>small['progress'][mask]).all()
    s,g=scene('holding_at');mask=(g.edge_relation==AT)&g.edge_valid
    for batch,edge in mask.nonzero().tolist():s[3][batch,int(g.edge_src[batch,edge])-2,0]=1.
    _,small=evaluate(s,g);s[4][...,:2]=1.2;_,large=evaluate(s,g)
    torch.testing.assert_close(small['progress'][mask],large['progress'][mask])


def test_anisotropic_rectangle_yaw_and_boundary_distance():
    support=torch.tensor([[0.,0.,0.,0.,0.,.70710678,.70710678]])
    size=torch.tensor([[.8,.2,.4]])
    # After yaw90 the narrow local Y axis lies on world X.
    point=torch.tensor([[.2,0.]])
    error=inner_xy_region_error(point,support,size,.1)
    torch.testing.assert_close(error,torch.tensor([.12]),atol=1e-6,rtol=0)


def test_new_success_and_phi_remain_identical_to_legacy():
    s,g=scene('random_scenario');s[2][...,0]=.7
    a,da=evaluate(s,g);b,db=evaluate(s,g,BASE['relationReward'])
    torch.testing.assert_close(a,b)
    for key in ['region_error','z_error','feet_height_error']:
        torch.testing.assert_close(da[key],db[key])
    assert not da['progress'][~g.edge_valid].any()
