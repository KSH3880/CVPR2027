"""PUSH integration: graph identity, physical sampling, AMP and checkpoint contracts."""
from copy import deepcopy
from pathlib import Path
import hashlib,json
import pytest
import torch
from utils.experiments import native_config, load_environment, validate_experiment
from utils.stage1_jointpush import compose_graph, sample_assets, box_densities, contract
from utils.task_role_spec import task_graph_packet
from utils.edge_scenario_spec import classify_templates, agent_object_indices, paired_placement_success
from utils.jointpush_scene import sample_layout, goal_angles_valid, paths_clear
from utils.evaluation_protocol import spatial_success, configure
from utils.unified_training import append_family, preserve_amp_labels, sample_family_matched, validate_unified_env


def env():return native_config('stage1_push')['env']


@pytest.mark.parametrize('agents,objects',[(1,1),(1,2),(2,4)])
def test_push_packet_binding_and_required_goal(agents,objects):
    templates=torch.full((3,agents),5)
    graph=compose_graph(templates,True,objects)
    packet=task_graph_packet(graph,3).reshape(3,agents,5).long()
    owner=torch.arange(agents).expand(3,-1)
    assert (packet[...,0]==1).all() and (packet[...,1]==4).all()
    assert packet[...,2].equal(owner)
    assert packet[...,3].equal(owner+agents)
    assert packet[...,4].equal(owner+agents+objects)
    rows=torch.arange(3).repeat_interleave(agents);slots=owner.flatten()
    assert (classify_templates(graph,rows,slots)==5).all()
    assert agent_object_indices(graph,rows,slots).equal(slots)
    at=(graph.edge_relation==7)&graph.edge_valid
    assert graph.required_goal.equal(at)
    paired=paired_placement_success(at,graph)
    assert paired.equal((graph.edge_relation==12)&graph.edge_valid)


def test_native_prior_matches_requested_task_mix():
    cfg=env()
    expected=[0.,.1,.15,.25,.2,.3]
    assert list(cfg['relationGraph']['template_probabilities'].values())==expected
    assert sum(expected)==1.0
    torch.manual_seed(18)
    _,weights=sample_assets(40000,cfg,'cpu')
    actual=torch.multinomial(weights.flatten(0,1),1).flatten().bincount(minlength=6)/80000.
    torch.testing.assert_close(actual,torch.tensor(expected),rtol=0,atol=.006)
    assert cfg['episodeLength']==900
    assert 'at_progress_min_delta' not in cfg['jointPush']


@pytest.mark.parametrize('agents,objects',[(1,1),(2,4)])
def test_push_density_and_cone_follow_local_box_axis(agents,objects):
    cfg=env();cfg.update(numAgents=agents,numObjects=objects)
    sizes,weights=sample_assets(128,cfg,'cpu','push')
    assert (weights[...,5]==1).all()
    density=box_densities(sizes,weights[...,5].bool(),cfg['jointPush']['box'])
    assert (density[:,:agents]==3).all() and (density[:,agents:]==100).all()
    mass=density[:,:agents]*sizes[:,:agents].prod(-1)
    assert mass.min()>=11.135 and mass.max()<=24.685
    boxes=torch.zeros(128,objects,13);boxes[...,6]=1
    roots=torch.zeros(128,agents,3)
    if agents==2:roots[:,0,0]=-2.5;roots[:,1,0]=2.5
    push=torch.ones(128,agents,dtype=torch.bool)
    sampled,goals=sample_layout(roots,boxes,sizes,push,cfg['jointPush']['scene'])
    assert sampled.shape==boxes.shape and goals.shape==(128,agents,3)
    assert goal_angles_valid(sampled,goals,push,cfg['jointPush']['scene']).all()
    assert paths_clear(sampled,sizes,goals,roots,push,cfg['jointPush']['scene']).all()
    for distance in ((sampled[:,:agents,:2]-roots[...,:2]).norm(dim=-1),(sampled[:,:agents,:2]-goals[...,:2]).norm(dim=-1)):
        assert (distance>=2-1e-5).all() and (distance<=10+1e-5).all()


def test_push_formal_geometry_is_at_not_ontop():
    graph=compose_graph(torch.tensor([[5]]),num_objects=1)
    packet=task_graph_packet(graph,1).reshape(1,1,5)
    roots=torch.zeros(1,1,3)
    boxes=torch.tensor([[[0.,0.,.75,0.,0.,0.,1.]]])
    sizes=torch.tensor([[[1.8,1.8,1.5]]]);goals=boxes[...,:3].clone()
    assert spatial_success(packet,roots,boxes,sizes,goals).all()
    boxes[...,0]=.19
    assert spatial_success(packet,roots,boxes,sizes,goals).all()
    boxes[...,0]=.21
    assert not spatial_success(packet,roots,boxes,sizes,goals).any()


def test_four_family_amp_labels_survive_normalization_and_replay():
    labels=torch.tensor([0,1,2,3])[:,None].expand(-1,10)
    raw=append_family(torch.randn(4,10,129),labels,4).flatten(1)
    normalized=preserve_amp_labels(raw,raw*2+7,10,4)
    assert normalized.shape==(4,1330)
    assert torch.equal(normalized.reshape(4,10,133)[...,-4:],raw.reshape(4,10,133)[...,-4:])
    result=sample_family_matched(raw.flip(0),raw,10,families=4)
    assert torch.equal(result,raw)


def test_downloaded_checkpoint_snapshot_is_authoritative_and_cannot_be_ordinary_stage1():
    path=Path('checkpoints/Stage1PushFinetune_00021000.pth')
    if not path.exists():pytest.skip('Local downloaded checkpoint required')
    cfg,source=load_environment('stage1_push','',str(path),saved=True)
    assert cfg['env']['relationGraph']['template_probabilities']['PUSH_AT']==.5
    assert cfg['env']['numAgents']==2
    validate_experiment('stage1_push',cfg['env'],native_config('stage1_push',training=True),evaluation=True)
    with pytest.raises(ValueError,match='PUSH'):
        validate_experiment('stage1',cfg['env'],native_config('stage1',training=True),evaluation=True)
    cfg['env']['episodeLength']=900
    assert contract(cfg['env']) != contract(env())


def test_all_push_motions_and_bank_match_saved_hashes():
    bank=json.loads(Path('tokenhsi/data/stage1_jointpush_rsi_bank.json').read_text())
    import yaml
    manifest=yaml.safe_load(Path('tokenhsi/data/dataset_stage1_jointpush.yaml').read_text())
    hashes=[hashlib.sha256((Path('tokenhsi/data')/r['file']).read_bytes()).hexdigest() for r in manifest['motions']['push']]
    assert hashes==bank['contract']['motions']
    assert len(bank['candidates']['push'])==949
    assert all(r['motion_id']!=7 for r in bank['candidates']['push'])


def test_timeout_boundary_is_exactly_nine_hundred_steps():
    # Load the pure reset function without importing Isaac Gym for CPU coverage.
    import ast
    path=Path('tokenhsi/env/tasks/multi_agent/humanoid_ma.py')
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compute_humanoid_reset')
    node.decorator_list=[]
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    reset,terminated=namespace['compute_humanoid_reset'](
        torch.zeros(3,dtype=torch.long),torch.tensor([898,899,900]),torch.zeros(3,2,3),
        torch.tensor([0]),torch.ones(3,2,3),900.,True,torch.full((2,),.15))
    assert reset.tolist()==[0,0,1] and terminated.tolist()==[0,0,0]


def test_real_78k_warm_start_preserves_four_task_policy_outputs():
    from torch import nn
    from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
    from learning.multi_agent.stage1_push_network import Stage1PushBuilder, transfer
    from stage1_test_utils import observations, all_pairs
    path=Path('checkpoints/Stage1_00078000.pth')
    if not path.exists():pytest.skip('Local original78k checkpoint required')
    source=torch.load(path,map_location='cpu',weights_only=False)
    train=native_config('stage1_push',training=True)['params']['network']
    graph=native_config('stage1')['env']['relationGraph']
    models=[]
    for builder,amp_width in ((AMPMultiAgentBuilder(),1320),(Stage1PushBuilder(),1330)):
        builder.load(train)
        model=nn.Module()
        model.a2c_network=builder.build('amp',actions_num=32,input_shape=(634,),amp_input_shape=(amp_width,),
            value_size=1,num_agents=2,num_objects=4,observation_mode='clean_scene',scene_entity_sizes=[223,30,1],
            scene_kinematic_size=7,scene_arena_scale=5.,relation_reward_mode=env()['relationReward']['mode'],
            relation_graph_spec=graph,device='cpu')
        model.eval()
        for encoder in (model.a2c_network.actor_encoder,model.a2c_network.critic_encoder):encoder.gta_diagnostics_first_forward=False
        models.append(model)
    old,new=models;old.load_state_dict(source['model'],strict=True)
    stats,report=transfer(new,source)
    assert len(report['expanded'])==3 and report['source_epoch']==78000
    assert stats['running_mean'][-1]==0 and stats['running_var'][-1]==1
    obs=observations(all_pairs())
    with torch.no_grad():
        for method in ('eval_actor','eval_critic'):
            a,b=[getattr(m.a2c_network,method)(obs) for m in models]
            if isinstance(a,tuple):a,b=a[0],b[0]
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
