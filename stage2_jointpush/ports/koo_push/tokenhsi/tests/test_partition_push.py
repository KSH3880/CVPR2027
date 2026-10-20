from pathlib import Path
import torch
import yaml
import pytest
from utils.partition_push import strip_geometry, nearest_boxes, partition_reward, group_tokens, group_layout, sample_groups, sample_asset_bank, sample_surround_spawn, exclusive_regions
from utils.open_push_spec import compile_graph, validate_env
from utils.edge_stage1_spec import semantic_graph_packet
from utils.edge_ontop_spec import expand_graph
from learning.multi_agent.push_at_coordination import PushAtCoordination


def test_profile_and_eight_token_graph():
    env = yaml.safe_load((Path(__file__).resolve().parents[1]/'data/cfg/multi_agent/partition_push_ca4.yaml').read_text())['env']
    validate_env(env)
    assert env['numObjects'] == 8
    assert env['partitionPush']['physical_boxes'] == 8  # initial ordinary bank
    graph = expand_graph(compile_graph(env['relationGraph'], 4, 8), 2)
    packet = semantic_graph_packet(graph, 2)
    assert packet.shape == (2, 40)
    graph.edge_dst[:, ::2] = torch.arange(4)+4
    graph.edge_src[:, 1::2] = torch.arange(4)+4
    bundles = PushAtCoordination().bundles(semantic_graph_packet(graph, 2), 4)
    assert bundles[..., 2].tolist() == [[4,5,6,7]]*2
    assert bundles[..., 3].eq(12).all()


def test_four_strips_cover_one_box_and_follow_angular_velocity():
    box = torch.zeros(1, 1, 13); box[..., 6] = 1; box[..., 2] = .75
    size = torch.tensor([[[5.2, 2.2, 1.5]]])
    box[..., 12] = 2
    states, sizes = strip_geometry(box, size)
    assert states.shape == (1, 4, 13)
    torch.testing.assert_close(sizes[..., 0], torch.full((1,4), 1.3))
    torch.testing.assert_close(states[..., 8], 2*states[..., 0])
    torch.testing.assert_close(states[:,0,0]-sizes[:,0,0]/2, torch.tensor([-2.6]))
    torch.testing.assert_close(states[:,-1,0]+sizes[:,-1,0]/2, torch.tensor([2.6]))
    roots = torch.zeros(1,4,13); roots[...,0] = states[...,0]; roots[...,1] = -3
    assert nearest_boxes(roots, states).tolist() == [[0,1,2,3]]


def test_contact_height_and_group_gate():
    boxes = torch.zeros(1,9,13); boxes[...,6]=1; boxes[:,8,2]=.6
    sizes = torch.full((1,9,3),1.3); sizes[:,8]=torch.tensor([4.,2.,1.2])
    assignment = torch.full((1,4),8,dtype=torch.long)
    roots = torch.zeros(1,4,13)
    local = torch.zeros(1,4,2,3); local[...,1]=-1
    local[...,0] = torch.tensor([-1.5,-.5,.5,1.5])[None,:,None]
    hands = local+boxes[:,8,None,None,:3]
    goals = boxes[:,8,None,:3].expand(1,4,3).clone()
    spec=dict(state_reward_weight=.2,progress_reward_weight=.2,success_reward_weight=.2,
        satisfaction_threshold=.9,hand_distance_scale=10.,at_distance_scale=10.,at_z_tolerance=.001,approach_distance=.8)
    result = partition_reward(roots,hands,boxes,goals,sizes,assignment,local,spec)
    assert result['done'].all()
    torch.testing.assert_close(result['targets'][...,2],torch.full((1,4,2),.6))
    hands[:,3,:,2]+=1
    result = partition_reward(roots,hands,boxes,goals,sizes,assignment,local,spec)
    assert not result['done'].any()


@pytest.mark.parametrize('leaders', [(0,0,0,0),(0,0,2,2),(0,0,0,3),(0,1,2,3),(0,1,1,1)])
@pytest.mark.parametrize('spawn_mode', [None, 'surround_independent'])
def test_reset_replaces_four_ordinary_boxes_and_preserves_other_env(leaders, spawn_mode):
    """Exercise actual task reset without importing Isaac Gym or creating a sim."""
    import ast
    from types import SimpleNamespace
    from utils.door_reference import rotate
    path = Path(__file__).resolve().parents[1]/'env/tasks/multi_agent/humanoid_ma_partition_push.py'
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in
               ('_reset_envs', '_select_regions', '_logical_box_values')]
    namespace = dict(torch=torch, rotate=rotate, nearest_boxes=nearest_boxes, group_tokens=group_tokens, group_layout=group_layout,
                     sample_groups=lambda n,*args: torch.tensor(leaders)[None].expand(n,-1),
                     sample_surround_spawn=sample_surround_spawn, exclusive_regions=exclusive_regions)
    for method in methods:
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    class Fake:
        pass
    for name in ('_reset_envs', '_select_regions', '_logical_box_values'):
        setattr(Fake, name, namespace[name])
    task = Fake(); task.num_envs = 3; task.device = 'cpu'
    task._joint_push_spec = dict(spawn_distance=[1.,2.], goal_distance=[2.2,5.],coupled_probability=.8)
    if spawn_mode is not None:
        task._joint_push_spec['spawn_mode'] = spawn_mode
    task._initial_box_states = torch.zeros(3,24,13); task._initial_box_states[...,6] = 1
    task._box_states = task._initial_box_states.clone()
    task._box_size, task._payload_masses = sample_asset_bank(3,dict(size_min=[1.,1.8,1.2],size_max=[1.3,2.2,1.5],mass_per_agent=[20.,30.]))
    task._group_leaders = torch.zeros(3,4,dtype=torch.long)
    task._group_counts = torch.zeros(3,4,dtype=torch.long)
    task._box_bps = torch.zeros(3,24,8,3)
    task._ordinary_keep = torch.arange(4)[None].expand(3,-1).clone()
    task._local_hands = torch.zeros(3,4,2,3)
    task._humanoid_root_states = torch.zeros(3,4,13)
    task._kinematic_humanoid_rigid_body_states = torch.zeros(3,4,1,13)
    task._agent_box_assignment = torch.zeros(3,4,dtype=torch.long)
    task._agent_region_assignment = torch.arange(4)[None].expand(3,-1).clone()
    task._tar_pos = torch.zeros(3,4,3)
    task._curr_amp_obs_buf = torch.zeros(3,4,5)
    task._hist_amp_obs_buf = torch.zeros(3,4,9,5)
    spec=dict(sampler='two_agent_open_push',semantic_only=True,task_probabilities={'OPEN':0.,'PUSH':1.})
    task.relation_runtime = SimpleNamespace(graph=expand_graph(compile_graph(spec,4,8),3),done=torch.ones(3,4,dtype=torch.bool))
    def actors(ids):
        task._humanoid_root_states[ids,:,:2] = task._spawn_roots[ids]
        task._kinematic_humanoid_rigid_body_states[ids,:,0] = task._humanoid_root_states[ids]
    task._reset_actors=actors
    task._reset_env_tensors=lambda ids: None
    task._compute_observations=lambda ids: None
    task._compute_amp_observations=lambda ids: None
    untouched = task._box_states[1].clone()
    task._reset_envs(torch.tensor([0,2]))
    torch.testing.assert_close(task._box_states[1], untouched)
    expected,counts,_ = group_layout(torch.tensor(leaders)[None].expand(2,-1))
    assert torch.equal(task._agent_box_assignment[[0,2]],expected)
    assert torch.equal(task._group_counts[[0,2]],counts)
    regions = task._agent_region_assignment[[0,2]]
    assert regions.sort(-1).values.eq(torch.arange(4)).all()
    assert torch.equal(expected.gather(1, regions), expected)
    assert torch.equal(task.relation_runtime.graph.edge_dst[[0,2],::2], regions+4)
    assert torch.equal(task.relation_runtime.graph.edge_src[[0,2],1::2], regions+4)
    # Crossing over another region cannot steal its ownership within the episode.
    fixed_regions = task._agent_region_assignment.clone()
    fixed_hands = task._local_hands.clone()
    task._humanoid_root_states[0,:,0] += torch.tensor([20.,-20.,20.,-20.])
    task._select_regions(torch.tensor([0]), live=True)
    assert torch.equal(task._agent_region_assignment, fixed_regions)
    torch.testing.assert_close(task._local_hands, fixed_hands)
    active = task._box_states[[0,2],:,0].abs()<100
    assert active.sum(-1).eq(len(set(leaders))+4).all()
    assert (task._box_states[[0,2],:8,0] > 900).sum(-1).eq(4).all()
    logical = task._logical_box_values(task._box_states, torch.tensor([0,2]))
    assert logical.shape == (2,8,13)
    assert (logical[...,0].abs()<100).all()  # parked boxes cannot leak into obs
    sizes = task._logical_box_values(task._box_size, torch.tensor([0,2]))
    assert (sizes[:,:4,0]>=1).all() and (sizes[:,:4,0]<=1.3).all()
    assert torch.equal(task.relation_runtime.graph.edge_dst[[0,2],1::2],12+torch.tensor(leaders)[None].expand(2,-1))
    # A second reset on just one scene cannot alter other scenes' assignments.
    untouched_assignment = task._agent_box_assignment[2].clone()
    untouched_regions = task._agent_region_assignment[2].clone()
    task._reset_envs(torch.tensor([0]))
    assert torch.equal(task._agent_box_assignment[2],untouched_assignment)
    assert torch.equal(task._agent_region_assignment[2],untouched_regions)


def test_exclusive_regions_resolve_shared_nearest_and_global_conflict():
    cost = torch.tensor([[[1.,2.,100.,100.], [1.1,100.,100.,100.],
                          [100.,100.,0.,100.], [100.,100.,100.,0.]]])
    assert cost.argmin(-1).tolist() == [[0,0,2,3]]
    assert exclusive_regions(cost,torch.ones_like(cost,dtype=torch.bool)).tolist()==[[1,0,2,3]]
    tied = torch.tensor([[[0.,1.,2.,3.]]]).expand(32,4,4)
    result = exclusive_regions(tied,torch.ones_like(tied,dtype=torch.bool))
    assert result.sort(-1).values.eq(torch.arange(4)).all()


@pytest.mark.parametrize('groups', [(0,0,0,0),(0,0,2,2),(0,0,0,3),(0,1,2,3),(0,1,1,1),(0,1,0,1)])
def test_exclusive_regions_cannot_take_another_groups_region(groups):
    owners=torch.tensor(groups)[None].expand(128,-1)
    same=owners[...,None]==owners[:,None,:]
    distances=torch.rand(128,4,4)
    result=exclusive_regions(distances,same)
    assert result.sort(-1).values.eq(torch.arange(4)).all()
    assert torch.equal(owners.gather(1,result),owners)


def test_exclusive_regions_reject_impossible_group_matching():
    with pytest.raises(ValueError,match='No valid exclusive'):
        exclusive_regions(torch.ones(1,4,4),torch.zeros(1,4,4,dtype=torch.bool))


def test_region_checkpoint_metadata_accepts_legacy_and_rejects_unknown_semantics():
    import ast
    import copy
    from types import SimpleNamespace
    path=Path(__file__).resolve().parents[1]/'env/tasks/multi_agent/humanoid_ma_partition_push.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef))
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='validate_inference_checkpoint')
    ns={}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
    contract=dict(policy='partition_push_ca4',partition=dict(version=2),friction=.6)
    task=SimpleNamespace(cfg={'args':SimpleNamespace(test=True),'env':{}},
        REGION_ASSIGNMENT='exclusive_reset_min_distance_v1',get_open_push_contract=lambda:copy.deepcopy(contract))
    validate=ns['validate_inference_checkpoint']
    validate(task,{'open_push_contract':contract})
    validate(task,{'open_push_contract':contract,'partition_region_assignment':task.REGION_ASSIGNMENT})
    with pytest.raises(ValueError,match='region assignment differs'):
        validate(task,{'open_push_contract':contract,'partition_region_assignment':'unknown'})
    with pytest.raises(ValueError,match='configuration differs'):
        validate(task,{'open_push_contract':dict(contract,friction=.1)})


def test_surround_spawn_clearance_separation_and_independent_heading():
    torch.manual_seed(513)
    boxes = torch.zeros(10000,4,13)
    boxes[..., :2] = torch.tensor([4.,-7.])
    sizes = torch.tensor([5.2,2.2,1.5]).expand(10000,4,3)
    positions, heading = sample_surround_spawn(boxes,sizes,[1.,2.],1.2)
    delta = positions-boxes[...,:2]
    clearance = delta.norm(dim=-1)-sizes[...,:2].norm(dim=-1)/2
    assert ((clearance>=.99999)&(clearance<=2.00001)).all()
    distances = torch.cdist(positions,positions)
    distances[:,torch.arange(4),torch.arange(4)] = float('inf')
    assert distances.min()>=1.19999
    angle = torch.atan2(delta[...,1],delta[...,0])
    for values in (angle,heading,heading-angle):
        # Reject half-circle/fixed/box-facing heading regressions.
        assert values.cos().mean().abs()<.025
        assert values.sin().mean().abs()<.025
    quadrants = ((angle+torch.pi)/(torch.pi/2)).long().clamp_max(3)
    assert torch.bincount(quadrants.flatten(),minlength=4).min()>9000


def test_different_strips_are_coupled_in_coordination():
    from torch import nn
    class Capture(nn.Module):
        def forward(self, humans, tokens, relations, valid):
            self.relations = relations
            return humans
    model = PushAtCoordination(); model.partition_count = 4
    capture = Capture(); model.blocks = nn.ModuleList([capture])
    bundles = torch.tensor([[[1,0,4,12],[1,1,5,12],[1,2,6,12],[1,3,7,12]]])
    tokens = torch.zeros(1,4,64); valid = torch.ones(1,4,dtype=torch.bool)
    model.task_tokens = lambda *args: (tokens, valid, bundles)
    model(tokens, tokens, None, None, None)
    assert capture.relations.eq(1).sum(-1).eq(1).all()
    assert capture.relations.eq(2).sum(-1).eq(3).all()
    model.partition_count = 1
    model(tokens, tokens, None, None, None)
    assert not capture.relations.eq(2).any()


def test_group_sampler_probabilities_and_fixed_population():
    torch.manual_seed(93)
    groups = sample_groups(40000,.8)
    same = groups[...,None] == groups[:,None,:]
    counts = same.sum(-1)
    assert counts.shape == (40000,4)
    assert abs(float(counts.eq(4).all(-1).float().mean())-.8/3)<.012
    assert abs(float(counts.eq(1).all(-1).float().mean())-.2**3)<.003
    assert group_layout(sample_groups(100,0.))[1].eq(1).all()
    assert group_layout(sample_groups(100,.8,preset='all_shared'))[1].eq(4).all()
    partitions = {tuple(sorted(row.tolist())) for row in counts}
    assert partitions == {(4,4,4,4),(1,3,3,3),(1,1,1,1),(1,1,2,2),(2,2,2,2)}


def test_asset_bank_per_agent_dimensions_and_random_distractors():
    sizes,masses = sample_asset_bank(128,dict(size_min=[1.,1.8,1.2],size_max=[1.3,2.2,1.5],mass_per_agent=[20.,30.]))
    members = torch.cat((torch.ones(8),torch.arange(1,5).repeat(4)))
    widths = sizes[...,0]/members
    mass_per_person = masses/members
    assert (widths>=1).all() and (widths<=1.3).all()
    assert (mass_per_person>=20).all() and (mass_per_person<=30).all()
    assert (sizes[...,1]>=1.8).all() and (sizes[...,1]<=2.2).all()
    assert (sizes[...,2]>=1.2).all() and (sizes[...,2]<=1.5).all()
    assert sizes[:,:8,0].std()>.05


def test_coordination_separates_two_groups_with_different_strip_tokens():
    from torch import nn
    class Capture(nn.Module):
        def forward(self,humans,tokens,relations,valid):
            self.relations=relations
            return humans
    model=PushAtCoordination();model.group_by_goal=True
    capture=Capture();model.blocks=nn.ModuleList([capture])
    bundles=torch.tensor([[[1,0,4,12],[1,1,5,12],[1,2,6,14],[1,3,7,14]]])
    nodes=torch.zeros(1,4,64);valid=torch.ones(1,4,dtype=torch.bool)
    model.task_tokens=lambda *args:(nodes,valid,bundles)
    model(nodes,nodes,None,None,None)
    assert capture.relations.eq(1).sum(-1).eq(1).all()
    assert capture.relations.eq(2).sum(-1).eq(1).all()
    assert capture.relations.eq(0).sum(-1).eq(2).all()


def test_reward_contact_gate_does_not_cross_groups():
    boxes=torch.zeros(1,24,13);boxes[...,6]=1;boxes[...,2]=.65
    sizes=torch.full((1,24,3),1.3)
    assignment=torch.tensor([[9,9,17,17]])
    roots=torch.zeros(1,4,13)
    local=torch.zeros(1,4,2,3);local[...,1]=-.65
    goals=boxes[:,9,None,:3].expand(1,4,3).clone()
    hands=local+goals[...,None,:]
    hands[:,3,:,2]+=1
    spec=dict(state_reward_weight=.2,progress_reward_weight=.2,success_reward_weight=.2,
        satisfaction_threshold=.9,hand_distance_scale=10.,at_distance_scale=10.,at_z_tolerance=.001,approach_distance=.8)
    result=partition_reward(roots,hands,boxes,goals,sizes,assignment,local,spec)
    assert result['done'].tolist()==[[True,True,False,False]]


def test_ablation_uses_per_person_size_and_mass(tmp_path,monkeypatch):
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/multi_agent/partition_push_setup.py'
    module_spec=importlib.util.spec_from_file_location('partition_push_setup',path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    env=yaml.safe_load((Path(__file__).resolve().parents[1]/'data/cfg/multi_agent/partition_push_ca4.yaml').read_text())['env']
    checkpoint=tmp_path/'source.pth';output=tmp_path/'env.yaml'
    torch.save(dict(relation_experiment_config=dict(env=env),open_push_contract=dict(policy='partition_push_ca4')),checkpoint)
    monkeypatch.setattr('sys.argv',['setup','config',str(checkpoint),str(output),'--size','1.2','2','1.35','--mass','25'])
    module.main()
    restored=yaml.safe_load(output.read_text())['env']['partitionPush']
    assert restored['size_min']==restored['size_max']==[1.2,2.,1.35]
    assert restored['mass_per_agent']==[25.,25.]
    assert restored['coupled_probability']==.8
