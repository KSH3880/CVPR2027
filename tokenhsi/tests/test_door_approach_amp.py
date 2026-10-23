from pathlib import Path
import pytest
import torch
import yaml
from utils.door_approach_amp import DoorApproachAMP, apply_reward_gate, validate
from utils.edge_door_spec import validate_env

ROOT=Path(__file__).resolve().parents[1]
SPEC=dict(full_distance=1.2,interaction_distance=.6,hand_distance=.25,angle_threshold=.05,spawn_distance=1.5)


def states():
    roots=torch.zeros(2,2,3);roots[...,1]=-1.5
    hands=torch.ones(2,2,2,3)*3
    return roots,hands,torch.zeros(2,2,3),torch.zeros(2,2)


def test_per_owner_latch_and_partial_reset():
    gate=DoorApproachAMP(2,2,'cpu',SPEC)
    roots,hands,handles,angles=states()
    assert torch.equal(gate.step(roots,hands,handles,angles),torch.ones(2,2))
    roots[0,0,1]=-.5
    assert gate.step(roots,hands,handles,angles).tolist()==[[0.,1.],[1.,1.]]
    roots[0,0,1]=-1.5
    assert gate.step(roots,hands,handles,angles)[0,0]==0
    angles[1,1]=.06
    gate.step(roots,hands,handles,angles)
    gate.reset(torch.tensor([0]))
    assert gate.step(roots,hands,handles,angles).tolist()==[[1.,1.],[1.,0.]]


def test_taper_and_hand_contact():
    gate=DoorApproachAMP(2,2,'cpu',SPEC)
    roots,hands,handles,angles=states()
    roots[0,0,1]=-.9
    assert gate.step(roots,hands,handles,angles)[0,0].item()==pytest.approx(.5)
    hands[0,1,0]=handles[0,1]
    weight=gate.step(roots,hands,handles,angles)
    assert weight[0,1]==0 and weight[1].eq(1).all()


def test_disabled_gate_keeps_full_amp_near_handle_contact_and_open_door():
    gate=DoorApproachAMP(2,2,'cpu',dict(SPEC,gate_enabled=False))
    roots,hands,handles,angles=states()
    for distance in (1.5,1.2,.9,.6,0.):
        roots[...,1]=-distance
        hands.copy_(handles[:,:,None].expand_as(hands))
        angles.fill_(1.2)
        weight=gate.step(roots,hands,handles,angles)
        assert torch.equal(weight,torch.ones(2,2))
        assert not gate.interacting.any()
        amp={'disc_rewards':torch.tensor([[.3],[.7],[1.],[2.]])}
        torch.testing.assert_close(apply_reward_gate(amp,weight.reshape(-1,1))['disc_rewards'],amp['disc_rewards'])
    gate.reset(torch.tensor([0]))
    assert gate.step(roots,hands,handles,angles).eq(1).all()
    with pytest.raises(ValueError,match='gate_enabled'):
        validate(dict(SPEC,gate_enabled=0))


def test_rollout_weights_preserve_history_and_do_not_renormalize_task():
    weights=torch.tensor([[[1.],[.5]],[[0.],[1.]]])
    original={'disc_rewards':torch.ones_like(weights)*2}
    result=apply_reward_gate(original,weights)
    assert result['disc_rewards'].tolist()==[[[2.],[1.]],[[0.],[2.]]]
    assert original['disc_rewards'].eq(2).all()
    task=torch.ones_like(weights)*.6
    combined=.5*task+.5*result['disc_rewards']
    assert combined[1,0].item()==pytest.approx(.3)
    with pytest.raises(ValueError):apply_reward_gate(original,weights[:1])


def test_config_unified_loco_only_and_paths():
    name='door_open_unified_approach_amp'
    env=yaml.safe_load((ROOT/f'data/cfg/multi_agent/{name}.yaml').read_text())['env']
    validate_env(env)
    assert env['skill']==['loco'] and not env.get('doorReference')
    assert env['doorApproachAMP']['gate_enabled'] is False
    assert env['stateInit']=='Start'
    train=yaml.safe_load((ROOT/f'data/cfg/train/rlg/{name}.yaml').read_text())
    assert not train['params']['network']['transformer']['share_edge_encoder']
    baseline=yaml.safe_load((ROOT/'data/cfg/train/rlg/door_open_unified_amp_on.yaml').read_text())
    assert train['params']['network']==baseline['params']['network']
    manifest=yaml.safe_load((ROOT/'data/dataset_door_loco.yaml').read_text())
    assert set(manifest['motions'])=={'loco'}
    for motion in manifest['motions']['loco']:
        assert (ROOT/'data'/motion['file']).is_file()
    for suffix in ['train','test','vnc']:
        assert (ROOT/f'scripts/multi_agent/{name}_{suffix}.sh').is_file()


def test_push_holding_matched_scales_and_old_checkpoint_rejected():
    from copy import deepcopy
    from env.tasks.multi_agent.edge_door_reward import door_reward
    from utils.edge_door_spec import compile_graph
    from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata
    env=yaml.safe_load((ROOT/'data/cfg/multi_agent/door_open_unified_approach_amp.yaml').read_text())['env']
    cfg=env['relationReward']
    validate_env(env)
    assert cfg['state_reward_weight']==.4
    graph=compile_graph(env['relationGraph'],2,2)
    handles=torch.tensor([[[0.,0.,1.],[0.,5.,1.]]])
    roots=handles.clone(); roots[...,1]-=1.5
    hands=handles[:,:,None].repeat(1,1,2,1)
    angles=torch.zeros(1,2)
    actual=door_reward(roots,hands,handles,angles,graph,cfg)
    torch.testing.assert_close(actual['approach_component'],torch.full((1,2),.1))
    torch.testing.assert_close(actual['interaction_component'],torch.full((1,2),.2))
    old=deepcopy(cfg); old['state_reward_weight']=.2
    previous=door_reward(roots,hands,handles,angles,graph,old)
    for key in ('approach_component','interaction_component'):
        torch.testing.assert_close(actual[key],2*previous[key])
    for key in ('progress_component','success_component'):
        torch.testing.assert_close(actual[key],previous[key])
    opened=door_reward(roots,hands,handles,angles+1.2,graph,cfg)
    torch.testing.assert_close(opened['total'],torch.full((1,2),.8))
    with pytest.raises(ValueError,match='reward config differs'):
        check_checkpoint_metadata({'relation_metadata':checkpoint_metadata(old)},checkpoint_metadata(cfg))


def test_full_amp_shared_comparison_same_env_and_ppo():
    from copy import deepcopy
    names=['door_open_unified_approach_amp','door_open_shared_edge_approach_amp']
    envs=[yaml.safe_load((ROOT/f'data/cfg/multi_agent/{name}.yaml').read_text())['env'] for name in names]
    assert envs[0]==envs[1]
    for env in envs:validate_env(env)
    trains=[yaml.safe_load((ROOT/f'data/cfg/train/rlg/{name}.yaml').read_text()) for name in names]
    assert not trains[0]['params']['network']['transformer']['share_edge_encoder']
    assert trains[1]['params']['network']['transformer']['share_edge_encoder']
    expected=deepcopy(trains[0])
    expected['params']['network']['transformer']['share_edge_encoder']=True
    expected['params']['config']['name']=trains[1]['params']['config']['name']
    assert expected==trains[1]
    for suffix in ('train','test','vnc'):
        wrapper=ROOT/f'scripts/multi_agent/{names[1]}_{suffix}.sh'
        assert wrapper.is_file()
        assert names[1] in wrapper.read_text()
        assert 'dataset_door_proxy' not in wrapper.read_text()


@pytest.mark.parametrize('field,value',[('interaction_distance',1.3),('spawn_distance',1.),('hand_distance',float('nan'))])
def test_invalid_thresholds(field,value):
    with pytest.raises(ValueError):validate(dict(SPEC,**{field:value}))


def test_door_loco_reset_removes_arena_and_clip_translation(monkeypatch):
    from types import SimpleNamespace
    from env.tasks.multi_agent.humanoid_ma_door import HumanoidMADoor
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    from utils.door_reference import yaw_from_quat
    roots = torch.zeros(3, 2, 13)
    roots[..., 6] = 1.
    roots[..., 0] = torch.tensor([0., 100., 200.])[:, None] + 7.
    roots[..., 2] = .9
    roots[..., 7] = 1.
    bodies = roots[:, :, None, :].repeat(1, 1, 2, 1)
    bodies[:, :, 1, 0] += .2
    state = SimpleNamespace(_door_reference_bank=None, _is_eval=False,
        _door_approach_spec=SPEC, _humanoid_root_states=roots,
        _kinematic_humanoid_rigid_body_states=bodies,
        _agent_spawn_offsets=torch.tensor([[1.78, -4.1, 0.], [1.78, .9, 0.]]),
        _dof_pos=torch.ones(3, 2, 2), _dof_vel=torch.zeros(3, 2, 2))
    state._set_env_state = lambda *args: HumanoidMACarry._set_env_state(state, *args)
    monkeypatch.setattr(HumanoidMACarry, '_reset_ref_state_init', lambda self, e, a: None)
    # Use an actual subclass instance for super() without building a simulator.
    task = HumanoidMADoor.__new__(HumanoidMADoor)
    task.__dict__.update(state.__dict__)
    task._set_env_state = lambda *args: HumanoidMACarry._set_env_state(task, *args)
    env = torch.arange(3).repeat_interleave(2)
    agent = torch.arange(2).repeat(3)
    task._reset_ref_state_init(env, agent)
    expected = state._agent_spawn_offsets[:, :2].expand(3, -1, -1)
    torch.testing.assert_close(roots[..., :2], expected)
    torch.testing.assert_close(yaw_from_quat(roots[..., 3:7]), torch.full((3, 2), torch.pi / 2))
    torch.testing.assert_close(roots[..., 7:10], torch.tensor([0., 1., 0.]).expand(3, 2, 3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(bodies[:, :, 1, :3] - bodies[:, :, 0, :3],
                               torch.tensor([0., .2, 0.]).expand(3, 2, 3), atol=1e-5, rtol=1e-5)


def test_door_gpu_reset_batches_humans_and_doors():
    from types import SimpleNamespace
    from env.tasks.multi_agent.humanoid_ma_door import HumanoidMADoor
    calls = {'root': [], 'dof': []}
    gym = SimpleNamespace(
        set_actor_root_state_tensor_indexed=lambda sim, tensor, ids, n: calls['root'].append(n),
        set_dof_state_tensor_indexed=lambda sim, tensor, ids, n: calls['dof'].append(n))
    task = SimpleNamespace(gym=gym, sim=None, _enable_markers=False, _is_eval=False,
        _humanoid_actor_ids=torch.tensor([[0, 1], [4, 5]], dtype=torch.int32),
        _box_actor_ids=torch.tensor([[2, 3], [6, 7]], dtype=torch.int32),
        _root_states=torch.zeros(8, 13), _dof_state=torch.zeros(8, 2),
        progress_buf=torch.ones(2), reset_buf=torch.ones(2), _terminate_buf=torch.ones(4))
    task._flat_slot_ids=lambda ids: (ids[:, None] * 2 + torch.arange(2)).flatten()
    HumanoidMADoor._reset_env_tensors(task, torch.tensor([1]))
    assert calls == {'root': [4], 'dof': [4]}
    assert task.progress_buf.tolist() == [1., 0.]
    assert task._terminate_buf.tolist() == [1., 1., 0., 0.]


def test_pull_config_axis_and_checkpoint_contract():
    import xml.etree.ElementTree as ET
    from types import SimpleNamespace
    from env.tasks.multi_agent.humanoid_ma_door import HumanoidMADoor
    env_path=ROOT/'data/cfg/multi_agent/door_pull_unified_approach_amp.yaml'
    cfg=yaml.safe_load(env_path.read_text())
    validate_env(cfg['env'])
    cfg['env']['motion_file']='tokenhsi/data/dataset_door_loco.yaml'
    cfg['args']=SimpleNamespace(cfg_train=str(ROOT/'data/cfg/train/rlg/door_pull_unified_approach_amp.yaml'))
    task=object.__new__(HumanoidMADoor);task.cfg=cfg
    task._door_cfg=cfg['env']['door'];task._door_reference_spec=None
    task._door_approach_spec=cfg['env']['doorApproachAMP']
    pull=task.get_door_experiment_contract()
    assert pull['opening_direction']=='pull'
    task._door_cfg={'resetAngleRange':[0.,0.]}
    assert pull!=task.get_door_experiment_contract()
    asset=ET.parse(ROOT/'assets/door/door_pull.urdf').getroot()
    assert asset.find("joint[@name='hinge']/axis").attrib['xyz']=='0 0 -1'
    assert tuple(map(float,asset.find("joint[@name='handle_mount']/origin").attrib['xyz'].split()))==(.78,-.25,1.)
    cfg['env']['door']['openingDirection']='invalid'
    with pytest.raises(ValueError):validate_env(cfg['env'])


@pytest.mark.parametrize('hinge', [{'damping': -1., 'friction': .5}, {'damping': float('nan'), 'friction': .5}, {'damping': True, 'friction': .5}, {'damping': 5.}])
def test_invalid_hinge_physics_rejected(hinge):
    env = yaml.safe_load((ROOT/'data/cfg/multi_agent/door_open_unified_approach_amp.yaml').read_text())['env']
    env['door']['hinge'] = hinge
    with pytest.raises(ValueError, match='hinge'):
        validate_env(env)


def test_hinge_physics_changes_checkpoint_contract():
    from types import SimpleNamespace
    from env.tasks.multi_agent.humanoid_ma_door import HumanoidMADoor
    cfg = yaml.safe_load((ROOT/'data/cfg/multi_agent/door_open_unified_approach_amp.yaml').read_text())
    cfg['env']['motion_file'] = 'tokenhsi/data/dataset_door_loco.yaml'
    cfg['args'] = SimpleNamespace(cfg_train=str(ROOT/'data/cfg/train/rlg/door_open_unified_approach_amp.yaml'))
    task = object.__new__(HumanoidMADoor)
    task.cfg = cfg
    task._door_cfg = cfg['env']['door']
    task._door_reference_spec = None
    task._door_approach_spec = cfg['env']['doorApproachAMP']
    current = task.get_door_experiment_contract()
    assert current['hinge_physics'] == {'damping': 2., 'friction': .1}
    del task._door_cfg['hinge']
    assert current != task.get_door_experiment_contract()
