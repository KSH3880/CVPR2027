import isaacgym
from copy import deepcopy
import pytest
import torch
from utils.warehouse_waypoint import advance_phase
from utils.experiments import (
    SENSOR_WAREHOUSE_WAYPOINT_BOXONLY, SENSOR_WAREHOUSE_WAYPOINT_ONTOP, native_config, validate_experiment,
)
from learning.multi_agent.warehouse_waypoint_network import WarehouseWaypointBuilder
from learning.multi_agent.sensor_stage2_network import SensorStage2Builder
from test_sensor_stage2 import build
from test_warehouse_before_cache import make_task
from env.tasks.multi_agent.warehouse_waypoint_cache import WaypointLayoutCache
from env.tasks.multi_agent.warehouse_before_cache import STATE_FIELDS, CONTEXT_FIELDS

NAME = 'stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint'


def test_contract_and_threshold_and_latches():
    env = native_config(NAME)['env']
    validate_experiment(NAME, env, native_config(NAME, True))
    phase = torch.tensor([0, 0, 0, 1, 2, 3, -1])
    root = torch.tensor([[1.299, 0], [1.3, 0], [1.301, 0], [4., 0], [4., 0], [4., 0], [0., 0]])
    payload = torch.full_like(root, 4.)
    payload[4] = torch.tensor([1.3, 0])
    door = torch.zeros_like(root)
    holding = torch.tensor([False, False, False, True, False, False, True])
    got = advance_phase(phase, root, payload, door, holding, 1.3)
    assert got.tolist() == [1, 1, 0, 2, 3, 3, -1]
    # Pickup already inside the radius immediately restores the final target.
    assert advance_phase(torch.tensor([1]), root[:1], door[:1], door[:1], holding[3:4], 1.3).item() == 3


def observations():
    from utils import stage2_spec
    from utils.task_role_spec import task_graph_packet
    env = native_config(NAME)['env']
    _, groups, pre, tasks = stage2_spec.sample_episode(torch.full((8, 4, 3), .5),
        env['box']['build']['taskSizeRanges'], env['relationGraph']['template_probabilities'], preset='mixed')
    graph = stage2_spec.compose_graph(tasks, groups, pre, 8)
    poses = torch.randn(8, 16, 7); poses[..., 3:7] = torch.tensor([0., 0., 0., 1.])
    clean = torch.cat((torch.randn(8, 1136), poses.flatten(1), task_graph_packet(graph, 8).float(), groups.float()), -1)
    follower = (pre >= 0).long().argmax(-1)
    leader = pre.gather(1, follower[:, None]).squeeze(1)
    meta = torch.cat((leader[:, None].float(), torch.tensor([0, 2, 1, 3, -1, -1, 0, 2])[:, None].float(),
                      torch.randn(8, 3)), -1)
    return torch.cat((clean, torch.randn(8, 9216)), -1), meta


def test_mixed_policy_only_guides_leader_and_replay_is_independent_of_task():
    torch.set_num_threads(1)
    base = build(SensorStage2Builder, NAME, 10488)
    guided = build(WarehouseWaypointBuilder, NAME, 10493)
    guided.load_state_dict(base.state_dict(), strict=True)
    obs, meta = observations()
    full = torch.cat((obs, meta), -1)
    for method in ('eval_actor', 'eval_critic'):
        expected = getattr(base.a2c_network, method)(obs)
        actual = getattr(guided.a2c_network, method)(full)
        if isinstance(actual, tuple): actual, expected = actual[0], expected[0]
        expected, actual = expected.reshape(8, 4, -1), actual.reshape(8, 4, -1)
        mask = torch.ones(8, 4, dtype=torch.bool)
        rows = torch.tensor([0, 1, 6, 7]); mask[rows, meta[rows, 0].long()] = False
        torch.testing.assert_close(actual[mask], expected[mask], atol=1e-6, rtol=1e-6)
        assert (actual[~mask]-expected[~mask]).abs().max() > 1e-7
        replay = getattr(guided.a2c_network, method)(full.clone())
        if isinstance(replay, tuple): replay = replay[0]
        torch.testing.assert_close(actual.flatten(0, 1), replay)
    # The mixed distribution must provide gradients to both the head and critic.
    net = guided.a2c_network
    loss = net.eval_actor(full)[0].square().mean()+net.eval_critic(full).square().mean()
    loss.backward()
    assert net.action_head[0][0].weight.grad.abs().sum() > 0
    assert net.critic_encoder.parameters().__next__().grad is not None


def cache_task():
    task = make_task(); task.num_envs = 3
    task._warehouse_start_mode = torch.zeros(3, dtype=torch.long)
    task._relation_graph_spec = {'shuffle_edge_order': False}
    task._waypoint_phase = torch.zeros(3, dtype=torch.long)
    return task


def test_device_cache_restores_both_variants_and_histories_without_touching_other_rows():
    task = cache_task()
    cache = WaypointLayoutCache(task, dict(capacity=2, samples=2, refresh_probability=0., warmup_resets=0))
    ids = torch.arange(3)
    assert not len(cache.restore(task, ids)); cache.finish(task, ids)
    expected = {name: getattr(task, name).clone() for name in (*STATE_FIELDS, *CONTEXT_FIELDS, '_hist_amp_obs_buf')}
    for name in STATE_FIELDS: getattr(task, name)[:2] = -999
    hits = cache.restore(task, ids[:2]); assert hits.tolist() == [0, 1]
    task._hist_amp_obs_buf[:2] = -999
    cache.finish(task, ids[:0])
    for name, value in expected.items(): torch.testing.assert_close(getattr(task, name), value)
    assert (task._waypoint_phase == 0).all()


@pytest.mark.parametrize('name', ['_box_size', '_env_origins', '_structure_yaw', '_surface_half', '_warehouse_start_mode', '_templates', '_before_near_pairs'])
def test_cache_miss_on_condition_change(name):
    task = cache_task()
    cache = WaypointLayoutCache(task, dict(capacity=2, samples=2, refresh_probability=0., warmup_resets=0))
    ids = torch.tensor([1]); cache.restore(task, ids); cache.finish(task, ids)
    value = getattr(task, name)
    if value.dtype == torch.bool: value[1] = ~value[1]
    else: value[1] += 1
    assert not len(cache.restore(task, ids))


def test_cache_refresh_and_eviction_and_warmup_preserve_rng_and_episode_counters():
    task = cache_task()
    task._scene_ids = torch.full((3,), 3)
    from utils.sensor_stage2_spec import WAREHOUSE
    task._scene_ids[:] = WAREHOUSE
    names = ('_sampling_counts', '_sampling_retries', '_scene_sampling', '_stage2_domain_counts', '_size_rsi_executed', '_relation_episode_id')
    for name in names: setattr(task, name, torch.zeros(3))
    cfg = dict(capacity=2, samples=2, refresh_probability=.25, warmup_resets=2)
    cache = WaypointLayoutCache(task, cfg)
    saved_flags, rng = task._warehouse_before_pairs.clone(), torch.get_rng_state()
    def reset(ids):
        for name in names: getattr(task, name).add_(1)
        cache.warmup(task, ids)
        assert not len(cache.restore(task, ids))
        cache.finish(task, ids)
    task._reset_ontop_context_envs = reset
    cache.warmup(task, torch.arange(3)); cache.warmup(task, torch.arange(3))
    assert torch.equal(saved_flags, task._warehouse_before_pairs)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(not getattr(task, name).any() for name in names)
    assert cache.refresh_probability == .25
    assert not cache.stats.any()
    cache.refresh_probability = 0.
    ids = torch.tensor([1])
    assert len(cache.restore(task, ids)) == 1
    cache.refresh_probability = 1.
    assert not len(cache.restore(task, ids))
    cache.finish(task, ids)
    for i in range(4):
        task._box_size[1] += 1
        assert not len(cache.restore(task, ids))
        cache.finish(task, ids)
    assert (cache.count[1] > 0).sum() <= 4  # two variants, two condition slots


def test_frozen_sensor_transfer_and_mixed_distribution_likelihood():
    from learning.multi_agent.stage2_push_network import Stage2PushBuilder
    from learning.multi_agent.sensor_stage2_network import transfer
    source = build(Stage2PushBuilder, 'stage2_distill_900', 1272)
    model = build(WarehouseWaypointBuilder, NAME, 10493)
    transfer(model, dict(model=source.state_dict(), relation_experiment_config=native_config('stage2_distill_900')))
    obs, metadata = observations(); obs = torch.cat((obs, metadata), -1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    net = model.a2c_network
    mu, log_sigma = net.eval_actor(obs)
    dist = torch.distributions.Normal(mu, log_sigma.exp())
    action = dist.sample(); logp = dist.log_prob(action).sum(-1).detach()
    replay_mu, replay_sigma = net.eval_actor(obs.clone())
    replay_logp = torch.distributions.Normal(replay_mu, replay_sigma.exp()).log_prob(action).sum(-1)
    torch.testing.assert_close(logp, replay_logp)
    loss = -replay_logp.mean()+net.eval_critic(obs).square().mean()+net.eval_disc(torch.randn(8, 1330)).square().mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    for name, value in before.items():
        if name.startswith(('a2c_network.actor_encoder.', 'a2c_network.coordination.')) or name == 'a2c_network.sigma':
            assert torch.equal(value, model.state_dict()[name]), name
    for prefix in ('action_head.', 'value_head.', 'actor_sensor_projection.', 'critic_sensor_projection.', '_disc_mlp.'):
        assert any(not torch.equal(value, model.state_dict()[name]) for name, value in before.items() if name.startswith('a2c_network.'+prefix))


def test_checkpoint_contract_does_not_mix_waypoint_and_existing_sensor_models():
    from utils.coupled_push_spec import checkpoint_contract, check_checkpoint
    env = native_config(NAME)['env']
    old = native_config('stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress')['env']
    checkpoint = {'coupled_push_contract': checkpoint_contract(env)}
    check_checkpoint(checkpoint, env)
    with pytest.raises(ValueError): check_checkpoint(checkpoint, old, inference=True)
    with pytest.raises(ValueError): check_checkpoint({'coupled_push_contract': checkpoint_contract(old)}, env)
    changed = deepcopy(env); changed['sensorScene']['warehouse']['waypoint']['switch_radius'] = 1.5
    with pytest.raises(ValueError): check_checkpoint(checkpoint, changed)


def test_ppo_snapshot_state_dict_and_optimizer_reload(tmp_path):
    torch.set_num_threads(1)
    model = build(WarehouseWaypointBuilder, NAME, 10493)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    obs, meta = observations(); full = torch.cat((obs, meta), -1)
    loss = model.a2c_network.eval_actor(full)[0].square().mean()+model.a2c_network.eval_critic(full).square().mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), tmp_path/'resume.pth')
    copy = build(WarehouseWaypointBuilder, NAME, 10493)
    restored = torch.optim.Adam(copy.parameters(), lr=2e-5)
    checkpoint = torch.load(tmp_path/'resume.pth', weights_only=False)
    copy.load_state_dict(checkpoint['model'], strict=True); restored.load_state_dict(checkpoint['optimizer'])
    for name, value in model.state_dict().items(): assert torch.equal(value, copy.state_dict()[name])
    torch.testing.assert_close(model.a2c_network.eval_actor(full)[0], copy.a2c_network.eval_actor(full)[0])
    assert len(restored.state) == len(optimizer.state)


def test_frozen_actor_feature_reuse_preserves_outputs_likelihood_and_trainable_gradients():
    from learning.multi_agent.sensor_stage2_network import freeze_backbone
    torch.set_num_threads(1)
    model = build(WarehouseWaypointBuilder, NAME, 10493)
    freeze_backbone(model)
    obs, meta = observations(); full = torch.cat((obs, meta), -1)
    net = model.a2c_network
    features = net.encode_actor_features(full)
    assert not features.requires_grad
    expected = net.eval_actor(full)[0]
    actual = net.actor_from_features(full, features)[0]
    torch.testing.assert_close(expected, actual)
    result = net({'obs': full, 'frozen_actor_features': features})
    torch.testing.assert_close(result[0], expected)
    torch.testing.assert_close(result[2], net.eval_critic(full))
    expected.square().mean().backward()
    grads = {name: p.grad.clone() for name,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    net.actor_from_features(full, features)[0].square().mean().backward()
    for name,p in model.named_parameters():
        if name in grads: torch.testing.assert_close(p.grad, grads[name])
    # Fresh head/sensor updates must not be hidden by feature reuse.
    with torch.no_grad(): net.action_head[0][0].weight.add_(.01)
    torch.testing.assert_close(net.eval_actor(full)[0], net.actor_from_features(full, features)[0])
    assert not torch.allclose(expected, net.actor_from_features(full, features)[0])


def test_cache_ignores_role_list_order_but_preserves_actor_bindings_and_regions():
    task = cache_task()
    cfg = dict(capacity=2, samples=2, refresh_probability=0., warmup_resets=0)
    cache = WaypointLayoutCache(task, cfg)
    ids = torch.tensor([1]); cache.restore(task, ids); cache.finish(task, ids)
    task._roles[1] = task._roles[1].flip(0)
    assert cache.restore(task, ids).tolist() == [1]
    task._regions[1, 0] += 1
    assert not len(cache.restore(task, ids))


def test_frozen_features_equivariant_with_runtime_token_and_human_shuffles():
    from learning.multi_agent.sensor_stage2_network import freeze_backbone
    model = build(WarehouseWaypointBuilder, NAME, 10493)
    freeze_backbone(model)
    net = model.a2c_network
    net.actor_encoder.relation_graph_spec['shuffle_token_order'] = True
    net.coordination.shuffle_human_order = True
    obs, meta = observations(); full = torch.cat((obs, meta), -1)
    features = net.encode_actor_features(full)
    for _ in range(3):
        torch.testing.assert_close(net.eval_actor(full)[0], net.actor_from_features(full, features)[0], atol=1e-5, rtol=1e-5)


def test_role_views_are_batched_once_and_real_view_outputs_are_preserved():
    from utils.before_object_alias import encoder_packet, goal_aligned_scene
    from utils.warehouse_waypoint import waypoint_scene
    model = build(WarehouseWaypointBuilder, NAME, 10493)
    base = build(SensorStage2Builder, NAME, 10488)
    base.load_state_dict(model.state_dict())
    net = model.a2c_network
    obs, metadata = observations(); full = torch.cat((obs,metadata),-1)
    active = torch.tensor([0,1,6,7]); scene = obs[:,:1268]
    expected = goal_aligned_scene(scene,4,8)
    guided = waypoint_scene(scene[active],metadata[active,0].long(),metadata[active,2:],4,8,metadata[active,1]==2)
    expected = torch.cat((expected,guided),0)
    expected = torch.cat((expected[:,:-20],encoder_packet(expected[:,-20:],4,8)),-1)
    for method, branch in (('eval_actor','actor_encoder'),('eval_critic','critic_encoder')):
        calls=[]
        handle=getattr(net,branch).register_forward_pre_hook(lambda module,args:calls.append(args[0].detach().clone()))
        actual=getattr(net,method)(full);handle.remove()
        assert len(calls)==1 and len(calls[0])==len(obs)+len(active)
        torch.testing.assert_close(calls[0],expected)
        # Both views keep intrinsic features; only GTA XYZ can change.
        torch.testing.assert_close(calls[0][:8,:1136],obs[:,:1136])
        torch.testing.assert_close(calls[0][8:,:1136],obs[active,:1136])
        reference=getattr(base.a2c_network,method)(obs)
        if isinstance(actual,tuple): actual,reference=actual[0],reference[0]
        actual,reference=actual.reshape(8,4,-1),reference.reshape(8,4,-1)
        mask=torch.ones(8,4,dtype=torch.bool);mask[active,metadata[active,0].long()]=False
        torch.testing.assert_close(actual[mask],reference[mask],atol=1e-6,rtol=1e-6)
        assert (actual[~mask]-reference[~mask]).abs().max()>1e-7


@pytest.mark.parametrize('phase',[0,2])
@pytest.mark.parametrize('family',[0,1,3])
def test_follower_goal_alias_and_leader_waypoint_are_distinct_and_pushers_use_real_view(phase,family):
    from utils.before_object_alias import alias_layout
    model=build(WarehouseWaypointBuilder,NAME,10493);net=model.a2c_network
    packet=torch.tensor([[[1,2,2,6,14],[1,family,0,4 if family==3 else 0,6],
                          [1,4,1,5,13],[1,4,3,7,15]]],dtype=torch.float)[:,[2,0,3,1]]
    poses=torch.randn(1,16,7);poses[...,3:7]=torch.tensor([0.,0.,0.,1.])
    poses[0,6,:3]=torch.tensor([3.,2.,.4]);poses[0,14,:3]=torch.tensor([-3.,-2.,.4])
    clean=torch.cat((torch.randn(1,1136),poses.flatten(1),packet.flatten(1),torch.tensor([[2.,1.,2.,1.]])),-1)
    door=torch.tensor([[-4.2,0.,.4]])
    meta=torch.cat((torch.tensor([[2.,float(phase)]]),door),-1)
    obs=torch.cat((clean,torch.randn(1,9216),meta),-1)
    captured=[]
    handle=net.actor_encoder.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    output=net.eval_actor(obs)[0].reshape(1,4,-1);handle.remove()
    assert len(captured)==1 and len(captured[0])==2
    encoded=captured[0][:,1136:1248].reshape(2,16,7)
    _,_,alias,_=alias_layout(packet,4,8)
    follower_record=(packet[0,:,2]==0).nonzero().item();follower_alias=alias[0,follower_record].item()
    assert follower_alias not in (6,14)
    # Real view (row0): even the leader's real source/final goal stay intact.
    torch.testing.assert_close(encoded[0,6],poses[0,6]);torch.testing.assert_close(encoded[0,14],poses[0,14])
    # Both views retain the follower's separate virtual support at the final goal.
    torch.testing.assert_close(encoded[:,follower_alias,:3],poses[0,14,:3].expand(2,-1))
    changed,kept=(6,14) if phase==0 else (14,6)
    torch.testing.assert_close(encoded[1,changed,:3],door[0])
    torch.testing.assert_close(encoded[1,kept],poses[0,kept])
    # Compare PUSH1, PUSH3 and follower0 with the exact same policy on a disabled waypoint.
    actual_obs=obs.clone();actual_obs[:,-4]=-1
    real_output=net.eval_actor(actual_obs)[0].reshape(1,4,-1)
    torch.testing.assert_close(output[:,[0,1,3]],real_output[:,[0,1,3]],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(obs[:,1136:1248].reshape(1,16,7),poses)


def test_ontop_training_roles_and_radius():
    from utils.coupled_push_spec import check_checkpoint, checkpoint_contract
    from utils.sensor_stage2_spec import ROOM, WAREHOUSE
    from utils.warehouse_waypoint import role_eligibility

    name = SENSOR_WAREHOUSE_WAYPOINT_ONTOP
    env = native_config(name)['env']
    validate_experiment(name, env, native_config(name, True))
    warehouse = env['sensorScene']['warehouse']
    assert warehouse['waypoint']['switch_radius'] == 1.2
    assert warehouse['cooperative']['before_pairs_probability'] == .2
    old_env = native_config(NAME)['env']
    with pytest.raises(ValueError):
        check_checkpoint({'coupled_push_contract': checkpoint_contract(old_env)}, env)

    scene = torch.tensor([WAREHOUSE, WAREHOUSE, WAREHOUSE, ROOM])
    before_pairs = torch.tensor([False, False, True, False])
    predecessor = torch.tensor([[2, -1, -1, -1]]).repeat(4, 1)
    tasks = torch.tensor([[4, 5, 3, 5], [1, 5, 3, 5], [4, 5, 3, 5], [4, 5, 3, 5]])
    roles, eligible = role_eligibility(scene, before_pairs, predecessor, tasks)
    assert roles.tolist() == [[2, 0]] * 4
    assert eligible.tolist() == [[True, True], [True, False], [False, False], [False, False]]

    phase = torch.tensor([[0, 0], [1, 1], [2, 2], [3, -1]])
    root = torch.tensor([[[1.2, 0.], [1.201, 0.]]]).expand(4, -1, -1)
    payload = torch.tensor([[[1.199, 0.], [1.201, 0.]]]).expand_as(root)
    holding = torch.tensor([[False, True], [False, True], [False, True], [True, True]])
    actual = advance_phase(phase, root, payload, torch.zeros(4, 1, 2), holding, 1.2)
    assert actual.tolist() == [[1, 0], [1, 2], [3, 2], [3, -1]]
    inside = advance_phase(torch.ones_like(phase), root, payload, torch.zeros(4, 1, 2),
                           torch.ones_like(holding), 1.2)
    assert inside.tolist() == [[3, 2]] * 4


@pytest.mark.parametrize('box_only', [False, True])
def test_ontop_training_partial_reset_clears_both_role_phases(monkeypatch, box_only):
    from env.tasks.multi_agent.humanoid_ma_sensor_stage2 import HumanoidMASensorStage2
    from env.tasks.multi_agent.warehouse_waypoint_task import WarehouseWaypointTrainingTask

    task = WarehouseWaypointTrainingTask.__new__(WarehouseWaypointTrainingTask)
    task._waypoint_box_only = box_only
    task._waypoint_phase = torch.tensor([[0, 2], [3, -1], [1, 3], [-1, -1]])
    if box_only:
        task._waypoint_door_pickup = torch.ones(4, 4, dtype=torch.bool)
    monkeypatch.setattr(HumanoidMASensorStage2, '_reset_envs', lambda task, ids: None)
    task._reset_envs(torch.tensor([1, 3]))
    assert task._waypoint_phase.tolist() == [[0, 2], [0, 0], [1, 3], [0, 0]]
    if box_only:
        assert task._waypoint_door_pickup.all(-1).tolist() == [True, False, True, False]
        assert not task._waypoint_door_pickup[[1, 3]].any()


def test_ontop_training_batched_role_outputs_and_frozen_replay():
    from learning.multi_agent.sensor_stage2_network import freeze_backbone
    from qualitative.warehouse_waypoint import ontop_waypoint_scene, waypoint_scene
    from utils.before_object_alias import encoder_packet, goal_aligned_scene

    name = SENSOR_WAREHOUSE_WAYPOINT_ONTOP
    model = build(WarehouseWaypointBuilder, name, 10495)
    freeze_backbone(model)
    net = model.a2c_network
    packet = torch.tensor([[1, 2, 2, 6, 14], [1, 3, 0, 4, 6], [1, 4, 1, 5, 13],
                           [1, 4, 3, 7, 15]]).float().repeat(2, 1, 1)[:, [2, 0, 3, 1]]
    poses = torch.randn(2, 16, 7)
    poses[..., 3:7] = torch.tensor([0., 0., 0., 1.])
    scene = torch.cat((torch.randn(2, 1136), poses.flatten(1), packet.flatten(1)), -1)
    door = torch.tensor([[-4.2, 0., .4], [-4.2, 0., .4]])
    metadata = torch.cat((torch.tensor([[2., 0.], [2., 2.]]), door,
                          torch.tensor([[0., 2.], [0., 0.]])), -1)
    obs = torch.cat((scene, torch.tensor([[1., 2., 1., 2.]]).expand(2, -1),
                     torch.randn(2, 9216), metadata), -1)
    snapshot = obs.clone()
    leader_view = waypoint_scene(scene, metadata[:, 0].long(), door, target=metadata[:, 1] == 2)
    follower_view = ontop_waypoint_scene(scene, metadata[:, 5].long(), door, target=metadata[:, 6] == 2)
    guided = torch.stack((leader_view, follower_view), dim=1).flatten(0, 1)
    expected = torch.cat((goal_aligned_scene(scene, 4, 8), guided))
    expected = torch.cat((expected[:, :-20], encoder_packet(expected[:, -20:], 4, 8)), -1)
    disabled = obs.clone()
    disabled[:, [-6, -1]] = -1
    for method, encoder, coordination in (
            ('eval_actor', net.actor_encoder, net.coordination),
            ('eval_critic', net.critic_encoder, net.critic_coordination)):
        calls, coordination_calls = [], []
        encoder_hook = encoder.register_forward_pre_hook(lambda _, args: calls.append(args[0].detach()))
        coordination_hook = coordination.register_forward_pre_hook(
            lambda _, args: coordination_calls.append(len(args[0])))
        evaluate = getattr(net, method)
        values = evaluate(obs)
        encoder_hook.remove()
        coordination_hook.remove()
        assert len(calls) == 1
        assert coordination_calls == [6]
        torch.testing.assert_close(calls[0], expected)
        real = evaluate(disabled)
        if method == 'eval_actor':
            values, real = values[0], real[0]
        torch.testing.assert_close(values.reshape(2, 4, -1)[:, [1, 3]],
                                   real.reshape(2, 4, -1)[:, [1, 3]], atol=1e-6, rtol=1e-6)
    features = net.encode_actor_features(obs)
    mean, sigma = net.eval_actor(obs)
    cached_mean, cached_sigma = net.actor_from_features(obs, features)
    actions = torch.randn_like(mean)
    actual = torch.distributions.Normal(mean, sigma.exp()).log_prob(actions)
    replay = torch.distributions.Normal(cached_mean, cached_sigma.exp()).log_prob(actions)
    torch.testing.assert_close(actual, replay)
    assert torch.equal(obs, snapshot)


def test_box_only_config_and_pickup_latch():
    from utils.coupled_push_spec import check_checkpoint, checkpoint_contract
    from utils.warehouse_waypoint import advance_box_only_phase

    name = SENSOR_WAREHOUSE_WAYPOINT_BOXONLY
    env = native_config(name)['env']
    train = native_config(name, True)
    validate_experiment(name, env, train)
    warehouse = env['sensorScene']['warehouse']
    assert warehouse['cooperative']['before_pairs_probability'] == .2
    assert warehouse['leader_door_progress']
    assert warehouse['waypoint']['switch_radius'] == 1.2
    assert train['params']['network']['waypoint_box_only']
    old = native_config(SENSOR_WAREHOUSE_WAYPOINT_ONTOP)['env']
    with pytest.raises(ValueError):
        check_checkpoint({'coupled_push_contract': checkpoint_contract(old)}, env, inference=True)

    phase = torch.tensor([[0, 0], [0, 0], [1, -1]])
    root = torch.tensor([[[1.2, 0.], [1.201, 0.]], [[4., 0.], [4., 0.]], [[4., 0.], [0., 0.]]])
    holding = torch.tensor([[False, False], [True, True], [True, True]])
    door = torch.zeros(3, 1, 2)
    phase = advance_box_only_phase(phase, root, door, holding, 1.2)
    assert phase.tolist() == [[1, 0], [3, 3], [3, -1]]
    departed = advance_box_only_phase(phase, root + 10., door, torch.zeros_like(holding), 1.2)
    torch.testing.assert_close(departed, phase)


def test_box_only_keeps_both_goals_and_batches_active_humans_once():
    from qualitative.warehouse_waypoint import ontop_waypoint_scene, waypoint_scene
    from utils.before_object_alias import encoder_packet, goal_aligned_scene

    model = build(WarehouseWaypointBuilder, SENSOR_WAREHOUSE_WAYPOINT_BOXONLY, 10495)
    packet = torch.tensor([[1, 2, 2, 6, 14], [1, 3, 0, 4, 6], [1, 4, 1, 5, 13],
                           [1, 4, 3, 7, 15]]).float().repeat(3, 1, 1)
    poses = torch.randn(3, 16, 7)
    poses[..., 3:7] = torch.tensor([0., 0., 0., 1.])
    scene = torch.cat((torch.randn(3, 1136), poses.flatten(1), packet.flatten(1)), -1)
    door = torch.tensor([[-4.2, 0., .4]]).repeat(3, 1)
    # Include a phase-2 row to verify that the builder never creates a guided target view.
    phase = torch.tensor([0., 3., 2.])
    roles = torch.stack((torch.full_like(phase, 2.), phase), -1)
    metadata = torch.cat((roles, door, torch.stack((torch.zeros_like(phase), phase), -1)), -1)
    obs = torch.cat((scene, torch.tensor([[1., 2., 1., 2.]]).expand(3, -1),
                     torch.randn(3, 9216), metadata), -1)
    expected = torch.cat((goal_aligned_scene(scene, 4, 8),
                          waypoint_scene(scene[:1], torch.tensor([2]), door[:1]),
                          ontop_waypoint_scene(scene[:1], torch.tensor([0]), door[:1])))
    expected = torch.cat((expected[:, :-20], encoder_packet(expected[:, -20:], 4, 8)), -1)
    net = model.a2c_network
    real = obs.clone()
    real[:, [-6, -1]] = -1
    for method, encoder in (('eval_actor', net.actor_encoder), ('eval_critic', net.critic_encoder)):
        calls = []
        hook = encoder.register_forward_pre_hook(lambda _, args: calls.append(args[0].detach()))
        evaluate = getattr(net, method)
        actual = evaluate(obs)
        hook.remove()
        assert len(calls) == 1
        torch.testing.assert_close(calls[0], expected)
        reference = evaluate(real)
        if method == 'eval_actor':
            actual, reference = actual[0], reference[0]
        actual, reference = actual.reshape(3, 4, -1), reference.reshape(3, 4, -1)
        torch.testing.assert_close(actual[1:], reference[1:], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual[0, [1, 3]], reference[0, [1, 3]], atol=1e-6, rtol=1e-6)
