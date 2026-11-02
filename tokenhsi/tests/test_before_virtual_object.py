from copy import deepcopy

import pytest
import torch

from utils.before_object_alias import VIRTUAL_MODE, alias_layout, encoder_packet, goal_aligned_scene
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.experiments import native_config, validate_experiment
from utils.unified_training import validate_unified_env
from test_before_object_alias import observations
from test_stage2_push_common import build, _common_observation

BASE = 'stage2_before_push_near50_pair50_source5'
NAME = BASE + '_virtual'


def test_config_preserves_source5_environment_and_training():
    base, cfg = native_config(BASE), native_config(NAME)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    restored = deepcopy(cfg['env'])
    assert restored['relationGraph'].pop('before_virtual_object') == VIRTUAL_MODE
    assert restored == base['env'] and cfg['sim'] == base['sim']
    train = native_config(NAME, True)
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)
    assert cfg['experiment']['transfer']['checkpoint'] == 'pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth'
    old, new = checkpoint_contract(base['env']), checkpoint_contract(cfg['env'])
    assert new.pop('before_virtual_object') == VIRTUAL_MODE and new == old
    for source, target in ((base, cfg), (cfg, base)):
        with pytest.raises(ValueError, match='checkpoint contract'):
            check_checkpoint({'before_contract': checkpoint_contract(source['env'])}, target['env'])
        with pytest.raises(ValueError, match='selected experiment'):
            validate_experiment(source['experiment']['name'], target['env'], native_config(source['experiment']['name'], True))


@pytest.mark.parametrize('mode,alias', [('unknown', True), (VIRTUAL_MODE, False)])
def test_invalid_virtual_mode_or_missing_alias_is_rejected(mode, alias):
    env = native_config(NAME)['env']
    env['relationGraph']['before_virtual_object'] = mode
    if not alias:
        env['relationGraph'].pop('before_object_alias')
    with pytest.raises(ValueError, match='virtual objects'):
        validate_unified_env(env)


@pytest.mark.parametrize('kind', [0, 1, 3])
def test_two_pairs_follow_packet_goals_and_only_alias_xyz_changes(kind):
    p = torch.tensor([[[1, 2, 3, 4, 13], [1, kind, 0, 6 if kind == 3 else 0, 4],
                       [1, 2, 1, 5, 15], [1, kind, 2, 7 if kind == 3 else 0, 5]]], dtype=torch.float)
    p = p.expand(3, -1, -1).clone()
    p[1] = p[1, [2, 0, 3, 1]]
    p[2] = p[2, [3, 2, 1, 0]]
    scene = observations(p)
    dependent, target, alias, _ = alias_layout(p, 4, 8)
    row, record = dependent.nonzero(as_tuple=True)
    changed = torch.zeros_like(scene, dtype=torch.bool)
    changed[:, 1136:1248].reshape(-1, 16, 7)[row, alias[row, record], :3] = True
    for _ in range(3):
        poses = scene[:, 1136:1248].reshape(-1, 16, 7)
        quat = torch.nn.functional.normalize(torch.randn(3, 8, 4), dim=-1)
        poses[:, 4:12, 3:7] = quat
        poses[row, alias[row, record], 3:7] = poses[row, target[row, record], 3:7]
        saved = scene.clone()
        result = goal_aligned_scene(scene, 4, 8)
        virtual = result[:, 1136:1248].reshape(-1, 16, 7)
        # Goal IDs deliberately differ from m+o+actor, and records are shuffled.
        goals = torch.where(p[row, record, 2] == 0, 13, 15)
        assert torch.equal(virtual[row, alias[row, record], :3], poses[row, goals, :3])
        assert torch.equal(virtual[row, alias[row, record], 3:7], poses[row, target[row, record], 3:7])
        assert torch.equal(result[~changed], scene[~changed])
        assert torch.equal(scene, saved)
        assert torch.equal(goal_aligned_scene(scene[[2, 0]], 4, 8), result[[2, 0]])


@pytest.mark.parametrize('case', ['before_sit', 'before_climb', 'before_ontop', 'two_pairs', 'solo', 'coupled'])
def test_actor_critic_receive_virtual_scene_and_coordination_keeps_original_packet(case):
    torch.set_num_threads(1)
    net = build(NAME).eval()
    obs = _common_observation(case).requires_grad_()
    saved = obs.detach().clone()
    scene = obs[:, :-4]
    virtual = goal_aligned_scene(scene, 4, 8)
    expected = torch.cat((virtual[:, :-20], encoder_packet(scene[:, -20:], 4, 8)), -1)
    seen = []
    def encoder_hook(module, args):
        assert torch.equal(args[0], expected)
        seen.append('encoder')
    def coordination_hook(module, args):
        assert torch.equal(args[2], scene[:, -20:])
        seen.append('coordination')
    hooks = [e.register_forward_pre_hook(encoder_hook) for e in (net.actor_encoder, net.critic_encoder)]
    hooks += [e.register_forward_pre_hook(coordination_hook) for e in (net.coordination, net.critic_coordination)]
    mu, _ = net.eval_actor(obs)
    value = net.eval_critic(obs)
    (mu.square().mean()+value.square().mean()).backward()
    assert seen == ['encoder', 'coordination', 'encoder', 'coordination']
    assert torch.equal(obs, saved) and torch.isfinite(obs.grad).all()
    for hook in hooks:
        hook.remove()


@pytest.mark.parametrize('case', ['solo', 'coupled'])
def test_virtual_option_preserves_independent_and_coupled_outputs_bitwise(case):
    torch.set_num_threads(1)
    old, new = build(BASE).eval(), build(NAME).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    for net in (old, new):
        net.relation_graph_spec['shuffle_token_order'] = False
        net.coordination.shuffle_human_order = net.critic_coordination.shuffle_human_order = False
    obs = _common_observation(case)
    assert torch.equal(goal_aligned_scene(obs[:, :-4], 4, 8), obs[:, :-4])
    with torch.no_grad():
        for method in ('eval_actor', 'eval_critic'):
            a, b = getattr(old, method)(obs), getattr(new, method)(obs)
            if isinstance(a, tuple):
                a, b = a[0], b[0]
            assert torch.equal(a, b)


def test_encoder_token_permutation_preserves_virtual_mapping():
    torch.set_num_threads(1)
    net = build(NAME).eval()
    scene = _common_observation('two_pairs')[:, :-4]
    net.relation_graph_spec['shuffle_token_order'] = False
    with torch.no_grad():
        for encoder in (net.actor_encoder, net.critic_encoder):
            expected = net._humans(encoder, scene)
            order = torch.tensor([15, 1, 7, 0, 12, 4, 10, 2, 9, 6, 8, 11, 3, 14, 5, 13])
            hook = encoder.register_forward_pre_hook(lambda module, args: (args[0], False, order))
            actual = net._humans(encoder, scene)
            hook.remove()
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_mixed_batches_and_invalid_at_records_do_not_create_false_aliases():
    cases = ['before_sit', 'solo', 'two_pairs', 'coupled', 'before_climb', 'before_ontop']
    scene = torch.cat([_common_observation(case)[:, :-4] for case in cases])
    invalid = _common_observation('before_sit')[:, :-4].clone()
    invalid[:, -20] = 0
    scene = torch.cat((scene, invalid))
    actual = goal_aligned_scene(scene, 4, 8)
    for row in range(len(scene)):
        assert torch.equal(actual[row:row+1], goal_aligned_scene(scene[row:row+1], 4, 8))
    assert torch.equal(actual[[1, 3, 6]], scene[[1, 3, 6]])


@pytest.mark.parametrize('m', [2, 6])
def test_pose_offsets_follow_entity_counts(m):
    o = 2*m
    p = torch.zeros(1, m, 5)
    p[0, 0] = torch.tensor([1, 2, 0, m, m+o+m-1])
    p[0, 1] = torch.tensor([1, 0, 1, 0, m])
    for actor in range(2, m):
        p[0, actor] = torch.tensor([1, 0, actor, 0, m+actor-1])
    intrinsic = torch.randn(1, m*223+o*30+m)
    poses = torch.randn(1, 2*m+o, 7)
    scene = torch.cat((intrinsic, poses.flatten(1), p.flatten(1)), -1)
    result = goal_aligned_scene(scene, m, o)
    virtual = result[:, intrinsic.shape[1]:-5*m].reshape_as(poses)
    alias = m+m-1
    assert torch.equal(virtual[0, alias, :3], poses[0, m+o+m-1, :3])
    assert torch.equal(virtual[0, alias, 3:7], poses[0, alias, 3:7])
    assert torch.equal(result[:, :intrinsic.shape[1]], intrinsic)


def test_26k_transfer_preserves_weights_and_freezing():
    from learning.multi_agent.stage2_push_network import transfer_stage1
    torch.set_num_threads(1)
    checkpoint = torch.load(native_config(NAME)['experiment']['transfer']['checkpoint'], map_location='cpu', weights_only=False)
    model = torch.nn.Module()
    model.a2c_network = build(NAME)
    report = transfer_stage1(model, checkpoint)
    assert report['source_epoch'] == 26000 and len(report['copied']) == 165
    assert all(torch.equal(model.state_dict()[key], value) for key, value in checkpoint['model'].items())
    assert all(not p.requires_grad for p in model.a2c_network.actor_encoder.parameters())
    assert all(p.requires_grad for p in model.a2c_network.critic_encoder.parameters())
