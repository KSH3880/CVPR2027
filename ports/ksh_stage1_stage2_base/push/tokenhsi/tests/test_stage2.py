from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from types import SimpleNamespace

from utils.stage2_spec import sample_graph, physical_mapping, validate_env
from utils.task_role_spec import task_graph_packet
from learning.multi_agent.stage2_network_builder import Stage2Builder, transfer_stage1
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd


ROOT = Path(__file__).resolve().parents[2]
ENV = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage2.yaml').read_text())['env']
TRAIN = yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/stage2.yaml').read_text())


def graph(preset, count=64):
    probabilities = torch.tensor([0., .1, .25, .325, .325]).expand(count, 4, 5)
    return sample_graph(probabilities, ENV['relationGraph'], preset)


@pytest.mark.parametrize('preset', ('random_scenario', 'all_solo', 'sit', 'climb',
                                    'two_pairs_at', 'two_pairs_ontop', 'one_pair_at', 'one_pair_ontop'))
def test_group_mapping_and_shared_targets(preset):
    g, groups, joint = graph(preset)
    mapping = physical_mapping(groups, joint)
    ordered = mapping.sort(-1).values
    assert (ordered[:, 1:] != ordered[:, :-1]).all()
    assert mapping.min() >= 0 and mapping.max() < 12
    packet = task_graph_packet(g, 64).reshape(64, 4, 5).long()
    assert packet[..., 0].all()
    assert ((packet[..., 1] >= 0) & (packet[..., 1] <= 3)).all()
    for person in range(4):
        same = groups == groups[:, person:person+1]
        shared = joint[:, person:person+1] & same
        assert (packet[..., 3:].masked_select(shared[..., None]) ==
                packet[:, person:person+1, 3:].expand_as(packet[..., 3:]).masked_select(shared[..., None])).all()


def test_contract_keeps_stage1_settings():
    validate_env(ENV)
    base = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml').read_text())['env']
    for key in ('relationReward', 'box', 'sizeAwareRsi', 'plane'):
        assert ENV[key] == base[key]
    for name, weights in ENV['templateRsi'].items():
        assert weights == ([.5, 0., 0., 0., 0., .1, .4, 0.] if name == 'HOLDING_ON_TOP' else base['templateRsi'][name])
    assert ENV['stage2']['joint_size_ranges'] == [[.2, .6], [.4, 1.2], [.2, .6]]
    assert ENV['numEnvs'] == 1024
    bad = deepcopy(ENV)
    bad['stage2']['anchor_inset'] = .2
    with pytest.raises(ValueError):
        validate_env(bad)


def test_stage1_transfer_identity_and_gradients():
    torch.set_num_threads(1)
    builder = Stage2Builder()
    builder.load(TRAIN['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(5072,), amp_input_shape=(1320,),
        value_size=1, num_agents=4, num_objects=8, observation_mode='clean_scene',
        scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7, scene_arena_scale=16.,
        relation_reward_mode=ENV['relationReward']['mode'], relation_graph_spec=ENV['relationGraph'], device='cpu')
    model = nn.Module()
    model.a2c_network = net
    source = ROOT/'stage1/Stage1V2Fixed_00030500.pth'
    if not source.exists():
        pytest.skip('Local Stage1 checkpoint is required for transfer verification')
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    report = transfer_stage1(model, checkpoint)
    assert len(report['copied']) == len(checkpoint['model'])
    for key in report['copied']:
        torch.testing.assert_close(model.state_dict()[key], checkpoint['model'][key], rtol=0, atol=0)
    g, _, _ = graph('all_solo', 2)
    poses = torch.randn(2, 16, 7)
    poses[..., 3:] = torch.nn.functional.normalize(poses[..., 3:], dim=-1)
    raw = torch.cat((torch.randn(2, 1136), poses.flatten(1), task_graph_packet(g, 2)), -1)
    obs = raw[:, None].expand(-1, 4, -1).flatten(1)
    mu, _ = net.eval_actor(obs)
    expected = net.action_head(net.actor_encoder(raw)).reshape(8, 32)
    torch.testing.assert_close(mu, expected, atol=1e-5, rtol=1e-5)
    (mu.square().mean()+net.eval_critic(obs).square().mean()).backward()
    assert all(p.grad is None for p in net.actor_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.coordination.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.action_head.parameters())
    rms = SceneRunningMeanStd([223, 30, 1], [4, 8, 4], normalized_sizes=[223, 30, 0],
                              kinematic_size=7, extra_passthrough_size=20, num_views=4)
    rms.load_state_dict(checkpoint['running_mean_std'], strict=True)
    rms.eval()
    assert rms(obs).shape == obs.shape


def test_assigned_anchor_matches_grasp_view_under_rotation():
    from env.tasks.multi_agent.stage2_scene import joint_holding_geometry
    from learning.multi_agent.grasp_view import make_grasp_views
    g, groups, _ = graph('two_pairs_at', 1)
    packet = task_graph_packet(g, 1).reshape(1, 4, 5).long()
    sides = torch.empty(1, 4)
    for leader in groups[0].unique():
        members = (groups[0] == leader).nonzero().flatten()
        sides[0, members] = torch.tensor([-1., 1.])
    boxes = torch.zeros(1, 8, 13)
    boxes[..., 5:7] = 2**-.5
    sizes = torch.tensor([.4, .8, .4]).expand(1, 8, 3).clone()
    hands = torch.zeros(1, 4, 15, 13)
    hands[:, :, [5, 8], 0] = (-.25*sides)[..., None]
    task = SimpleNamespace(num_envs=1, device='cpu', relation_runtime=SimpleNamespace(graph=g),
                           _kinematic_humanoid_rigid_body_states=hands, _key_body_ids=torch.tensor([5, 8, 11, 14]),
                           _box_states=boxes, _box_size=sizes, _grasp_sides=sides,
                           cfg={'env': {'stage2': {'anchor_inset': .15}}}, _joint=torch.ones(1, 4, dtype=torch.bool),
                           _logical_box_values=lambda values, ids: values[ids])
    phi, diag = joint_holding_geometry(task, torch.zeros(1, 8), {'target': torch.zeros(1, 8, 3), 'distance': torch.zeros(1, 8)}, torch.tensor([0]))
    holding = g.edge_relation == 6
    torch.testing.assert_close(phi[holding], torch.ones(4))
    raw = torch.zeros(1, 1268)
    corners = torch.tensor([[x, y, z] for x in (-.2, .2) for y in (-.4, .4) for z in (-.2, .2)]).flatten()
    raw[:, 892:1132].reshape(1, 8, 30)[..., 6:] = corners
    poses = raw[:, 1136:1248].reshape(1, 16, 7)
    poses[:, 4:12] = boxes[..., :7]
    raw[:, -20:] = packet.flatten(1)
    views = make_grasp_views(raw, sides, .15).reshape(1, 4, 1268)
    for edge in holding[0].nonzero().flatten():
        owner = g.edge_owner[0, edge]
        payload = packet[0, owner, 3]
        view_target = views[:, owner, 1136:1248].reshape(1, 16, 7)[:, payload, :3]
        torch.testing.assert_close(view_target, diag['target'][:, edge], atol=1e-6, rtol=0)
    hands[:, 0, [5, 8], 0] *= -1
    wrong, _ = joint_holding_geometry(task, phi, diag, torch.tensor([0]))
    assert (wrong[holding & (g.edge_owner == 0)] < .9).all()


def test_viewer_camera_ignores_parked_physical_boxes(monkeypatch):
    import sys
    from env.tasks.multi_agent.edge_ontop_task import SampledOnTopTaskMixin
    monkeypatch.setitem(sys.modules, 'isaacgym', SimpleNamespace(
        gymapi=SimpleNamespace(Vec3=lambda *coordinates: coordinates)))
    visible = torch.tensor([0, 3, 5, 6, 8, 9, 10, 11])
    parked = torch.tensor([1, 2, 4, 7])
    boxes = torch.zeros(1, 12, 13)
    boxes[0, visible, 0] = torch.linspace(-2, 2, 8)
    boxes[0, visible, 1] = torch.linspace(-3, 3, 8)
    boxes[0, visible, 2] = .4
    humans = torch.zeros(1, 4, 13)
    humans[..., 2] = 1.
    views = []
    task = SimpleNamespace(cfg={'args': SimpleNamespace(task_camera='stack')},
        _box_states=boxes, _humanoid_root_states=humans, _camera_follow_flag=True,
        _cam_prev_char_pos=torch.zeros(3).numpy(), viewer=object(),
        _logical_box_values=lambda values: values[:, visible],
        gym=SimpleNamespace(viewer_camera_look_at=lambda viewer, env, camera, target: views.append((camera, target))))
    for distance in (1000., 1000000.):
        boxes[0, parked, :2] = distance
        SampledOnTopTaskMixin._update_camera(task)
    assert views[0] == views[1]
    assert views[0][1] == pytest.approx((0., 0., .8))


def test_empty_rsi_profile_triggers_regeneration(tmp_path, monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(ROOT))
    from utils import stage2_rsi
    monkeypatch.chdir(tmp_path)
    directory = tmp_path/'rsi'
    profile = directory/'8_16_8'
    profile.mkdir(parents=True)
    (profile/'carryWith.npz').write_bytes(b'empty profile')
    (profile/'carryWith.json').write_text(json.dumps(dict(accepted=0, contract_digest='test', size=[.4,.8,.4])))
    monkeypatch.setattr(stage2_rsi, 'contract_digest', lambda *args: 'test')
    def regenerate(*args, **kwargs):
        raise RuntimeError('regeneration requested')
    monkeypatch.setattr(stage2_rsi.subprocess, 'run', regenerate)
    task = SimpleNamespace(cfg={'env': {'stage2': {'rsi_directory': str(directory), 'anchor_inset': .15}}}, device='cpu')
    with pytest.raises(RuntimeError, match='regeneration requested'):
        stage2_rsi.JointRsiLibrary(task).sample('carryWith', torch.tensor([[.4,.8,.4]]))


@pytest.mark.parametrize('cached', ('missing', 'stale', 'ready'))
def test_rsi_loads_or_builds_fifteen_cm_cache(tmp_path, monkeypatch, cached):
    import hashlib
    import json
    import numpy as np
    monkeypatch.syspath_prepend(str(ROOT))
    from utils import stage2_rsi
    monkeypatch.chdir(tmp_path)
    directory = tmp_path/'inset15_single'
    profile = directory/'8_16_8'
    calls = []
    def save(signature):
        profile.mkdir(parents=True, exist_ok=True)
        np.savez(profile/'carryWith.npz', root_state=np.zeros((1,2,13)),
                 dof_position=np.zeros((1,2,32)), dof_velocity=np.zeros((1,2,32)),
                 body_state=np.zeros((1,2,15,13)), box_state=np.zeros((1,13)),
                 terminal_box_xy=np.zeros((1,2)), source_frame=np.array([42]))
        (profile/'carryWith.json').write_text(json.dumps(dict(accepted=1, size=[.4,.8,.4],
            contract_digest=signature, sha256=hashlib.sha256((profile/'carryWith.npz').read_bytes()).hexdigest())))
    if cached != 'missing':
        save('single:0.15' if cached == 'ready' else 'outdated')
    def build(command, **kwargs):
        calls.append(command)
        assert command[command.index('--source')+1] == 'single'
        assert command[command.index('--inset')+1] == '0.15'
        assert command[command.index('--directory')+1] == str(directory)
        save('single:0.15')
    monkeypatch.setattr(stage2_rsi, 'contract_digest', lambda source, inset: source+':'+str(inset))
    monkeypatch.setattr(stage2_rsi.subprocess, 'run', build)
    task = SimpleNamespace(cfg={'env': {'stage2': {'rsi_directory': str(directory), 'anchor_inset': .15}}}, device='cpu')
    result = stage2_rsi.JointRsiLibrary(task).sample('carryWith', torch.tensor([[.4,.8,.4]]))
    assert result['source_frame'].item() == 42
    assert len(calls) == (0 if cached == 'ready' else 1)
