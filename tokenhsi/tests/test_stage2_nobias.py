from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from learning.multi_agent.stage2_teachers import Stage2Teachers
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from learning.multi_agent.stage2_push_network import transfer_stage1
from utils.experiments import native_config, validate_experiment, experiment_horizon
from utils.unified_training import validate_unified_env
from utils.coupled_push_spec import checkpoint_contract, check_checkpoint
from test_stage2_push_common import build, _common_observation


NAME = 'stage2_distill_900_v4_nobias'


def teacher_task(directory):
    task = SimpleNamespace(device='cpu', num_agents=4, num_objects=8,
        cfg={'env': native_config(NAME)['env'], 'args': SimpleNamespace(output_path=str(directory))})
    for key, value in dict(get_obs_size=1272, get_num_amp_obs=1330, get_action_size=32,
            get_policy_obs_mode='clean_scene', get_scene_entity_sizes=[223, 30, 1],
            get_scene_kinematic_size=7, get_scene_arena_scale=16.,
            get_scene_normalized_entity_sizes=[223, 30, 0], get_relation_suffix_size=24).items():
        setattr(task, key, lambda result=value: result)
    return task


@pytest.fixture(scope='module')
def teachers(tmp_path_factory):
    torch.set_num_threads(1)
    cfg = native_config(NAME, True)['params']['config']['distillation']
    return Stage2Teachers(teacher_task(tmp_path_factory.mktemp('teachers')),
        cfg['teacher_before'], cfg['teacher_coupled'])


def test_config_preserves_v4_except_dependency_bias_and_teacher_selection():
    cfg = native_config(NAME)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    assert experiment_horizon(NAME) == 900
    baseline = native_config('stage2_distill_900_v4')
    restored = deepcopy(cfg)
    assert restored['env']['relationGraph'].pop('human_relation_bias') is False
    restored['experiment']['name'] = baseline['experiment']['name']
    assert restored == baseline
    train = native_config(NAME, True)
    original = native_config('stage2_distill_900_v4', True)
    train['params']['config']['name'] = original['params']['config']['name']
    for key in ('teacher_before', 'teacher_coupled'):
        assert train['params']['config']['distillation'][key].endswith('_00010000.pth')
        train['params']['config']['distillation'][key] = original['params']['config']['distillation'][key]
    assert train == original
    saved = {'coupled_push_contract': checkpoint_contract(cfg['env'])}
    with pytest.raises(ValueError):
        check_checkpoint(saved, baseline['env'])


def test_teacher_weights_and_normalizers_load_exactly(teachers):
    for teacher in (teachers.before, teachers.coupled):
        saved = torch.load(teacher.metadata['path'], map_location='cpu', weights_only=False)
        assert saved['epoch'] == 10000
        actual = {'a2c_network.'+k: v for k, v in teacher.a2c_network.state_dict().items()}
        assert actual.keys() == saved['model'].keys()
        assert all(torch.equal(v, saved['model'][k]) for k, v in actual.items())
        assert all(torch.equal(v, saved['running_mean_std'][k]) for k, v in teacher.normalizer.state_dict().items())
        assert not any(p.requires_grad for p in teacher.parameters())
        for branch in (teacher.a2c_network.coordination, teacher.a2c_network.critic_coordination):
            assert all(not block.use_relation_bias for block in branch.blocks)


@pytest.mark.parametrize('case', ['before_sit', 'before_climb', 'before_ontop', 'two_pairs', 'coupled', 'solo'])
def test_teacher_routing_uses_shared_push_only(teachers, case):
    obs = _common_observation(case)
    mask = torch.tensor([[False, False, True, True]])
    rng = torch.random.get_rng_state()
    with torch.no_grad():
        expected = torch.where(mask.reshape(-1, 1), teachers.coupled.means(obs), teachers.before.means(obs))
        torch.random.set_rng_state(rng)
        actual = teachers.means(obs, mask)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
    assert torch.isfinite(actual).all()


def test_student_updates_and_reloads_without_dependency_bias(teachers):
    torch.manual_seed(37)
    model = torch.nn.Module()
    model.a2c_network = build(NAME)
    source = torch.load(native_config(NAME)['experiment']['transfer']['checkpoint'],
        map_location='cpu', weights_only=False)
    report = transfer_stage1(model, source)
    assert report['source_epoch'] == 26000
    net = model.a2c_network
    normalizer = SceneRunningMeanStd([223, 30, 1], [4, 8, 4],
        normalized_sizes=[223, 30, 0], kinematic_size=7, extra_passthrough_size=24)
    normalizer.load_state_dict(source['running_mean_std'], strict=True)
    normalizer.eval()
    model.requires_grad_(False)
    net.coordination.requires_grad_(True)
    net.action_head.requires_grad_(True)
    model.eval()
    for branch in (net.coordination, net.critic_coordination):
        assert all(not block.use_relation_bias and block.bias_projection is None for block in branch.blocks)
    assert net.actor_encoder.edge_encoder.bias_projection is not None
    assert not any('coordination.blocks' in k and 'bias_projection' in k for k in model.state_dict())
    net.coordination.shuffle_human_order = False
    net.relation_graph_spec['shuffle_token_order'] = False
    obs = torch.cat([_common_observation(case) for case in ('before_sit', 'coupled', 'solo')])
    with torch.no_grad():
        target = teachers.means(obs, torch.tensor([[False]*4, [True]*4, [False]*4]))
        before = torch.nn.functional.mse_loss(net.eval_actor(normalizer(obs))[0], target)
    encoder = {k: v.clone() for k, v in net.actor_encoder.state_dict().items()}
    fusion = net.coordination.relation_fusion.mlp[-1].weight.detach().clone()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    for _ in range(8):
        loss = torch.nn.functional.mse_loss(net.eval_actor(normalizer(obs))[0], target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    assert torch.isfinite(loss) and loss < before
    assert not torch.equal(fusion, net.coordination.relation_fusion.mlp[-1].weight)
    assert all(torch.equal(v, encoder[k]) for k, v in net.actor_encoder.state_dict().items())
    restored = build(NAME)
    restored.load_state_dict(net.state_dict(), strict=True)
    restored.eval()
    restored.coordination.shuffle_human_order = False
    restored.relation_graph_spec['shuffle_token_order'] = False
    torch.testing.assert_close(restored.eval_actor(normalizer(obs))[0], net.eval_actor(normalizer(obs))[0], rtol=0, atol=0)


def test_nobias_attention_ignores_stored_coupled_projection(teachers):
    net = teachers.coupled.a2c_network
    obs = _common_observation('coupled')
    rng = torch.random.get_rng_state()
    with torch.no_grad():
        expected = teachers.coupled.means(obs)
        projections = [block.bias_projection for block in net.coordination.blocks]
        saved = [p.clone() for p in projections]
        try:
            for p in projections:
                p.fill_(1000.)
            torch.random.set_rng_state(rng)
            actual = teachers.coupled.means(obs)
        finally:
            for p, value in zip(projections, saved):
                p.copy_(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
