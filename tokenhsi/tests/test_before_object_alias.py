from copy import deepcopy

import pytest
import torch

from test_before_adapter import build, config
from utils.before_object_alias import MODE, alias_layout, encoder_packet, replace_distractors
from utils.before_spec import validate_env, checkpoint_contract
from utils.stage2_spec import check_checkpoint
from utils.task_role_spec import task_edges
from learning.multi_agent.human_coordination import human_relations
from learning.multi_agent.stage2_network_builder import freeze_actor, transfer_stage1
from utils.relation_task_spec import checkpoint_metadata

BASE = 'stage2_before_ground_atprogress_nonemask_rsi433_near50_1200'
NAME = 'stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200'
BASE226 = 'stage2_before_ground_mix226_atprogress_1200'
NAME226 = 'stage2_before_ground_mix226_atprogress_objalias_1200'
RSI_NAME226 = 'stage2_before_ground_mix226_atprogress_objalias_rsi433_1200'
CONFIG_PAIRS = [(BASE, NAME), (BASE226, NAME226)]
MODEL_PAIRS = CONFIG_PAIRS + [(NAME226, RSI_NAME226)]


def packet(kind=3):
    # Ha=0 carries Oa=4 to Ga=12. Hb=1 uses Oa as its support.
    # The other pair is two independent SIT tasks; task objects form a prefix.
    carry = kind == 3
    return torch.tensor([[[1, 2, 0, 4, 12], [1, kind, 1, 5 if carry else 0, 4],
        [1, 0, 2, 0, 6 if carry else 5], [1, 0, 3, 0, 7 if carry else 6]]], dtype=torch.float)


def observations(p):
    poses = torch.randn(len(p), 16, 7)
    poses[..., 3:7] = torch.tensor([0., 0., 0., 1.])
    obs = torch.cat((torch.randn(len(p), 1136), poses.flatten(1), p.flatten(1)), -1)
    dependent, target, alias, _ = alias_layout(p, 4, 8)
    r, f = dependent.nonzero(as_tuple=True)
    objects = obs[:, 892:1132].reshape(-1, 8, 30)
    objects[r, alias[r, f]-4] = objects[r, target[r, f]-4]
    kinematics = obs[:, 1136:1248].reshape(-1, 16, 7)
    kinematics[r, alias[r, f]] = kinematics[r, target[r, f]]
    return obs


@pytest.mark.parametrize('base,name', CONFIG_PAIRS)
def test_config_contract_and_token_budget(base, name):
    old, new = config(base), config(name)
    validate_env(new['env'])
    plain = deepcopy(new['env'])
    assert plain['relationGraph'].pop('before_object_alias') == MODE
    assert plain == old['env'] and new['sim'] == old['sim']
    a, b = config(base, 'train/rlg'), config(name, 'train/rlg')
    b['params']['config']['name'] = a['params']['config']['name']
    assert a == b
    for src, dst in ((old, new), (new, old)):
        with pytest.raises(ValueError, match='BEFORE checkpoint'):
            check_checkpoint({'before_contract': checkpoint_contract(src['env'])}, dst['env'])
    new['env']['numObjects'] = 7
    with pytest.raises(ValueError, match=r'2\*numAgents'):
        validate_env(new['env'])
    new['env']['numObjects'] = 8
    new['env']['relationGraph']['before_object_alias'] = 'unknown'
    with pytest.raises(ValueError, match='before_object_alias'):
        validate_env(new['env'])


def test_mix226_alias_rsi_changes_only_follower_initialization():
    from utils.before_pickup_carry_rsi import SPEC
    old, new = config(NAME226), config(RSI_NAME226)
    validate_env(new['env'])
    plain = deepcopy(new['env'])
    assert plain['before'].pop('follower_pickup_carry_rsi') == SPEC
    assert plain == old['env'] and new['sim'] == old['sim']
    assert new['experiment']['transfer'] == old['experiment']['transfer']
    a, b = config(NAME226, 'train/rlg'), config(RSI_NAME226, 'train/rlg')
    b['params']['config']['name'] = a['params']['config']['name']
    assert a == b
    for other in (NAME226, NAME, BASE, 'stage2_before_ground_mix226_atprogress_nonemask_rsi433_1200'):
        for source, target in ((config(other), new), (new, config(other))):
            with pytest.raises(ValueError, match='BEFORE checkpoint'):
                check_checkpoint({'before_contract': checkpoint_contract(source['env'])}, target['env'])


@pytest.mark.parametrize('kind', [0, 1, 3])
def test_role_edges_physical_identity_and_original_before(kind):
    original = packet(kind)
    saved = original.clone()
    mapping = torch.tensor([[7, 2, 10, 13, 8, 9, 11, 12]])
    replaced, count = replace_distractors(mapping, original, 4)
    dst = 8 if kind == 3 else 7
    assert int(count) == dst-4+1
    assert replaced[0, dst-4] == mapping[0, 0]
    assert len(replaced[0].unique()) == 7
    assert torch.equal(replaced[0, :dst-4], mapping[0, :dst-4])
    assert torch.equal(replaced[0, dst-3:], mapping[0, dst-3:])
    converted = encoder_packet(original, 4, 8)
    assert converted[0, 1, 4] == dst
    valid, src, target, tasks = task_edges(converted)
    edges = {(int(s), int(t)) for s, t in zip(src[0][valid[0]], target[0][valid[0]])}
    expected = {(0, 4), (0, 12), (4, 12), (1, dst)}
    expected |= {(1, 5), (5, dst), (2, 6), (3, 7)} if kind == 3 else {(2, 5), (3, 6)}
    assert edges == expected
    assert torch.equal(original, saved)
    assert torch.equal(converted[..., :4], original[..., :4])
    # Human SA must receive the original packet: the redirected packet loses BEFORE.
    assert human_relations(original, 4, before=True)[0, 1, 0] == 2
    assert human_relations(converted, 4, before=True)[0, 1, 0] == 0
    # Record permutation must not change alias assignment (rank by actor ID).
    order = torch.tensor([3, 1, 0, 2])
    torch.testing.assert_close(encoder_packet(original[:, order], 4, 8), converted[:, order])
    assert torch.equal(replace_distractors(mapping, original[:, order], 4)[0], replaced)


def test_two_pairs_keep_eight_tokens_and_six_physical_objects():
    p = torch.tensor([[[1, 2, 0, 4, 12], [1, 3, 1, 5, 4],
                       [1, 2, 2, 6, 14], [1, 3, 3, 7, 6]]], dtype=torch.float)
    mapping = torch.arange(8)[None]
    result, count = replace_distractors(mapping, p, 4)
    assert result.tolist() == [[0, 1, 2, 3, 0, 2, 6, 7]]
    assert count.tolist() == [6] and len(result[0].unique()) == 6
    assert encoder_packet(p, 4, 8)[0, :, 4].tolist() == [12, 8, 14, 9]
    with pytest.raises(ValueError, match='token budget'):
        replace_distractors(mapping[:, :5], p, 4)


@pytest.mark.parametrize('humans', [1, 2, 3, 4, 6])
def test_random_graphs_preserve_task_objects_with_fixed_budget(humans):
    from utils.before_spec import sample_graph
    from utils.task_role_spec import task_graph_packet
    torch.manual_seed(31)
    env = config(NAME)['env']
    objects = 2*humans
    graph, predecessor, _ = sample_graph(torch.full((64, humans, 3), .4),
        env['box']['build']['taskSizeRanges'], 'random_scenario', objects)
    p = task_graph_packet(graph, 64).reshape(64, humans, 5)
    for row in range(len(p)):
        # Independent reference compaction, using only objects actually named by tasks.
        used = set()
        for valid, task, actor, payload, target in p[row].long().tolist():
            if valid and task >= 2: used.add(payload)
            if valid and humans <= target < humans+objects: used.add(target)
        remap = {old: humans+i for i, old in enumerate(sorted(used))}
        for record in p[row]:
            if record[1] >= 2: record[3] = remap[int(record[3])]
            if int(record[4]) in remap: record[4] = remap[int(record[4])]
    mapping = torch.arange(objects).expand(len(p), -1)
    result, _ = replace_distractors(mapping, p, humans)
    converted = encoder_packet(p, humans, objects)
    for row in range(len(p)):
        assert len(result[row].unique()) == objects-int((predecessor[row] >= 0).sum())
        for record in range(humans):
            old, new = int(p[row, record, 4]), int(converted[row, record, 4])
            if predecessor[row, record] >= 0:
                assert old != new and result[row, old-humans] == result[row, new-humans]
            else:
                assert old == new


@pytest.mark.parametrize('base,name', MODEL_PAIRS)
def test_solo_identity_and_actor_critic_use_separate_encoder_packet(base, name):
    torch.set_num_threads(1)
    old, new = build(base), build(name)
    new.load_state_dict(old.state_dict(), strict=True)
    solo = packet(); solo[0, 1, 4] = 8  # independent ON_TOP support
    obs = observations(solo)
    assert torch.equal(encoder_packet(solo, 4, 8), solo)
    for method in ('eval_actor', 'eval_critic'):
        a, b = getattr(old, method)(obs), getattr(new, method)(obs)
        if isinstance(a, tuple): a, b = a[0], b[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    obs = observations(packet()).requires_grad_()
    saved = obs.detach().clone()
    seen = []
    def encoder_hook(module, args):
        p = args[0][:, -20:].reshape(-1, 4, 5)
        assert p[0, 1, 4] == 8
        assert args[0].shape == obs.shape
        seen.append('encoder')
    def sa_hook(module, args):
        p = args[2].reshape(-1, 4, 5)
        assert p[0, 1, 4] == 4
        assert human_relations(p, 4, before=True)[0, 1, 0] == 2
        seen.append('sa')
    for encoder, sa in ((new.actor_encoder, new.coordination), (new.critic_encoder, new.critic_coordination)):
        encoder.register_forward_pre_hook(encoder_hook)
        sa.register_forward_pre_hook(sa_hook)
        # All scene attention entries stay finite: no new token/padding masks.
        def finite_bias(module, args, output):
            assert torch.isfinite(output[0]).all()
        encoder.context_fusion.register_forward_hook(finite_bias)
    loss = new.eval_actor(obs)[0].square().mean() + new.eval_critic(obs).square().mean()
    loss.backward()
    assert seen == ['encoder', 'sa', 'encoder', 'sa']
    assert torch.equal(saved, obs.detach()) and torch.isfinite(obs.grad).all()


@pytest.mark.parametrize('base,name', MODEL_PAIRS)
def test_stage1_transfer_frozen_encoder_head_updates_and_reload(base, name):
    torch.set_num_threads(1)
    source = torch.nn.Module(); source.a2c_network = build(base)
    state = {k: v for k, v in source.state_dict().items()
             if not k.startswith(('a2c_network.coordination.', 'a2c_network.critic_coordination.'))}
    env = config('stage1_v2_fixed')['env']
    model = torch.nn.Module(); model.a2c_network = build(name)
    report = transfer_stage1(model, dict(model=state, epoch=78000,
        relation_metadata=checkpoint_metadata(env['relationReward']), relation_experiment_config={'env': env}))
    assert set(report['copied']) == set(state)
    frozen = deepcopy(model.a2c_network.actor_encoder.state_dict())
    head = deepcopy(model.a2c_network.action_head.state_dict())
    obs = observations(packet())
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    for _ in range(2):
        optimizer.zero_grad()
        (model.a2c_network.eval_actor(obs)[0].square().mean() + model.a2c_network.eval_critic(obs).square().mean()).backward()
        optimizer.step()
    net = model.a2c_network
    assert all(torch.equal(v, net.actor_encoder.state_dict()[k]) for k, v in frozen.items())
    assert all(not torch.equal(v, net.action_head.state_dict()[k]) for k, v in head.items())
    restored = build(name); restored.load_state_dict(net.state_dict(), strict=True)
    torch.testing.assert_close(net.eval_actor(obs)[0], restored.eval_actor(obs)[0], rtol=0, atol=0)
    torch.testing.assert_close(net.eval_critic(obs), restored.eval_critic(obs), rtol=0, atol=0)
