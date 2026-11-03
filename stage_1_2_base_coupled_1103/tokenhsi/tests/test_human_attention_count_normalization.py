from unittest.mock import patch

import pytest
import torch

from learning.multi_agent.human_coordination import CoordinationBlock
from learning.multi_agent.stage2_push_network import transfer_stage1
from test_stage2_push_common import build, _common_observation
from utils.coupled_push_spec import checkpoint_contract, check_checkpoint
from utils.experiments import native_config, validate_experiment

BASE = 'stage2_coupled_push_individual_goal_solo5'
VARIANTS = [(BASE+'_countnorm_coupled', 'coupled'),
            (BASE+'_countnorm_coupled_none', 'coupled_none')]


def probabilities(block, humans, tokens, relations, padding=None):
    seen = []
    original = block.attention.forward

    def capture(*args, **kwargs):
        kwargs.update(need_weights=True, average_attn_weights=False)
        output, weights = original(*args, **kwargs)
        seen.append(weights)
        return output, weights

    with patch.object(block.attention, 'forward', side_effect=capture):
        block(humans, tokens, relations, torch.tensor(False) if padding is None else padding)
    return seen[0]


def group_mass(coupled, none, mode):
    block = CoordinationBlock(8, 2, 16, 4, count_normalization=mode)
    with torch.no_grad():
        block.relation_bias.copy_(torch.tensor([[.7, .2, -.3, 1.3], [-.4, .5, .1, .8]]))
    relation = torch.tensor([[[1]+[3]*coupled+[0]*none]])
    weights = probabilities(block, torch.zeros(1, 1, 8), torch.zeros(1, 1+coupled+none, 8), relation)
    return torch.stack([weights[..., relation[0, 0] == kind].sum(-1) for kind in (1, 3, 0)])


def test_relation_mass_does_not_grow_when_equal_members_are_replicated():
    expected = group_mass(1, 1, 'coupled_none')
    for coupled, none in ((2, 5), (3, 7), (7, 13), (31, 40)):
        torch.testing.assert_close(group_mass(coupled, none, 'coupled_none'), expected)
    assert group_mass(7, 1, 'none')[1].mean() > group_mass(1, 1, 'none')[1].mean()


def test_coupled_only_keeps_none_count_sensitive():
    torch.testing.assert_close(group_mass(1, 2, 'coupled'), group_mass(7, 2, 'coupled'))
    assert group_mass(3, 8, 'coupled')[2].mean() > group_mass(3, 2, 'coupled')[2].mean()


@pytest.mark.parametrize('mode', ['coupled', 'coupled_none'])
def test_count_normalization_preserves_preferences_within_each_relation(mode):
    torch.manual_seed(17)
    original = CoordinationBlock(8, 2, 16, 4)
    normalized = CoordinationBlock(8, 2, 16, 4, count_normalization=mode)
    normalized.load_state_dict(original.state_dict(), strict=True)
    humans, tokens = torch.randn(2, 1, 8), torch.randn(2, 8, 8)
    relation = torch.tensor([[[1, 3, 3, 3, 0, 0, 2, 2]]]).expand(2, -1, -1)
    before = probabilities(original, humans, tokens, relation)
    after = probabilities(normalized, humans, tokens, relation)
    for kind in range(4):
        selected = relation[0, 0] == kind
        a, b = before[..., selected], after[..., selected]
        torch.testing.assert_close(a/a.sum(-1, keepdim=True), b/b.sum(-1, keepdim=True))
    self_key, before_key = relation[0, 0] == 1, relation[0, 0] == 2
    torch.testing.assert_close(before[..., self_key].sum(-1)/before[..., before_key].sum(-1),
                               after[..., self_key].sum(-1)/after[..., before_key].sum(-1))


@pytest.mark.parametrize('mode', ['coupled', 'coupled_none'])
def test_masked_members_do_not_change_visible_relation_mass(mode):
    torch.manual_seed(18)
    block = CoordinationBlock(8, 2, 16, 4, count_normalization=mode)
    query, keys = torch.randn(1, 1, 8), torch.randn(1, 5, 8)
    relation = torch.tensor([[[1, 3, 3, 0, 0]]])
    expected = probabilities(block, query, keys, relation)
    keys = torch.cat((keys, torch.randn(1, 3, 8)), 1)
    relation = torch.cat((relation, torch.tensor([[[3, 3, 0]]])), -1)
    padding = torch.tensor([[[[False]*5+[True]*3]]])
    actual = probabilities(block, query, keys, relation, padding)
    torch.testing.assert_close(actual[..., :5], expected)
    assert not actual[..., 5:].any()


@pytest.mark.parametrize('mode', ['coupled', 'coupled_none'])
@pytest.mark.parametrize('people', [1, 4, 8, 17])
def test_empty_relations_and_shuffling_have_finite_outputs_and_gradients(mode, people):
    torch.manual_seed(19)
    block = CoordinationBlock(8, 2, 16, 4, count_normalization=mode)
    humans = torch.randn(2, people, 8, requires_grad=True)
    relation = torch.eye(people, dtype=torch.long)[None].expand(2, -1, -1)
    before = block(humans, None, relation, torch.tensor(False))
    order = torch.randperm(people)
    after = block(humans[:, order], None, relation[:, order][:, :, order], torch.tensor(False))
    torch.testing.assert_close(before[:, order], after)
    before.square().mean().backward()
    assert torch.isfinite(before).all() and torch.isfinite(humans.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in block.parameters() if p.grad is not None)


@pytest.mark.parametrize('name,mode', VARIANTS)
def test_new_configs_only_change_count_normalization_and_identity(name, mode):
    cfg = native_config(name)
    validate_experiment(name, cfg['env'], native_config(name, True))
    assert cfg['env']['relationGraph'].pop('human_attention_count_normalization') == mode
    cfg['experiment']['name'] = BASE
    assert cfg == native_config(BASE)
    train = native_config(name, True)
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)
    env = native_config(name)['env']
    saved = {'coupled_push_contract': checkpoint_contract(env)}
    check_checkpoint(saved, env)
    for other in [BASE]+[n for n, _ in VARIANTS if n != name]:
        with pytest.raises(ValueError, match='settings differ'):
            check_checkpoint(saved, native_config(other)['env'], inference=True)


@pytest.mark.parametrize('name,mode', VARIANTS)
def test_actor_critic_and_stage1_transfer_use_the_selected_mode(name, mode):
    torch.set_num_threads(1)
    net = build(name)
    for branch in (net.coordination, net.critic_coordination):
        assert len(branch.blocks) == 4
        assert all(block.count_normalization == mode for block in branch.blocks)
        branch.shuffle_human_order = False
    checkpoint = torch.load(native_config(name)['experiment']['transfer']['checkpoint'], map_location='cpu', weights_only=False)
    model = torch.nn.Module()
    model.a2c_network = net
    report = transfer_stage1(model, checkpoint)
    assert report['source_epoch'] == 26000
    assert all(torch.equal(model.state_dict()[key], value) for key, value in checkpoint['model'].items())
    net.eval()
    net.relation_graph_spec['shuffle_token_order'] = False
    observation = _common_observation('coupled')
    saved = observation.clone()
    mu, sigma = net.eval_actor(observation)
    value = net.eval_critic(observation)
    assert torch.isfinite(mu).all() and torch.isfinite(sigma).all() and torch.isfinite(value).all()
    assert torch.equal(observation, saved)


@pytest.mark.parametrize('mode', ['bad', 'divide_logits'])
def test_unknown_count_normalization_is_rejected(mode):
    with pytest.raises(ValueError, match='count normalization'):
        CoordinationBlock(8, 2, 16, 4, count_normalization=mode)
