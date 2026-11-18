from unittest.mock import patch

import pytest
import torch

from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from learning.multi_agent.stage2_push_network import transfer_stage1
from test_stage2_push_common import build, _common_observation
from utils.coupled_push_peer_selection import SingleCoupledPeerSelection
from utils.coupled_push_spec import checkpoint_contract, check_checkpoint
from utils.experiments import native_config, validate_experiment

BASE = 'stage2_coupled_push_individual_goal_solo5'
MODES = ('keep1', 'shrink1')


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('people', [1, 2, 3, 4, 6, 8, 13])
def test_one_peer_readiness_transition_and_global_rng(mode, people):
    groups = torch.zeros(2, people, dtype=torch.long)
    ready = torch.zeros_like(groups, dtype=torch.bool)
    selector = SingleCoupledPeerSelection(mode, groups, 82)
    rng = torch.get_rng_state().clone()
    first = selector.update(groups, ready).clone()
    assert torch.equal(rng, torch.get_rng_state())
    if people == 1:
        assert (first == -1).all()
        return
    assert ((first >= 0) & (first < people)).all()
    assert not (first == torch.arange(people)).any()
    private_rng = selector.generator.get_state().clone()
    assert torch.equal(selector.update(groups, ready), first)
    assert torch.equal(private_rng, selector.generator.get_state())
    ready[:] = True
    final = selector.update(groups, ready).clone()
    assert torch.equal(final, first) if mode == 'keep1' else (final == -1).all()
    ready[:, 0] = False
    again = selector.update(groups, ready)
    assert (again[:, 1:] == 0).all()
    assert (again[:, 0] >= 0).all() if mode == 'keep1' else (again[:, 0] == -1).all()


@pytest.mark.parametrize('mode', MODES)
def test_single_peer_changes_on_contact_and_isolates_partial_resets(mode):
    groups = torch.tensor([[0, 0, 0, 0, 4], [0, 0, 2, 2, 4], [0, 0, 0, 0, 4]])
    ready = torch.zeros_like(groups, dtype=torch.bool)
    selector = SingleCoupledPeerSelection(mode, groups, 83)
    initial = selector.update(groups, ready).clone()
    assert (initial[:, 4] == -1).all()
    assert initial[1].tolist() == [1, 0, 3, 2, -1]
    selected = initial[0, 0]
    ready[0, selected] = True
    ids = torch.tensor([0])
    changed = selector.update(groups[ids], ready[ids], ids)
    assert changed[0, 0] != selected
    assert not ready[0, changed[0, 0]]
    assert torch.equal(selector.peers[1:], initial[1:])
    selector.reset(torch.tensor([1]))
    assert (selector.peers[1] == -1).all()
    assert torch.equal(selector.peers[2], initial[2])
    before = selector.peers.clone()
    selector.update(groups[1:2], ready[1:2], torch.tensor([1]))
    assert torch.equal(selector.peers[[0, 2]], before[[0, 2]])


@pytest.mark.parametrize('mode', MODES)
def test_configs_only_change_identity_and_peer_selection(mode):
    name = BASE+'_'+mode
    config = native_config(name)
    validate_experiment(name, config['env'], native_config(name, True))
    assert config['env']['relationGraph'].pop('coupled_peer_selection') == mode
    config['experiment']['name'] = BASE
    assert config == native_config(BASE)
    train = native_config(name, True)
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)
    env = native_config(name)['env']
    saved = {'coupled_push_contract': checkpoint_contract(env)}
    check_checkpoint(saved, env)
    for other in (BASE, BASE+'_'+('shrink1' if mode == 'keep1' else 'keep1')):
        with pytest.raises(ValueError, match='settings differ'):
            check_checkpoint(saved, native_config(other)['env'], inference=True)


@pytest.mark.parametrize('mode', MODES)
def test_stage1_transfer_normalization_and_actor_critic_four_layer_masks(mode):
    torch.set_num_threads(1)
    net = build(BASE+'_'+mode)
    model = torch.nn.Module()
    model.a2c_network = net
    checkpoint = torch.load(native_config(BASE)['experiment']['transfer']['checkpoint'], map_location='cpu', weights_only=False)
    assert transfer_stage1(model, checkpoint)['source_epoch'] == 26000
    assert all(torch.equal(model.state_dict()[key], value) for key, value in checkpoint['model'].items())
    observation = _common_observation('coupled')
    peers = torch.tensor([[1., 2., 3., 0.]]) if mode == 'keep1' else torch.full((1, 4), -1.)
    observation = torch.cat((observation[:, :-4], peers, observation[:, -4:]), -1)
    normalizer = SceneRunningMeanStd([223, 30, 1], [4, 8, 4], [223, 30, 0], 7, 28)
    normalizer.load_state_dict(checkpoint['running_mean_std'], strict=True)
    normalizer.eval()
    observation = normalizer(observation)
    assert observation.shape == (1, 1276)
    assert torch.equal(observation[:, -8:-4], peers)
    net.eval()
    net.relation_graph_spec['shuffle_token_order'] = False
    seen = []
    permutation = torch.tensor([2, 0, 3, 1])
    relation = (peers[..., None] == torch.arange(4)).long()*3
    relation[:, torch.arange(4), torch.arange(4)] = 1
    relation = relation[:, permutation][:, :, permutation]
    for branch in (net.coordination, net.critic_coordination):
        with torch.no_grad():
            for block in branch.blocks:
                block.relation_bias.copy_(torch.tensor([[.1, .2, .3, .4], [.5, .6, .7, .8]]))

    def audit(module, args, kwargs):
        actual = kwargs['attn_mask'].reshape(1, 2, 4, 4)
        expected = torch.tensor([[.1, .2, .3, .4], [.5, .6, .7, .8]])[:, relation].permute(1, 0, 2, 3)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        seen.append(True)

    handles = [block.attention.register_forward_pre_hook(audit, with_kwargs=True)
        for branch in (net.coordination, net.critic_coordination) for block in branch.blocks]
    with patch('torch.randperm', return_value=permutation):
        mean, sigma = net.eval_actor(observation)
        value = net.eval_critic(observation)
    for handle in handles:
        handle.remove()
    assert len(seen) == 8
    assert all(torch.isfinite(item).all() for item in (mean, sigma, value))


@pytest.mark.parametrize('mode', MODES)
def test_ppo_replays_saved_peer_choices_independent_of_minibatch_order(mode):
    torch.set_num_threads(1)
    net = build(BASE+'_'+mode)
    net.eval()
    net.relation_graph_spec['shuffle_token_order'] = False
    for branch in (net.coordination, net.critic_coordination):
        branch.shuffle_human_order = False
        with torch.no_grad():
            for block in branch.blocks:
                block.attention.out_proj.weight.normal_(std=.02)
                block.ffn[-1].weight.normal_(std=.02)
    scene = _common_observation('coupled').expand(3, -1)
    peers = torch.tensor([[1., 2., 3., 0.], [2., 3., 0., 1.], [-1., 0., 1., 2.]])
    obs = torch.cat((scene[:, :-4], peers, scene[:, -4:]), -1)
    with torch.no_grad():
        expected_mu = net.eval_actor(obs)[0].reshape(3, 4, -1)
        expected_value = net.eval_critic(obs).reshape(3, 4, -1)
        for indices in (torch.tensor([2, 0]), torch.tensor([1])):
            actual_mu = net.eval_actor(obs[indices])[0].reshape(len(indices), 4, -1)
            actual_value = net.eval_critic(obs[indices]).reshape(len(indices), 4, -1)
            torch.testing.assert_close(actual_mu, expected_mu[indices], atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(actual_value, expected_value[indices], atol=2e-6, rtol=1e-5)
    net.zero_grad()
    net.eval_actor(obs)[0].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.coordination.parameters())
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
