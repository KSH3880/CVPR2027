import pytest
import torch

from depbias_two_person import TwoPersonShuffle
from test_depbias_ablation import fixture


def test_two_person_policy_rng_and_weights_are_identical():
    network, humans, packet, groups = fixture()
    humans, packet, groups = humans[:, :2], packet[:, :2], groups[:, :2]
    modules = (network.coordination, network.critic_coordination)
    before = {k:v.clone() for k,v in network.state_dict().items()}
    rng = torch.get_rng_state()
    expected = [module(humans, None, packet, None, groups=groups) for module in modules]
    expected_rng = torch.get_rng_state()
    handler = TwoPersonShuffle(network, 'shuffled', 100001)
    handler.enabled = True
    torch.set_rng_state(rng)
    actual = [module(humans, None, packet, None, groups=groups) for module in modules]
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))
    assert torch.equal(expected_rng, torch.get_rng_state())
    proof = handler.finish(1)
    assert proof['degenerate_two_person_shuffle']
    assert len(proof['calls']) == 8
    assert all(v['changed_entries'] == 0 for v in proof['checks'].values())
    assert all(torch.equal(v,network.state_dict()[k]) for k,v in before.items())


def test_two_person_arbitrary_bias_and_padding_are_identical():
    network, *_ = fixture()
    handler = TwoPersonShuffle(network, 'shuffled', 100007)
    bias = torch.tensor([[[[1.,2.],[3.,4.]],[[5.,6.],[7.,8.]]]])
    bias[:, :, 1, 0] = float('-inf')
    order = torch.tensor([1,0])
    for _ in range(3):
        assert torch.equal(handler.transform(bias,order,'test'),bias)


def test_identity_exception_does_not_accept_larger_scenes_or_missing_calls():
    network, *_ = fixture()
    handler = TwoPersonShuffle(network, 'shuffled', 100001)
    with pytest.raises(ValueError, match='two-person'):
        handler.transform(torch.zeros(1,1,4,4),torch.arange(4),'test')
    with pytest.raises(ValueError, match='missing'):
        handler.finish(1)
