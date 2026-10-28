from copy import deepcopy

import pytest
import torch

from test_before_adapter import config, build, observations, ADAPTER, BASE
from learning.multi_agent.stage2_network_builder import freeze_actor, transfer_stage1
from utils.before_spec import validate_env, checkpoint_contract
from utils.stage2_spec import check_checkpoint
from utils.relation_task_spec import checkpoint_metadata

NAME = 'stage2_before_ground_mix226_atprogress_adapter_bias0_1200'
PLAIN_NAME = 'stage2_before_ground_mix226_atprogress_bias0_1200'
CASES = [(ADAPTER, NAME), (BASE, PLAIN_NAME)]


@pytest.mark.parametrize('base,name', CASES)
def test_both_bias_config_and_checkpoint_isolation(base, name):
    old, new = config(base), config(name)
    validate_env(new['env'])
    comparable = deepcopy(new['env'])
    for side in ('actor', 'critic'):
        assert comparable['relationGraph'].pop(side+'_human_bias') == 'before_only'
    assert comparable == old['env'] and new['sim'] == old['sim']
    assert new['experiment']['transfer']['checkpoint'].endswith('00078000.pth')
    a, b = config(base, 'train/rlg'), config(name, 'train/rlg')
    b['params']['config']['name'] = a['params']['config']['name']
    assert a == b
    for other in (old['env'], config('stage2_before_ground_mix226_atprogress_actorbias0_1200')['env'],
                  config(PLAIN_NAME if name == NAME else NAME)['env']):
        for src, dst in ((other,new['env']), (new['env'],other)):
            with pytest.raises(ValueError, match='BEFORE checkpoint'):
                check_checkpoint({'before_contract': checkpoint_contract(src)}, dst)
    actor_only = deepcopy(new['env']); actor_only['relationGraph'].pop('critic_human_bias')
    with pytest.raises(ValueError, match='BEFORE checkpoint'):
        check_checkpoint({'before_contract': checkpoint_contract(actor_only)}, new['env'])
    explicit = deepcopy(old['env']); explicit['relationGraph']['critic_human_bias'] = 'all'
    assert checkpoint_contract(explicit) == checkpoint_contract(old['env'])
    explicit['relationGraph']['critic_human_bias'] = 'invalid'
    with pytest.raises(ValueError, match='critic_human_bias'):
        validate_env(explicit)


@pytest.mark.parametrize('base,name', CASES)
def test_actor_critic_fixed_zero_gradients_head_freeze_and_reload(base, name):
    torch.set_num_threads(1)
    old, net = build(base), build(name)
    has_adapter = base == ADAPTER
    assert hasattr(net, 'action_adapters') == has_adapter
    for key,value in old.state_dict().items():
        if key.endswith('relation_bias') and key.startswith(('coordination.', 'critic_coordination.')):
            assert value.shape == (2,3) and net.state_dict()[key].shape == (2,1)
        else:
            torch.testing.assert_close(value,net.state_dict()[key],rtol=0,atol=0)
    obs = observations()
    torch.testing.assert_close(net.eval_actor(obs)[0],old.eval_actor(obs)[0],rtol=0,atol=0)
    torch.testing.assert_close(net.eval_critic(obs),old.eval_critic(obs),rtol=0,atol=0)
    model = torch.nn.Module(); model.a2c_network = net; freeze_actor(model)
    frozen = {k:v.clone() for k,v in net.state_dict().items()
              if k.startswith('actor_encoder.') or (has_adapter and k.startswith('action_head.'))}
    head_before = {k:v.clone() for k,v in net.action_head.state_dict().items()}
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=2e-5)
    for step in range(3):
        opt.zero_grad()
        (net.eval_actor(obs)[0].square().mean()+net.eval_critic(obs).square().mean()).backward()
        for branch in (net.coordination,net.critic_coordination):
            for block in branch.blocks:
                assert block.relation_bias.shape == (2,1)
                assert torch.count_nonzero(block.effective_relation_bias()[:,:2]) == 0
                if step:
                    assert block.relation_bias.grad.abs().sum() > 0
                    assert block.attention.in_proj_weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in net.actor_encoder.parameters())
        if has_adapter:
            assert all(p.grad is None for p in net.action_head.parameters())
            assert all(p.grad.abs().sum() > 0 for p in net.action_adapters.parameters())
        else:
            assert all(p.requires_grad and p.grad.abs().sum() > 0 for p in net.action_head.parameters())
        opt.step()
    for k,v in frozen.items():torch.testing.assert_close(net.state_dict()[k],v,rtol=0,atol=0)
    if not has_adapter:
        assert all(not torch.equal(v, net.action_head.state_dict()[k]) for k,v in head_before.items())
    restored = build(name); restored.load_state_dict(net.state_dict(),strict=True)
    for branch in (restored.coordination,restored.critic_coordination):
        for block in branch.blocks:
            assert torch.count_nonzero(block.effective_relation_bias()[:,:2]) == 0
            assert torch.count_nonzero(block.relation_bias) > 0
    torch.testing.assert_close(restored.eval_actor(obs)[0],net.eval_actor(obs)[0],rtol=0,atol=0)
    torch.testing.assert_close(restored.eval_critic(obs),net.eval_critic(obs),rtol=0,atol=0)


@pytest.mark.parametrize('base,name', CASES)
def test_stage1_transfer_with_both_bias_tables_restricted(base, name):
    source = torch.nn.Module(); source.a2c_network = build(ADAPTER)
    state = {k:v for k,v in source.state_dict().items() if not k.startswith((
        'a2c_network.coordination.', 'a2c_network.critic_coordination.', 'a2c_network.action_adapters.'))}
    env = config('stage1_v2_fixed')['env']
    target = torch.nn.Module(); target.a2c_network = build(name)
    report = transfer_stage1(target,dict(model=state,epoch=78000,
        relation_metadata=checkpoint_metadata(env['relationReward']),relation_experiment_config={'env':env}))
    assert set(report['copied']) == set(state)
    for k,v in state.items():torch.testing.assert_close(target.state_dict()[k],v,rtol=0,atol=0)
    assert all(p.requires_grad == (base == BASE) for p in target.a2c_network.action_head.parameters())
    assert all(not p.requires_grad for p in target.a2c_network.actor_encoder.parameters())
    if base == ADAPTER:
        assert all(torch.count_nonzero(p) == 0 for p in target.a2c_network.action_adapters.parameters())
    else:
        assert not hasattr(target.a2c_network, 'action_adapters')
