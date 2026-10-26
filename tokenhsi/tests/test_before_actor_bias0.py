from copy import deepcopy
from pathlib import Path
import io

import pytest
import torch
import yaml

from learning.multi_agent.human_coordination import CoordinationBlock, HumanCoordination
from learning.multi_agent.before_network import BeforeBuilder
from learning.multi_agent.stage2_network_builder import freeze_actor
from utils.before_spec import validate_env, checkpoint_contract, sample_graph
from utils.stage2_spec import check_checkpoint
from utils.task_role_spec import task_graph_packet

ROOT = Path(__file__).resolve().parents[2]
NAMES = [('stage2_before_ground_mix226_1200', 'stage2_before_ground_mix226_actorbias0_1200'),
         ('stage2_before_ground_mix226_atprogress_1200', 'stage2_before_ground_mix226_atprogress_actorbias0_1200')]


def config(name, folder='multi_agent'):
    return yaml.safe_load((ROOT/f'tokenhsi/data/cfg/{folder}/{name}.yaml').read_text())


@pytest.mark.parametrize('old,new', NAMES)
def test_configs_and_checkpoint_contracts(old, new):
    a, b = config(old), config(new)
    validate_env(b['env'])
    env = deepcopy(b['env'])
    assert env['relationGraph'].pop('actor_human_bias') == 'before_only'
    assert env == a['env']
    assert a['sim'] == b['sim']
    assert b['experiment']['transfer']['checkpoint'].endswith('00078000.pth')
    train_old, train_new = config(old, 'train/rlg'), config(new, 'train/rlg')
    train_new['params']['config']['name'] = train_old['params']['config']['name']
    assert train_old == train_new
    check_checkpoint({'before_contract': checkpoint_contract(b['env'])}, b['env'])
    for src, dst in [(a['env'], b['env']), (b['env'], a['env'])]:
        with pytest.raises(ValueError, match='BEFORE checkpoint'):
            check_checkpoint({'before_contract': checkpoint_contract(src)}, dst)
    legacy = checkpoint_contract(a['env'])
    assert set(legacy) == {'version', 'before', 'template_rsi', 'sizes', 'reset'}
    explicit = deepcopy(a['env']); explicit['relationGraph']['actor_human_bias'] = 'all'
    assert checkpoint_contract(explicit) == legacy
    invalid = deepcopy(b['env']); invalid['relationGraph']['actor_human_bias'] = 'unknown'
    with pytest.raises(ValueError, match='actor_human_bias'):
        validate_env(invalid)


def test_zero_bias_is_structural_and_before_bias_receives_gradients():
    torch.manual_seed(26)
    block = CoordinationBlock(16, 2, 32, 3, bias_mode='before_only')
    assert block.relation_bias.shape == (2, 1)
    humans = torch.randn(3, 4, 16, requires_grad=True)
    relations = torch.eye(4, dtype=torch.long).expand(3, -1, -1).clone()
    relations[:, 1, 0] = 2
    padding = torch.tensor(False)
    target = torch.randn_like(humans)
    opt = torch.optim.Adam(block.parameters(), lr=.02)
    before = block.relation_bias.detach().clone()
    for _ in range(4):
        opt.zero_grad()
        ((block(humans, None, relations, padding)-target)**2).mean().backward()
        assert block.relation_bias.grad.abs().sum() > 0
        assert block.attention.in_proj_weight.grad.abs().sum() > 0
        opt.step()
        assert torch.count_nonzero(block.effective_relation_bias()[:, :2]) == 0
    assert not torch.equal(before, block.relation_bias)
    assert humans.grad.abs().sum() > 0
    # Reload preserves the restriction; a legacy full table cannot be loaded silently.
    buffer = io.BytesIO(); torch.save(block.state_dict(), buffer); buffer.seek(0)
    copy = CoordinationBlock(16, 2, 32, 3, bias_mode='before_only')
    copy.load_state_dict(torch.load(buffer, weights_only=False))
    torch.testing.assert_close(copy(humans, None, relations, padding), block(humans, None, relations, padding))
    legacy = CoordinationBlock(16, 2, 32, 3)
    with pytest.raises(RuntimeError, match='size mismatch'):
        legacy.load_state_dict(block.state_dict())


def test_effective_attention_equals_full_table_with_self_none_zero():
    torch.manual_seed(7)
    constrained = CoordinationBlock(16, 2, 32, 3, bias_mode='before_only')
    with torch.no_grad(): constrained.relation_bias.copy_(torch.tensor([[.7], [-.2]]))
    full = CoordinationBlock(16, 2, 32, 3)
    state = deepcopy(constrained.state_dict())
    state['relation_bias'] = constrained.effective_relation_bias().detach().clone()
    full.load_state_dict(state)
    x = torch.randn(2, 4, 16)
    relations = torch.eye(4, dtype=torch.long).expand(2, -1, -1).clone()
    relations[:, 2, 0] = 2
    torch.testing.assert_close(constrained(x, None, relations, torch.tensor(False)),
                               full(x, None, relations, torch.tensor(False)), rtol=0, atol=0)
    with pytest.raises(ValueError, match='BEFORE coordination'):
        HumanCoordination(before=False, bias_mode='before_only')


@pytest.mark.parametrize('old,new', NAMES)
def test_actor_only_wiring_and_unchanged_initial_forward(old, new):
    torch.set_num_threads(1)
    def build(name):
        cfg = config(name); env = cfg['env']
        torch.manual_seed(26)
        builder = BeforeBuilder(); builder.load(config(name, 'train/rlg')['params']['network'])
        return builder.build('amp', actions_num=32, input_shape=(1268,), amp_input_shape=(1320,),
            value_size=1, num_agents=4, num_objects=8, observation_mode='clean_scene',
            scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7, scene_arena_scale=16.,
            relation_reward_mode=env['relationReward']['mode'], relation_graph_spec=env['relationGraph'], device='cpu')
    baseline, net = build(old), build(new)
    for a, c in zip(net.coordination.blocks, net.critic_coordination.blocks):
        assert a.relation_bias.shape == (2, 1)
        assert c.relation_bias.shape == (2, 3)
    for key,value in baseline.state_dict().items():
        if key.startswith('coordination.') and key.endswith('relation_bias'): continue
        torch.testing.assert_close(value, net.state_dict()[key], rtol=0, atol=0)
    env = config(new)['env']
    g,_,_ = sample_graph(torch.full((2,4,3),.4), env['box']['build']['taskSizeRanges'], 'before_stack')
    poses = torch.zeros(2,16,7); poses[...,6] = 1
    obs = torch.cat((torch.randn(2,1136), poses.flatten(1), task_graph_packet(g,2)), -1)
    for n in (baseline,net):
        n.actor_encoder.relation_graph_spec['shuffle_token_order'] = False
        n.coordination.shuffle_human_order = False
        n.critic_coordination.shuffle_human_order = False
    torch.testing.assert_close(net.eval_actor(obs)[0], baseline.eval_actor(obs)[0], rtol=0, atol=0)
    torch.testing.assert_close(net.eval_critic(obs), baseline.eval_critic(obs), rtol=0, atol=0)
    model = torch.nn.Module(); model.a2c_network = net
    freeze_actor(model)
    (net.eval_actor(obs)[0].square().mean()+net.eval_critic(obs).square().mean()).backward()
    assert all(p.grad is None for p in net.actor_encoder.parameters())
    assert net.coordination.blocks[0].attention.out_proj.weight.grad.abs().sum() > 0
    assert net.action_head[-1].weight.grad.abs().sum() > 0
