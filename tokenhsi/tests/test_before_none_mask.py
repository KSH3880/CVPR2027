from copy import deepcopy

import pytest
import torch

from test_before_adapter import config, build, observations, BASE
from learning.multi_agent.human_coordination import CoordinationBlock, HumanCoordination
from learning.multi_agent.stage2_network_builder import freeze_actor, transfer_stage1
from utils.before_spec import validate_env, checkpoint_contract
from utils.stage2_spec import check_checkpoint
from utils.relation_task_spec import checkpoint_metadata

NAME = 'stage2_before_ground_mix226_atprogress_nonemask_1200'
BIAS0 = 'stage2_before_ground_mix226_atprogress_bias0_1200'
MODE = 'before_only_mask_none'


def test_config_and_same_shape_checkpoint_isolation():
    old, new = config(BIAS0), config(NAME)
    validate_env(new['env'])
    comparable = deepcopy(new['env'])
    for side in ('actor', 'critic'):
        assert comparable['relationGraph'][side+'_human_bias'] == MODE
        comparable['relationGraph'][side+'_human_bias'] = 'before_only'
    assert comparable == old['env'] and new['sim'] == old['sim']
    assert new['experiment']['transfer']['checkpoint'].endswith('00078000.pth')
    a, b = config(BIAS0, 'train/rlg'), config(NAME, 'train/rlg')
    b['params']['config']['name'] = a['params']['config']['name']
    assert a == b
    for other in (BASE, BIAS0, 'stage2_before_ground_mix226_atprogress_actorbias0_1200',
                  'stage2_before_ground_mix226_atprogress_adapter_bias0_1200'):
        for src,dst in ((config(other)['env'],new['env']), (new['env'],config(other)['env'])):
            with pytest.raises(ValueError, match='BEFORE checkpoint'):
                check_checkpoint({'before_contract':checkpoint_contract(src)},dst)
    with pytest.raises(ValueError, match='BEFORE coordination'):
        HumanCoordination(before=False,bias_mode=MODE)


def test_none_cannot_affect_output_or_gradient_and_self_only_has_no_qk_gradient():
    torch.manual_seed(28);torch.set_num_threads(1)
    block = CoordinationBlock(16,2,32,3,bias_mode=MODE)
    h = torch.randn(2,4,16,requires_grad=True)
    rel = torch.eye(4,dtype=torch.long).expand(2,-1,-1).clone()
    rel[:,1,0]=2;rel[:,3,2]=2
    seen=[]
    def audit(module,inputs,kwargs):
        q,k,v=inputs;mask=kwargs['attn_mask'].reshape(2,2,4,4)
        ids=rel[:,None].expand_as(mask)
        assert torch.isneginf(mask[ids==0]).all()
        assert (mask[ids==1]==0).all() and torch.isfinite(mask[ids==2]).all()
        # Reconstruct real attention probabilities without recursively calling the module.
        q,k,_=torch.nn.functional.linear(q,module.in_proj_weight,module.in_proj_bias).chunk(3,-1)
        q=q.reshape(2,4,2,8).transpose(1,2);k=k.reshape(2,4,2,8).transpose(1,2)
        prob=((q@k.transpose(-1,-2))/(8**.5)+mask).softmax(-1)
        assert (prob[ids==0]==0).all() and torch.isfinite(prob).all()
        assert (prob[:,:,0,0]==1).all()
        seen.append(True)
    handle=block.attention.register_forward_pre_hook(audit,with_kwargs=True)
    out=block(h,None,rel,torch.tensor(False))
    out[:,1].square().mean().backward()
    assert h.grad[:,:2].abs().sum()>0 and torch.count_nonzero(h.grad[:,2:])==0
    assert block.relation_bias.grad.abs().sum()>0
    assert block.attention.in_proj_weight.grad[:32].abs().sum()>0
    changed=h.detach().clone();changed[:,2:]=torch.randn_like(changed[:,2:])*20
    torch.testing.assert_close(out[:,:2],block(changed,None,rel,torch.tensor(False))[:,:2],rtol=0,atol=0)
    block.zero_grad();h.grad=None
    block(h,None,rel,torch.tensor(False))[:,0].square().mean().backward()
    assert torch.count_nonzero(block.attention.in_proj_weight.grad[:32])==0
    assert block.attention.in_proj_weight.grad[32:].abs().sum()>0
    assert torch.count_nonzero(h.grad[:,1:])==0
    assert seen;handle.remove()


def test_actor_critic_match_explicit_mask_train_and_reload():
    torch.set_num_threads(1)
    net, reference=build(NAME),build(BIAS0)
    assert not hasattr(net,'action_adapters')
    # Exercise nonzero SA updates, not just the identity initialization.
    for branch in (net.coordination,net.critic_coordination):
        for block in branch.blocks:
            assert block.bias_mode==MODE and block.relation_bias.shape==(2,1)
            torch.nn.init.normal_(block.attention.out_proj.weight,std=.02)
            torch.nn.init.normal_(block.ffn[-1].weight,std=.02)
    reference.load_state_dict(net.state_dict(),strict=True)
    def explicit_mask(module,inputs):
        h,t,r,p=inputs
        return h,t,r,p|(r==0)[:,None]
    for branch in (reference.coordination,reference.critic_coordination):
        for block in branch.blocks:block.register_forward_pre_hook(explicit_mask)
    obs=observations()
    torch.testing.assert_close(net.eval_actor(obs)[0],reference.eval_actor(obs)[0],rtol=0,atol=0)
    torch.testing.assert_close(net.eval_critic(obs),reference.eval_critic(obs),rtol=0,atol=0)
    # Human shuffling must permute the mask with the humans and undo the permutation.
    for branch in (net.coordination,net.critic_coordination):
        h=torch.randn(2,4,64);packet=obs[:,-20:]
        plain=branch(h,None,packet,None)
        branch.shuffle_human_order=True
        torch.testing.assert_close(branch(h,None,packet,None),plain,rtol=1e-5,atol=1e-6)
        branch.shuffle_human_order=False
    model=torch.nn.Module();model.a2c_network=net;freeze_actor(model)
    frozen={k:v.clone() for k,v in net.actor_encoder.state_dict().items()}
    head={k:v.clone() for k,v in net.action_head.state_dict().items()}
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=2e-5)
    for _ in range(3):
        opt.zero_grad()
        (net.eval_actor(obs)[0].square().mean()+net.eval_critic(obs).square().mean()).backward()
        for branch in (net.coordination,net.critic_coordination):
            for block in branch.blocks:
                assert block.relation_bias.grad.abs().sum()>0
                assert block.attention.in_proj_weight.grad[:128].abs().sum()>0
                assert torch.count_nonzero(block.effective_relation_bias()[:,:2])==0
        assert all(p.grad is None for p in net.actor_encoder.parameters())
        assert all(p.requires_grad and p.grad.abs().sum()>0 for p in net.action_head.parameters())
        opt.step()
    assert all(torch.equal(v,net.actor_encoder.state_dict()[k]) for k,v in frozen.items())
    assert all(not torch.equal(v,net.action_head.state_dict()[k]) for k,v in head.items())
    restored=build(NAME);restored.load_state_dict(net.state_dict(),strict=True)
    torch.testing.assert_close(net.eval_actor(obs)[0],restored.eval_actor(obs)[0],rtol=0,atol=0)
    torch.testing.assert_close(net.eval_critic(obs),restored.eval_critic(obs),rtol=0,atol=0)


def test_stage1_transfer_identity_and_head_training():
    old=build(BASE);source=torch.nn.Module();source.a2c_network=old
    state={k:v for k,v in source.state_dict().items() if not k.startswith((
        'a2c_network.coordination.','a2c_network.critic_coordination.'))}
    env=config('stage1_v2_fixed')['env']
    target=torch.nn.Module();target.a2c_network=build(NAME)
    report=transfer_stage1(target,dict(model=state,epoch=78000,
        relation_metadata=checkpoint_metadata(env['relationReward']),relation_experiment_config={'env':env}))
    assert set(report['copied'])==set(state)
    for k,v in state.items():torch.testing.assert_close(v,target.state_dict()[k],rtol=0,atol=0)
    obs=observations()
    torch.testing.assert_close(old.eval_actor(obs)[0],target.a2c_network.eval_actor(obs)[0],rtol=0,atol=0)
    torch.testing.assert_close(old.eval_critic(obs),target.a2c_network.eval_critic(obs),rtol=0,atol=0)
    assert all(not p.requires_grad for p in target.a2c_network.actor_encoder.parameters())
    assert all(p.requires_grad for p in target.a2c_network.action_head.parameters())
