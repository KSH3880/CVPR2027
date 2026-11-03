"""Post-LN contract for the isolated virtual-asym experiment."""
from copy import deepcopy
import torch
from learning.multi_agent.human_coordination import CoordinationBlock, HumanCoordination
from learning.multi_agent.stage2_push_network import Stage2PushBuilder
from utils.experiments import native_config, validate_experiment

NAME='stage2_before_virtual_asym_postln'
BASE='stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute'

def test_config_preserves_virtual_asym_reward_and_training():
    a,b=native_config(BASE),native_config(NAME)
    b=deepcopy(b);b['env']['numEnvs']=a['env']['numEnvs']
    assert a['env']==b['env'] and a['sim']==b['sim']
    c,d=native_config(BASE,True),native_config(NAME,True)
    d['params']['config']['name']=c['params']['config']['name']
    assert c==d
    validate_experiment(NAME,native_config(NAME)['env'],native_config(NAME,True))

def test_postln_formula_raw_qkv_and_backward():
    torch.manual_seed(88);torch.set_num_threads(1)
    block=CoordinationBlock(16,2,32,4)
    with torch.no_grad():block.relation_bias.normal_()
    x=torch.randn(2,4,16,requires_grad=True)*3+2
    rel=torch.randint(4,(2,4,4));padding=torch.tensor(False)
    seen=[]
    hook=block.attention.register_forward_pre_hook(lambda m,args: seen.append(tuple(t.detach().clone() for t in args)))
    y=block(x,None,rel,padding);hook.remove()
    for t in seen[0]:torch.testing.assert_close(t,x)
    bias=block.effective_relation_bias()[:,rel].permute(1,0,2,3).flatten(0,1)
    context,_=block.attention(x,x,x,attn_mask=bias,need_weights=False)
    h=block.query_norm(x+context)
    reference=block.ffn_norm(h+block.ffn(h))
    torch.testing.assert_close(y,reference)
    (y*torch.randn_like(y)).sum().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in block.parameters())

def test_zero_updates_normalize_instead_of_preserving_input():
    branch=HumanCoordination(width=16,heads=2,num_layers=4,feedforward_dim=32,relations=4)
    branch.init_identity();x=torch.randn(2,4,16)*4+3;rel=torch.ones(2,4,4,dtype=torch.long)
    out=x
    for b in branch.blocks:
        reference=b.ffn_norm(b.query_norm(out))
        out=b(out,None,rel,torch.tensor(False));torch.testing.assert_close(out,reference)
    assert not torch.allclose(out,x)

def test_actor_and_critic_each_have_four_postln_blocks():
    torch.set_num_threads(1)
    cfg=native_config(NAME);builder=Stage2PushBuilder();builder.load(native_config(NAME,True)['params']['network'])
    net=builder.build('amp',actions_num=32,input_shape=(1272,),amp_input_shape=(1330,),value_size=1,
       num_agents=4,num_objects=8,observation_mode='clean_scene',scene_entity_sizes=[223,30,1],
       scene_kinematic_size=7,scene_arena_scale=16.,relation_reward_mode=cfg['env']['relationReward']['mode'],
       relation_graph_spec=cfg['env']['relationGraph'],device='cpu')
    assert net.coordination is not net.critic_coordination
    for branch in (net.coordination,net.critic_coordination):
        assert len(branch.blocks)==4
        for b in branch.blocks:
            assert b.query_norm.normalized_shape==(64,) and b.ffn_norm.normalized_shape==(64,)
            assert b.relation_bias.shape==(2,4)
            x=torch.randn(1,4,64)*3+2
            rel=torch.ones(1,4,4,dtype=torch.long)
            torch.testing.assert_close(b(x,None,rel,torch.tensor(False)),b.ffn_norm(b.query_norm(x)))
