from pathlib import Path
import pytest
import torch
from learning.multi_agent.push_at_coordination import PushAtCoordination, attach_push_at_coordination
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder, EdgeEncoder

torch.set_num_threads(1)
TYPES=torch.tensor([0,0,1,1,1,1,2,2])
def packet():
    return torch.tensor([[[1,0,4,12,0],[1,4,6,7,0],[1,1,4,12,1],[1,4,6,7,1]]],dtype=torch.float).flatten(1)

def test_bundles_order_and_both_relations():
    ca=PushAtCoordination();edge=EdgeEncoder(4,2,num_relation_types=13);nodes=torch.randn(1,8,64)
    p=packet();token,_,bundle=ca.task_tokens(nodes,p,edge,TYPES)
    assert bundle.tolist()==[[[1,0,4,6],[1,1,4,6]]]
    shuffled=p.reshape(1,4,5)[:,[3,1,0,2]].flatten(1)
    torch.testing.assert_close(token,ca.task_tokens(nodes,shuffled,edge,TYPES)[0])
    for relation in (7,12):
        with torch.no_grad():edge.relation_embed.weight[relation].add_(3)
        changed=ca.task_tokens(nodes,p,edge,TYPES)[0]
        assert not torch.allclose(token,changed);token=changed

def test_missing_at_empty_and_permutation_gradients():
    ca=PushAtCoordination();edge=EdgeEncoder(4,2,num_relation_types=13);nodes=torch.randn(1,8,64)
    p=packet().reshape(1,4,5);p[:,1,0]=0
    with pytest.raises(ValueError,match='matching'):ca.bundles(p.flatten(1),2)
    empty=ca(nodes[:,:2],nodes,torch.zeros_like(packet()),edge,TYPES)
    assert torch.isfinite(empty).all() and torch.count_nonzero(empty)==0
    with torch.no_grad():
        for block in ca.blocks:block.relation_bias[:,2]=.7
    p=packet();out=ca(nodes[:,:2],nodes,p,edge,TYPES)
    perm=torch.tensor([1,0,2,3,4,5,6,7]);swapped=p.reshape(1,4,5).clone()
    swapped[...,1]=perm[swapped[...,1].long()];swapped[...,2]=perm[swapped[...,2].long()];swapped[...,4]=1-swapped[...,4]
    result=ca(nodes[:,perm][:,:2],nodes[:,perm],swapped.flatten(1),edge,TYPES)
    torch.testing.assert_close(result,out[:,[1,0]],atol=2e-6,rtol=1e-5)
    out.square().mean().backward()
    assert ca.grounding[0].weight.grad.abs().sum()>0
    assert all(b.attention.in_proj_weight.grad.abs().sum()>0 for b in ca.blocks)
    assert edge.relation_embed.weight.grad is None

def test_real_checkpoint_initial_actions_and_gradients():
    root=Path(__file__).resolve().parents[5]
    weights=torch.load(root/'runs/jointpush/push_stage1_repro_20261012/checkpoints/OpenPushPush180ContactAtWarmstart_00029000.pth',map_location='cpu',weights_only=False)
    snapshot=weights['relation_experiment_config'];builder=AMPMultiAgentBuilder();builder.load(snapshot['train']['params']['network'])
    amp_width=weights['model']['a2c_network._disc_mlp.0.weight'].shape[1]
    network=builder.build('push_test',actions_num=32,input_shape=(684,),amp_input_shape=(amp_width,),value_size=1,
        num_agents=2,num_objects=4,observation_mode='clean_scene',scene_entity_sizes=[223,40,1],scene_kinematic_size=7,
        scene_arena_scale=14.,relation_reward_mode=snapshot['env']['relationReward']['mode'],relation_graph_spec=snapshot['env']['relationGraph'],device='cpu')
    network.load_state_dict({k[len('a2c_network.'):]:v for k,v in weights['model'].items()},strict=True)
    network.eval();obs=torch.randn(2,684)*.1;kin=obs[:,608:664].reshape(2,8,7);kin[...,3:7]=0;kin[...,6]=1;obs[:,-20:]=packet().expand(2,-1)
    with torch.no_grad():before=network.eval_actor(obs)[0].clone()
    attach_push_at_coordination(network);after=network.eval_actor(obs)[0]
    torch.testing.assert_close(after,before,atol=1e-6,rtol=1e-5)
    assert all(not p.requires_grad for p in network.actor_encoder.parameters())
    after.square().mean().backward();assert network.action_head[0][0].weight.grad[:,64:].abs().sum()>0
    optimizer=torch.optim.SGD((p for p in network.parameters() if p.requires_grad),lr=.001);optimizer.step();optimizer.zero_grad()
    network.eval_actor(obs)[0].square().mean().backward();assert network.push_at_coordination.grounding[0].weight.grad.abs().sum()>0
    assert any(k.startswith('push_at_coordination.') for k in network.state_dict())
