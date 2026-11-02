import torch
from learning.multi_agent.attention_viewer import AttentionProbe, canonical_attention, human_attention
from before_test_utils import build, observations


def test_probe_preserves_actions_rng_weights_and_normalized_probabilities():
    torch.set_num_threads(1)
    net=build('stage2_before').eval();obs=observations()
    net.actor_encoder.relation_graph_spec['shuffle_token_order']=True
    net.coordination.shuffle_human_order=True
    weights={k:v.clone() for k,v in net.state_dict().items()}
    torch.manual_seed(55);expected=net.eval_actor(obs)[0];rng=torch.get_rng_state()
    probe=AttentionProbe(net);probe.enabled=True
    torch.manual_seed(55);actual=net.eval_actor(obs)[0]
    assert torch.equal(actual,expected) and torch.equal(torch.get_rng_state(),rng)
    assert len(probe.maps)==8
    for key,p in probe.maps.items():
        assert p.shape==(2,16,16) if key.startswith('gta') else p.shape==(2,4,4)
        assert torch.isfinite(p).all() and (p>=0).all()
        torch.testing.assert_close(p.sum(-1),torch.ones_like(p.sum(-1)))
    for k,v in net.state_dict().items():assert torch.equal(v,weights[k])
    probe.close()


def test_restore_both_query_and_key_order():
    p=torch.arange(32).reshape(2,4,4)
    order=torch.tensor([2,0,3,1])
    assert torch.equal(canonical_attention(p[:,order][:,:,order],order),p)


def test_reconstructed_human_attention_matches_pytorch_weights():
    torch.manual_seed(7)
    module=torch.nn.MultiheadAttention(64,2,dropout=0,batch_first=True).eval()
    q=torch.randn(3,4,64);bias=torch.randn(6,4,4)
    bias[:,0,2]=float('-inf')
    _,reference=module(q,q,q,attn_mask=bias,need_weights=True,average_attn_weights=False)
    actual=human_attention(module,(q,q,q),{'attn_mask':bias})
    torch.testing.assert_close(actual,reference[0],atol=1e-7,rtol=1e-6)


def test_scene_capture_is_canonical_despite_shuffle():
    net=build('stage2_before').eval();obs=observations();probe=AttentionProbe(net);probe.enabled=True
    encoder=net.actor_encoder
    encoder.relation_graph_spec['shuffle_token_order']=False
    encoder(obs);reference={k:v.clone() for k,v in probe.maps.items()}
    order=torch.randperm(16)
    encoder(obs,token_order=order)
    for key,p in reference.items():torch.testing.assert_close(probe.maps[key],p,atol=2e-6,rtol=1e-5)
    probe.close()
