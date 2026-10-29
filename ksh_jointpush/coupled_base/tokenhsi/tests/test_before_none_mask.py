from copy import deepcopy

import pytest
import torch

from before_test_utils import config, build, observations, BASE
from learning.multi_agent.human_coordination import CoordinationBlock, HumanCoordination
from learning.multi_agent.before_network import freeze_actor, transfer_stage1
from utils.before_spec import validate_env, checkpoint_contract
from utils.before_spec import check_checkpoint
from utils.relation_task_spec import checkpoint_metadata

NAME = 'stage2_before_nonemask_rsi433_near50'
MODE = 'before_only_mask_none'




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




