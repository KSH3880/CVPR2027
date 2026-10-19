"""TeamHOI dual AMP mechanics, adapted to native TokenHSI129-D windows.

Reference: sail-sg/TeamHOI@6fdc9885f8a9c82854be99adca6f2570e67ac63a.
MIT license/copyright Sea AI Lab2026: third_party/teamhoi/LICENSE.
"""
import torch
import torch.nn.functional as F
from utils.lift_spec import mask_lift_amp, inverse_rotate


def mask_history(obs):
    if obs.shape[-1]%129:raise ValueError('Expected native129-D history')
    return mask_lift_amp(obs.reshape(*obs.shape[:-1],-1,129)).reshape_as(obs)


def blend_weight(distance_history,threshold=1.,sharpness=15.):
    return torch.sigmoid((threshold-distance_history)*sharpness).mean(-1,keepdim=True)


def outside_history(distance_history,threshold=1.):
    return distance_history.amin(-1)>threshold


def style_reward(logits,scale):
    # Algebraically equivalent to TeamHOI -log(max(1-sigmoid(logit),1e-4)),
    # retaining its saturation while avoiding sigmoid subtraction cancellation.
    return F.softplus(logits).clamp_max(9.210340371976184)*scale


def blend_rewards(full,masked,distance_history,threshold=1.,sharpness=15.):
    weight=blend_weight(distance_history,threshold,sharpness)
    return full*(1-weight)+masked*weight


def rim_distance(roots,box,size,samples=64):
    """Root XY -> nearest uniformly sampled lower box rim, as TeamHOI."""
    width,depth=size[...,0],size[...,1]
    perimeter=2*(width+depth)
    t=torch.arange(samples,device=box.device,dtype=box.dtype)/samples
    s=perimeter[...,None]*t
    w=width[...,None];d=depth[...,None]
    x=torch.where(s<w,-w/2+s,torch.where(s<w+d,w/2,
        torch.where(s<2*w+d,w/2-(s-w-d),-w/2)))
    y=torch.where(s<w,-d/2,torch.where(s<w+d,-d/2+s-w,
        torch.where(s<2*w+d,d/2,d/2-(s-2*w-d))))
    z=-size[...,2,None].expand_as(x)/2
    local=torch.stack((x,y,z),-1)
    # inverse_rotate(conjugate(q)) is forward rotation.
    q=box[...,3:7].clone();q[...,:3]*=-1
    world=inverse_rotate(q[...,None,:].expand(*local.shape[:-1],4),local)+box[...,None,:3]
    return (world[...,:2]-roots[...,None,:2]).norm(dim=-1).amin(-1)


def discriminator_loss(agent_logits,demo_logits,demo_obs,network,masked,config):
    neg=F.binary_cross_entropy_with_logits(agent_logits,torch.zeros_like(agent_logits))
    pos=F.binary_cross_entropy_with_logits(demo_logits,torch.ones_like(demo_logits))
    loss=.5*(neg+pos)
    get_logit=network.get_disc_logit_weights_masked if masked else network.get_disc_logit_weights
    get_weights=network.get_disc_weights_masked if masked else network.get_disc_weights
    logit_reg=get_logit().square().sum()
    grad=torch.autograd.grad(demo_logits,demo_obs,grad_outputs=torch.ones_like(demo_logits),
                            create_graph=True,retain_graph=True,only_inputs=True)[0]
    grad_penalty=grad.square().sum(-1).mean()
    loss=loss+config['disc_logit_reg']*logit_reg+config['disc_grad_penalty']*grad_penalty
    if config['disc_weight_decay']:
        loss=loss+config['disc_weight_decay']*torch.cat(get_weights()).square().sum()
    return dict(disc_loss=loss,disc_grad_penalty=grad_penalty.detach(),disc_logit_loss=logit_reg.detach(),
        disc_agent_acc=(agent_logits<0).float().mean().detach(),
        disc_demo_acc=(demo_logits>0).float().mean().detach(),
        disc_agent_logit=agent_logits.detach(),disc_demo_logit=demo_logits.detach())


def check_dual_checkpoint(weights,expected):
    if weights.get('dual_amp_metadata')!=expected or 'amp_masked_input_mean_std' not in weights:
        raise ValueError('Dual AMP checkpoint/normalizer contract mismatch; use the dedicated config')
