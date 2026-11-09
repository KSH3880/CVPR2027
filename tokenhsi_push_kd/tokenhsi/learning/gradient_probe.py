"""Opt-in diagnostic: read PPO/KD gradients without accumulating .grad."""
import json
import os
from pathlib import Path
import torch


def gradient_stats(ppo_loss, kd_loss, parameters, coefficient):
    ppo = torch.autograd.grad(ppo_loss, parameters, retain_graph=True, allow_unused=True)
    kd = torch.autograd.grad(kd_loss, parameters, retain_graph=True, allow_unused=True)
    device = ppo_loss.device
    p2 = torch.zeros((), device=device)
    k2 = torch.zeros_like(p2)
    dot = torch.zeros_like(p2)
    for p, k in zip(ppo, kd):
        if p is not None: p2 += p.detach().float().square().sum()
        if k is not None: k2 += k.detach().float().square().sum()
        if p is not None and k is not None: dot += (p.detach().float()*k.detach().float()).sum()
    pn, kn = p2.sqrt().item(), k2.sqrt().item()
    return dict(ppo_grad_norm=pn, kd_grad_norm=kn,
                weighted_kd_grad_norm=coefficient*kn,
                weighted_kd_over_ppo=coefficient*kn/pn if pn > 1e-12 else None,
                cosine=dot.item()/(pn*kn) if pn*kn > 1e-12 else None,
                coefficient_for_ratio_0p3=.3*pn/kn if kn > 1e-12 else None)


def record_probe(agent, ppo_loss, kd_loss, push_mask, kd_mask):
    output = os.environ.get('PUSH_KD_GRAD_PROBE')
    if not output:
        return
    epoch = agent.epoch_num
    if getattr(agent, '_probe_epoch', None) != epoch:
        agent._probe_epoch, agent._probe_update = epoch, 0
    update = agent._probe_update
    agent._probe_update += 1
    updates_per_epoch = len(agent.dataset)*agent.mini_epochs_num
    if update not in (0, updates_per_epoch//2, updates_per_epoch-1):
        return
    parameters = [p for name,p in agent.model.named_parameters()
                  if 'a2c_network.transformer_encoder.' in name and p.requires_grad]
    if not parameters:
        raise RuntimeError('No shared transformer parameters for gradient probe')
    stats = gradient_stats(ppo_loss, kd_loss, parameters, agent._kd_coef)
    row = dict(epoch=epoch, update=update+1, updates_per_epoch=updates_per_epoch,
               scope='shared_transformer_encoder', coefficient=agent._kd_coef,
               ppo_loss=ppo_loss.detach().item(), kd_kl=kd_loss.detach().item(),
               push_samples=int(push_mask.sum()), kd_samples=int(kd_mask.sum()), **stats)
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream: stream.write(json.dumps(row)+'\n')
    print('GRAD_PROBE '+json.dumps(row), flush=True)
