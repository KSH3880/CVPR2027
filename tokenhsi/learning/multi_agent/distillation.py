import hashlib

import torch


def gaussian_forward_kl(student_mu, student_sigma, teacher_mu, teacher_sigma):
    teacher_mu = teacher_mu.detach().float()
    teacher_sigma = teacher_sigma.detach().float()
    student_mu = student_mu.float()
    student_sigma = student_sigma.float()
    kl = (torch.log(student_sigma / teacher_sigma)
          + (teacher_sigma.square() + (teacher_mu - student_mu).square())
          / (2 * student_sigma.square()) - 0.5)
    return kl.sum(dim=-1).mean()


def teacher_digest(teacher):
    digest = hashlib.sha256()
    for module in (teacher.model, teacher.rms):
        for value in module.state_dict().values():
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def gradient_report(model, kl_loss, actor_loss):
    network = model.a2c_network
    groups = {
        'human_tokenizer': network.actor_encoder.tokenizers[0],
        'object_tokenizer': network.actor_encoder.tokenizers[1],
        'goal_tokenizer': network.actor_encoder.tokenizers[2],
        'transformer': network.actor_encoder.layers,
        'action_head': network.action_head,
    }
    if hasattr(network.actor_encoder, 'edge_encoder'):
        groups['edge_encoder'] = network.actor_encoder.edge_encoder
    actor_params = list(network.actor_encoder.parameters()) + list(network.action_head.parameters())
    actor_params = [p for p in actor_params if p.requires_grad]
    kl_grads = torch.autograd.grad(kl_loss, actor_params, retain_graph=True, allow_unused=True)
    ppo_grads = torch.autograd.grad(actor_loss, actor_params, retain_graph=True, allow_unused=True)

    def norm(grads):
        return torch.stack([g.detach().float().square().sum() for g in grads if g is not None]).sum().sqrt().item()

    if not all(torch.isfinite(g).all() for g in kl_grads + ppo_grads if g is not None):
        raise RuntimeError('Non-finite actor gradient')
    by_id = dict(zip(map(id, actor_params), kl_grads))
    report = {name: norm([by_id[id(p)] for p in module.parameters() if id(p) in by_id])
              for name, module in groups.items()}
    report['type_embedding'] = norm([by_id[id(network.actor_encoder.type_embed)]])
    if hasattr(network.actor_encoder, 'edge_encoder'):
        edge = network.actor_encoder.edge_encoder
        report['edge_internal'] = norm([by_id[id(p)] for name, p in edge.named_parameters()
                                        if name != 'bias_projection'])
    report['weighted_kl_actor_grad'] = norm(kl_grads)
    report['ppo_actor_grad'] = norm(ppo_grads)
    report['kl_to_ppo_grad_ratio'] = report['weighted_kl_actor_grad'] / max(report['ppo_actor_grad'], 1e-12)
    critic_params = list(network.critic_encoder.parameters()) + list(network.value_head.parameters())
    critic_grads = torch.autograd.grad(kl_loss, critic_params, retain_graph=True, allow_unused=True)
    report['kl_critic_grad_absent'] = all(g is None for g in critic_grads)
    return report
