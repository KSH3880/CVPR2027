"""Pure PyTorch losses; no simulator dependency."""
import torch


def gaussian_forward_kl(mu, sigma, teacher_mu, teacher_sigma):
    """KL(teacher || student), sum over actions, mean over samples."""
    if mu.shape[0] == 0:
        return mu.sum() * 0
    tmu, ts = teacher_mu.detach().float(), teacher_sigma.detach().float()
    mu, sigma = mu.float(), sigma.float()
    return (torch.log(sigma / ts) + (ts.square() + (tmu-mu).square()) /
            (2*sigma.square()) - .5).sum(-1).mean()


def masked_mean(value, mask):
    value = value.reshape(len(mask), -1).mean(-1)
    return value[mask].mean() if mask.any() else value.sum()*0


def push_advantages(returns, values, mask, normalize=True):
    advantage = (returns-values).sum(-1)
    result = torch.zeros_like(advantage)
    selected = advantage[mask]
    if normalize and selected.numel():
        selected = (selected-selected.mean())/(selected.std(unbiased=False)+1e-8)
    result[mask] = selected
    return result


def teacher_observation(obs):
    """397 student -> 354 original: H223 + old task127 + old onehot4."""
    if obs.shape[-1] != 397:
        raise ValueError('Expected TokenHSI PUSH 397D observation')
    return torch.cat((obs[..., :350], obs[..., 392:396]), -1)


def normalize_policy(obs, rms, tasks):
    """Task labels bypass RMS exactly; do not recover labels after clipping."""
    labels = obs[..., -tasks:].clone()
    result = rms(obs)
    result[..., -tasks:] = labels
    return result
