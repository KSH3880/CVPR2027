"""CPU-testable selection contract for the opt-in BEFORE carry initialization."""
import torch


def select_carry(predecessor, templates, probability, available):
    if not 0. <= probability <= 1.:
        raise ValueError('carry RSI probability must be in [0, 1]')
    eligible = (predecessor >= 0) & (templates == 4)
    requested = torch.zeros_like(eligible)
    if probability == 1.:
        requested = eligible.clone()
    elif probability > 0.:
        requested[eligible] = torch.rand(int(eligible.sum()), device=templates.device) < probability
    return eligible, requested, requested & available
