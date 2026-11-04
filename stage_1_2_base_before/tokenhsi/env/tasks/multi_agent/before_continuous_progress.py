"""First-opening, position-driven progress; diagnostic reads never commit anchors."""
import torch


def validate_continuous_progress(option):
    if option != {'mode': 'first_open_distance_v1', 'wait_fraction': 0.5}:
        raise ValueError('continuous_progress requires first_open_distance_v1 and wait_fraction=0.5')


class ContinuousProgress:
    def __init__(self, shape, device):
        self.seen = torch.zeros(shape, dtype=torch.bool, device=device)
        self.q0 = torch.zeros(shape, device=device)
        self.r0 = torch.zeros(shape, device=device)

    def reset(self, ids):
        self.seen[ids] = False
        self.q0[ids] = 0
        self.r0[ids] = 0

    def step(self, approach, waiting, gate, dependent):
        # Curves are evaluated at the SAME geometry, so gate alone cannot jump.
        waiting = 0.5 * waiting
        opening = gate & dependent & ~self.seen
        self.q0.copy_(torch.where(opening, approach, self.q0))
        self.r0.copy_(torch.where(opening, waiting, self.r0))
        self.seen |= opening
        span = 1. - self.q0
        safe_span = torch.where(span > 0, span, torch.ones_like(span))
        value = (self.r0 + (1-self.r0)*(approach-self.q0)/safe_span).clamp(0., 1.)
        # Already at exact goal: no remaining distance to allocate. Preserve continuity.
        value = torch.where(span > 0, value, self.r0 * approach)
        return torch.where(self.seen & dependent, value, waiting)
