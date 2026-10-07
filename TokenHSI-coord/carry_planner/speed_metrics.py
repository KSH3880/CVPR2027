"""Executed agent-step speed moments; population variance, units m/s."""
import math

NAMES = ("requested", "command", "actual")


class SpeedMoments:
    def __init__(self, rows, device):
        import torch
        self.data = torch.zeros(rows, 3, 3, dtype=torch.float64, device=device)

    def update(self, requested, command, actual):
        import torch
        values = torch.stack((requested, command, actual), dim=-1).double()
        self.data[..., 0] += 1
        self.data[..., 1] += values
        self.data[..., 2] += values.square()

    def reset(self, rows):
        self.data[rows] = 0

    def record(self, row):
        return dict(zip(NAMES, self.data[row].tolist()))


def summarize_speed(records):
    result = {}
    for name in NAMES:
        n, total, squares = (sum(r[name][i] for r in records) for i in range(3))
        mean = total / max(n, 1)
        variance = max(0., squares / max(n, 1) - mean * mean)
        result[name] = dict(agent_steps=int(n), mean_mps=mean,
                            variance_mps2=variance, std_mps=math.sqrt(variance))
    return result
