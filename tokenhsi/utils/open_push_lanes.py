"""Owner-to-physical-lane mapping; objects remain in their original lanes."""
import torch


def sample_lanes(count, device, randomized=False, mode='random'):
    if type(randomized) is not bool:
        raise ValueError('openPushRandomizeLanes must be boolean')
    if mode not in ('random', 'fixed', 'swapped'):
        raise ValueError('OPEN_PUSH_LANES must be random, fixed or swapped')
    lanes = torch.arange(2, device=device).expand(count, 2).clone()
    if mode == 'swapped':
        return lanes.flip(-1)
    if randomized and mode == 'random':
        swap = torch.rand(count, device=device) < .5
        lanes[swap] = lanes[swap].flip(-1)
    return lanes


def owner_values(values, lanes):
    """Gather physical lane tensors into canonical agent/owner order."""
    index = lanes.reshape(*lanes.shape, *([1] * (values.ndim - 2)))
    return values.gather(1, index.expand(-1, -1, *values.shape[2:]))


def logical_order(lanes):
    return torch.cat((lanes, lanes + 2), -1)
