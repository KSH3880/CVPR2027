import torch


LOCO_PLATFORM = dict(probability=.5, heightSampleMax=1., maxTopSurfaceHeight=1.2,
                     size=[.4, .4, .02], initialGap=.05)


def platform_heights(box_height, draws, spec):
    minimum = spec['size'][2]
    maximum = spec['maxTopSurfaceHeight'] - box_height
    if (maximum < minimum).any():
        raise ValueError('Payload is too tall for the loco platform height limits')
    return torch.minimum((draws * spec['heightSampleMax']).clamp_min(minimum), maximum)
