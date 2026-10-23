"""Immutable creation-time cube sizes with constant density."""
import math
import torch


def size_values(env):
    spec=env.get('pushSizeRandomization')
    if spec is None:return [env['pushBox']['size'][0]]
    presets = [([1.1,1.4],31), ([1.2,1.4],21)]
    if not any(spec == {'side_range':bounds, 'num_sizes':count, 'sampling':'uniform_creation'}
               for bounds,count in presets):
        raise ValueError('PUSH sizes require 31 cubes in 1.1..1.4m or 21 cubes in 1.2..1.4m')
    if env['pushBox']['size'] != [1.3]*3 or not env.get('pushGoalContact'):
        raise ValueError('Random sizes require base cube and goal-face contact')
    return [round(spec['side_range'][0]+i*.01,2) for i in range(spec['num_sizes'])]


def size_tensor(boxes,size):
    return torch.as_tensor(size,device=boxes.device,dtype=boxes.dtype).expand(*boxes.shape[:2],3)


def density(env):
    return env['pushBox']['mass']/math.prod(env['pushBox']['size'])
