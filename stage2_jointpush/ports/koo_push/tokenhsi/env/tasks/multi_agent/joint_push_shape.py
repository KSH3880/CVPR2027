"""Finite per-environment cuboid shape grid, fixed throughout each environment lifetime."""
import itertools
import math
import torch


def shape_grid(spec):
    if spec['mode']!='env_fixed_grid' or spec['mass_mode']!='random_mass_grid':
        raise ValueError('Only environment-fixed shape/mass grids are supported')
    axes=[]
    for lo,hi,step in zip(spec['size_min'],spec['size_max'],spec['step']):
        if not all(math.isfinite(v) for v in (lo,hi,step)) or not 0<lo<=hi or step<=0:
            raise ValueError('Invalid shape bounds')
        count=round((hi-lo)/step)
        if abs(lo+count*step-hi)>1e-6:raise ValueError('Shape range must be divisible by step')
        axes.append([round(lo+i*step,6) for i in range(count+1)])
    if len(axes)!=3 or spec['size_min'][0]<2 or spec['size_min'][1]<1 or spec['size_min'][2]<.8:
        raise ValueError('Require width>=2m, depth>=1m, height>=hand target')
    return list(itertools.product(*axes))


def shape_mass_grid(spec):
    shapes=shape_grid(spec)
    lo,hi,step=spec['mass_min'],spec['mass_max'],spec['mass_step']
    if not all(math.isfinite(v) for v in (lo,hi,step)) or not 0<lo<=hi<=60 or step<=0:
        raise ValueError('Require finite random mass bounds with upper bound <=60kg')
    count=round((hi-lo)/step)
    if abs(lo+count*step-hi)>1e-6:raise ValueError('Mass interval must be divisible by step')
    return [(size,round(lo+i*step,6)) for size in shapes for i in range(count+1)]
