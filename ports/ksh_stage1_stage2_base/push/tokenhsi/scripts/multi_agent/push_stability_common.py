"""Geometry/metrics shared by the rectangular, equal-mass PUSH probes."""
import itertools
import math

SIDES = [1.2, 1.35, 1.5]
MASSES = [10., 15., 20., 25., 30.]
HEIGHT_FRACTIONS = [.30, .35, .40, .45, .50]
SHAPES = list(itertools.product(SIDES, repeat=3))


def tilt_degrees(q):
    import torch
    q = q/q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    up_x = 2*(x*z+w*y); up_y = 2*(y*z-w*x)
    up_z = 1-2*(x*x+y*y)
    return torch.atan2(torch.sqrt(up_x.square()+up_y.square()), up_z)*180/math.pi


def static_envelope(size, mass, fraction, mu=.6):
    x, y, z = size; height = fraction*z
    sliding = mu*mass*9.81
    tipping = mass*9.81*(y/2)/height
    return dict(sliding_reference_n=sliding,tipping_reference_n=tipping,
                tipping_over_sliding=tipping/sliding,
                assumptions='Horizontal quasistatic force at rear-face centerline, central COM, Coulomb coefficient0.6. Actual simulator contact may differ; no humanoid or impact guarantee.')
