"""Foot-ground projection and vertical surface support checks for RSI."""
import numpy as np
import open3d as o3d
from rsi_native.geometry import rotate


def foot_bottom(body, shapes):
    ids,points,radii=shapes;mask=np.isin(ids,[11,14]);ids=ids[mask];points=points[mask];radii=radii[mask]
    offset=rotate(body[:,ids,3:7],points[None]);world=body[:,ids,:3]+offset
    bottom=world[...,2]-radii;chosen=bottom.argmin(-1);rows=np.arange(len(body))
    vel=body[:,ids,7:10]+np.cross(body[:,ids,10:13],offset)
    return bottom[rows,chosen],vel[rows,chosen,2]


def project_ground(root, body, shapes, target=.005):
    bottom,speed=foot_bottom(body,shapes);delta=target-bottom
    root=root.copy();body=body.copy();root[:,2]+=delta;body[:,:,2]+=delta[:,None]
    root[:,9]-=speed;body[:,:,9]-=speed[:,None]
    return root,body,delta


def supported(body, geometry, object_z, tolerance=.03):
    ids,points,radii=geometry.shapes;mask=np.isin(ids,[0,11,14]);ids=ids[mask];points=points[mask];radii=radii[mask]
    world=body[:,ids,:3]+rotate(body[:,ids,3:7],points[None]);bottom=world[...,2]-radii[None]
    ground=(bottom>=-.02)&(bottom<=tolerance)
    origins=world.copy();origins[...,2]=bottom+.02-object_z
    rays=np.concatenate([origins,np.broadcast_to([0,0,-1],origins.shape)],-1).astype(np.float32)
    hit=geometry.scene.cast_rays(o3d.core.Tensor(rays.reshape(-1,6)))
    distance=hit['t_hit'].numpy().reshape(bottom.shape);normal=hit['primitive_normals'].numpy().reshape(*bottom.shape,3)
    gap=distance-.02
    surface=np.isfinite(distance)&(gap>=-.015)&(gap<=tolerance)&(np.abs(normal[...,2])>=.5)
    return (ground|surface).any(-1),np.minimum(np.where(ground,bottom,np.inf),np.where(surface,np.abs(gap),np.inf)).min(-1)
