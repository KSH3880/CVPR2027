"""Conservative Box RSI support check using native foot vertices/pelvis sphere."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from isaacgym.torch_utils import quat_rotate


def load_support_points(xml_path,device):
    import trimesh
    xml_path=Path(xml_path);xml=ET.parse(xml_path).getroot();meshes={m.get('name'):xml_path.parent/m.get('file') for m in xml.findall('asset/mesh')};bodies=list(xml.findall('.//worldbody//body'));ids=[];points=[];radii=[]
    for index in [0,11,14]:
        geom=bodies[index].find('geom');pos=np.fromstring(geom.get('pos','0 0 0'),sep=' ')
        if geom.get('type')=='mesh':vertices=trimesh.load(meshes[geom.get('mesh')],force='mesh').vertices+pos;radius=0.
        else:vertices=pos[None];radius=float(geom.get('size').split()[0])
        ids.extend([index]*len(vertices));points.extend(vertices);radii.extend([radius]*len(vertices))
    return torch.tensor(ids,device=device),torch.tensor(np.asarray(points),device=device,dtype=torch.float32),torch.tensor(radii,device=device,dtype=torch.float32)


def box_supported(body,boxes,sizes,shapes,tolerance=.03):
    ids,points,radii=shapes;n=len(body)
    if not n:return torch.zeros(0,device=body.device,dtype=torch.bool)
    q=body[:,ids,3:7];world=body[:,ids,:3]+quat_rotate(q.reshape(-1,4),points[None].expand(n,-1,-1).reshape(-1,3)).reshape(n,-1,3)
    bottom=world[...,2]-radii[None];support=((bottom>=-.05)&(bottom<=tolerance)).any(-1)
    inv=boxes[:,3:7].clone();inv[:,:3]*=-1
    local=quat_rotate(inv[:,None].expand(-1,len(ids),-1).reshape(-1,4),(world-boxes[:,None,:3]).reshape(-1,3)).reshape(n,-1,3)
    for axis in range(3):
        basis=torch.zeros(n,3,device=body.device);basis[:,axis]=1
        nz=quat_rotate(boxes[:,3:7],basis)[:,2];sign=torch.where(nz>=0,1.,-1.)
        gap=local[...,axis]*sign[:,None]-sizes[:,axis,None]/2-radii[None]
        inside=torch.ones_like(gap,dtype=torch.bool)
        for other in range(3):
            if other!=axis:inside&=local[...,other].abs()<=sizes[:,other,None]/2+radii[None]+.005
        support|=((gap>=-.015)&(gap<=tolerance)&inside&(nz.abs()[:,None]>=.5)).any(-1)
    return support
