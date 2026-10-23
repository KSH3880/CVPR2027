"""Conservative collider bounds for PUSH scene initialization only."""
import hashlib
import itertools
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
from utils.door_reference import rotate


def asset_geometry(env):
    path = Path(env['asset']['assetRoot'])/env['asset']['assetFileName']
    data = path.read_bytes(); tree = ET.fromstring(data)
    meshes = {m.get('name'): (path.parent/m.get('file'), m) for m in tree.findall('./asset/mesh')}
    digest = hashlib.sha256(data)
    for name, (source, _) in sorted(meshes.items()):
        digest.update(name.encode()); digest.update(source.read_bytes())
    return tree, meshes, digest.hexdigest()


def reset_geometry_contract(env):
    spec = env['pushTask']
    if 'reset_geometry_filter' not in spec:
        return None
    return {key: spec[key] for key in ('reset_geometry_filter', 'reset_geometry_margin_m', 'reset_floor_clearance_m', 'reset_geometry_sha256')}


class PushResetGeometry:
    """Enclose spheres/capsules and complete mesh bounds in box-local AABBs.

    AABB rejection is conservative: it can reject a safe pose, but covers the
    actual foot mesh extent rather than testing only its ankle joint center.
    """
    def __init__(self, env, body_names, device):
        tree, meshes, digest = asset_geometry(env)
        spec = env['pushTask']
        if digest != spec['reset_geometry_sha256']:
            raise ValueError('PUSH reset collider asset/hash differs')
        self.margin = spec['reset_geometry_margin_m']
        self.floor_clearance = spec['reset_floor_clearance_m']
        points=[]; radii=[]; indices=[]; self.names=[]
        for body in tree.findall('.//worldbody//body'):
            if body.get('name') not in body_names:
                continue
            for geom in body.findall('geom'):
                kind = geom.get('type', 'capsule')
                position = [float(x) for x in geom.get('pos','0 0 0').split()]
                radius = 0.
                if geom.get('quat') not in (None, '1 0 0 0') or geom.get('euler') is not None:
                    raise ValueError('Unsupported rotated local PUSH collider')
                if kind == 'sphere':
                    vertices = [position]; radius = float(geom.get('size').split()[0])
                elif kind == 'capsule':
                    ends = [float(x) for x in geom.get('fromto').split()]
                    vertices = [ends[:3],ends[3:]];radius = float(geom.get('size').split()[0])
                elif kind == 'mesh':
                    source,mesh = meshes[geom.get('mesh')]
                    values = torch.tensor([[float(x) for x in line.split()[1:4]] for line in source.read_text().splitlines() if line.startswith('v ')])
                    values *= torch.tensor([float(x) for x in mesh.get('scale','1 1 1').split()])
                    lo,hi = values.amin(0),values.amax(0)
                    vertices = [[float((hi if sign else lo)[axis])+position[axis] for axis,sign in enumerate(corner)] for corner in itertools.product((0,1),repeat=3)]
                else:
                    raise ValueError('Unsupported PUSH collider type: '+kind)
                points.append(vertices+[vertices[-1]]*(8-len(vertices)))
                radii.append(radius);indices.append(body_names.index(body.get('name')));self.names.append(geom.get('name'))
        self.points = torch.tensor(points,device=device)
        self.radii = torch.tensor(radii,device=device)
        self.indices = torch.tensor(indices,device=device,dtype=torch.long)

    def bounds(self, body, rotations, boxes):
        q = rotations[:,self.indices,None,:].expand(-1,-1,8,-1)
        world = rotate(q,self.points[None].expand(len(body),-1,-1,-1))+body[:,self.indices,None,:]
        inv = boxes[:,3:7].clone();inv[:,:3]*=-1
        local = rotate(inv[:,None,None].expand_as(q),world-boxes[:,None,None,:3])
        return local.amin(-2)-self.radii[None,:,None],local.amax(-2)+self.radii[None,:,None]

    def invalid(self, body, rotations, boxes, sizes):
        lo,hi = self.bounds(body,rotations,boxes)
        overlap = (hi+self.margin > -sizes[:,None]/2)&(lo-self.margin < sizes[:,None]/2)
        return overlap.all(-1).any(-1)

    def rear_shift(self, body, rotations, boxes, sizes):
        lo,hi = self.bounds(body,rotations,boxes)
        overlap = (hi+self.margin > -sizes[:,None]/2)&(lo-self.margin < sizes[:,None]/2)
        near_xz = overlap[...,0]&overlap[...,2]
        # Keep a millimeter beyond the boundary to absorb float32 world-offset subtraction.
        return torch.where(near_xz,hi[...,1]+self.margin+sizes[:,None,1]/2+.001,0.).amax(-1).clamp_min(0)

    def floor_shift(self, body, rotations):
        identity=body.new_zeros(len(body),13);identity[:,6]=1
        lo,_=self.bounds(body,rotations,identity)
        return (self.floor_clearance-lo[...,2].amin(-1)).clamp_min(0)
