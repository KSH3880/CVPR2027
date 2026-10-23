"""Foot-edge/object-triangle crossings, independent of mesh parity closure."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import open3d as o3d
from rsi_native.geometry import rotate


class FootSurface:
    def __init__(self,xml_path,names):
        p=Path(xml_path);xml=ET.parse(p).getroot();meshes={m.get('name'):p.parent/m.get('file') for m in xml.findall('asset/mesh')};ids=[];edges=[]
        for body in xml.findall('.//worldbody//body'):
            if body.get('name') not in ['right_foot','left_foot']:continue
            g=body.find('geom');mesh=trimesh.load(meshes[g.get('mesh')],force='mesh',process=True);v=mesh.vertices+np.fromstring(g.get('pos','0 0 0'),sep=' ');e=v[mesh.edges_unique]
            ids.extend([names.index(body.get('name'))]*len(e));edges.extend(e)
        self.ids=np.asarray(ids);self.edges=np.asarray(edges,dtype=np.float32)

    def crossing(self,body,geometry,object_z,tolerance=.002):
        if not len(body):return np.zeros(0,dtype=bool)
        world=body[:,self.ids,None,:3]+rotate(body[:,self.ids,None,3:7],self.edges[None]);world[...,2]-=object_z
        start=world[:,:,0];direction=world[:,:,1]-start
        rays=np.concatenate([start,direction],-1).astype(np.float32)
        hit=geometry.scene.cast_rays(o3d.core.Tensor(rays.reshape(-1,6)))
        t=hit['t_hit'].numpy().reshape(start.shape[:2]);normals=hit['primitive_normals'].numpy().reshape(start.shape)
        projected=np.abs((direction*normals).sum(-1));safe_t=np.where(np.isfinite(t),t,-1.);depth=np.minimum(safe_t,1-safe_t)*projected
        return ((t>0)&(t<1)&(depth>tolerance)).any(-1)
