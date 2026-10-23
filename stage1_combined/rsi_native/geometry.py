"""Sample native humanoid collision shapes against actual object triangles."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d
import trimesh


def rotate(q,v):
    q=np.asarray(q);v=np.asarray(v)
    t=2*np.cross(q[...,:3],v)
    return v+q[...,3:4]*t+np.cross(q[...,:3],t)


def collision_samples(xml_path,names,spacing=.015):
    xml_path=Path(xml_path);xml=ET.parse(xml_path).getroot()
    meshes={e.get('name'):xml_path.parent/e.get('file') for e in xml.findall('asset/mesh')}
    ids=[];points=[];radii=[]
    for body in xml.findall('.//worldbody//body'):
        for geom in body.findall('geom'):
            spec=geom.attrib;offset=np.fromstring(spec.get('pos','0 0 0'),sep=' ');radius=0.
            if spec.get('type')=='mesh':
                mesh=trimesh.load(meshes[spec['mesh']],force='mesh',process=True)
                pts=np.concatenate([mesh.vertices,mesh.triangles_center])+offset
            elif 'fromto' in spec:
                v=np.fromstring(spec['fromto'],sep=' ');count=int(np.ceil(np.linalg.norm(v[3:]-v[:3])/spacing))+1
                pts=np.linspace(v[:3],v[3:],count);radius=float(spec['size'].split()[0])
            else:
                pts=offset[None];radius=float(spec['size'].split()[0])
            ids.extend([names.index(body.get('name'))]*len(pts));points.extend(pts);radii.extend([radius]*len(pts))
    return np.asarray(ids),np.asarray(points,dtype=np.float32),np.asarray(radii,dtype=np.float32)


class NativeGeometry:
    def __init__(self,folder,shapes):
        folder=Path(folder);xml=ET.parse(folder/'asset.urdf')
        parts=[]
        for collision in xml.findall('.//collision'):
            mesh=collision.find('geometry/mesh');assert mesh is not None
            part=trimesh.load(folder/mesh.get('filename'),force='mesh',process=True)
            if mesh.get('scale'):part.apply_scale(np.fromstring(mesh.get('scale'),sep=' '))
            origin=collision.find('origin')
            if origin is not None:
                import scipy.spatial.transform
                matrix=np.eye(4);matrix[:3,:3]=scipy.spatial.transform.Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix();matrix[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ');part.apply_transform(matrix)
            parts.append(part)
        mesh=trimesh.util.concatenate(parts);self.watertight=bool(mesh.is_watertight)
        vertices=o3d.core.Tensor(np.asarray(mesh.vertices,dtype=np.float32));triangles=o3d.core.Tensor(np.asarray(mesh.faces,dtype=np.uint32))
        self.scene=o3d.t.geometry.RaycastingScene(nthreads=4);self.scene.add_triangles(vertices,triangles);self.shapes=shapes

    def check(self,body,body_rot,object_z):
        ids,points,radii=self.shapes
        world=body[:,ids]+rotate(body_rot[:,ids],points[None])
        local=world.copy();local[...,2]-=object_z
        query=o3d.core.Tensor(local.reshape(-1,3).astype(np.float32))
        distance=self.scene.compute_distance(query).numpy().reshape(local.shape[:2])
        inside=self.scene.compute_occupancy(query,nsamples=3).numpy().reshape(local.shape[:2])>.5
        clearance=np.where(inside,-distance,distance)-radii[None]
        return clearance.min(-1), (world[...,2]-radii[None]).min(-1)
