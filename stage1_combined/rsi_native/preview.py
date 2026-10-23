"""Schematic native-mesh/RSI collision-sample preview, not a rendered screenshot."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trimesh
from rsi_native.geometry import NativeGeometry,collision_samples,rotate
from rsi_native.repair import NativeFK


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.audit/'manifest.json').read_text());entries=manifest['assets']
    fk=NativeFK('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml');shapes=collision_samples('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml',fk.names)
    fig=plt.figure(figsize=(18,11),constrained_layout=True)
    for panel,(category,skill,status) in enumerate(( (c,s,ok) for c in ['Chair','Table'] for s in ['sit','climb'] for ok in [True,False]),1):
        selected=None
        for e in entries:
            if e['category']!=category:continue
            d=np.load(a.audit/'audited_bank'/e['id']/(skill+'.npz'))
            mask=d['usable'] if status else d['phase']&~d['usable']
            ids=np.flatnonzero(mask)
            if len(ids):selected=e,d,int(ids[len(ids)//2]);break
        ax=fig.add_subplot(2,4,panel,projection='3d');ax.set_title(f'{category} {skill} / '+('passed' if status else 'rejected'))
        if selected is None:ax.text2D(.1,.5,'No candidate',transform=ax.transAxes);continue
        e,d,k=selected;folder=Path(e['path'])
        import xml.etree.ElementTree as ET
        from scipy.spatial.transform import Rotation
        for collision in ET.parse(folder/'asset.urdf').findall('.//collision'):
            mesh=collision.find('geometry/mesh');part=trimesh.load(folder/mesh.get('filename'),force='mesh')
            if mesh.get('scale'):part.apply_scale(np.fromstring(mesh.get('scale'),sep=' '))
            origin=collision.find('origin')
            if origin is not None:
                part.vertices=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).apply(part.vertices)+np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            v=part.vertices.copy();v[:,2]+=e['bbox'][2]/2
            faces=part.faces[::max(1,len(part.faces)//2000)]
            ax.plot_trisurf(v[:,0],v[:,1],v[:,2],triangles=faces,color='gray',alpha=.2,linewidth=0)
        body=d['body_state'][k];ids,points,radii=shapes;world=body[ids,:3]+rotate(body[ids,3:7],points)
        ax.scatter(*world[::3].T,s=2,c='green' if status else 'red')
        for i,parent in enumerate(fk.parents):
            if parent>=0:ax.plot(*body[[parent,i],:3].T,color='navy',linewidth=2)
        ax.set_xlim(-.8,.8);ax.set_ylim(-.8,.8);ax.set_zlim(0,2.2);ax.set_box_aspect((1.6,1.6,2.2));ax.view_init(18,40)
        ax.set_xticks([-.5,0,.5]);ax.set_yticks([-.5,0,.5]);ax.set_title(f'{category} {skill} / '+('passed' if status else 'rejected')+f'\n{e["id"][:8]} / candidate {k}',fontsize=10)
    fig.suptitle('Initial RSI pose: native triangles + collision sample centers (schematic; radii not shown)')
    fig.savefig(a.audit/'preview.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
