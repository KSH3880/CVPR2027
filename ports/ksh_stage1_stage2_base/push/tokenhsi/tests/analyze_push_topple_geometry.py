"""Measure actual foot-mesh vertices inside initial oriented boxes in a replay."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def rotate(q,v):
    q=np.asarray(q);v=np.asarray(v);t=2*np.cross(q[:3],v)
    return v+q[3]*t+np.cross(q[:3],t)


def analyze(source,asset):
    report=json.loads(source.read_text());tree=ET.parse(asset).getroot()
    meshes={m.get('name'):asset.parent/m.get('file') for m in tree.findall('./asset/mesh')}
    rows=[]
    for ci,case in enumerate(report['cases']):
        state=case['trace'][0]
        if state['age']!=0 or 'body_rot' not in state:
            continue
        box=np.asarray(state['box']);half=np.asarray(case['box_size'])/2
        inv=box[3:7].copy();inv[:3]*=-1
        overlaps=[]
        for name in ('left_foot','right_foot'):
            i=report['body_names'].index(name)
            vertices=np.asarray([[float(x) for x in line.split()[1:4]] for line in meshes[name+'_mesh'].read_text().splitlines() if line.startswith('v ')])
            world=rotate(state['body_rot'][i],vertices)+np.asarray(state['body'][i])
            local=rotate(inv,world-box[:3]);depth=(half-np.abs(local)).min(-1)
            if depth.max()>1e-5:
                overlaps.append(dict(body=name,inside_vertices=int((depth>1e-5).sum()),max_vertex_inside_depth_m=float(depth.max())))
        rows.append(dict(case=ci,env=case['env'],owner=case['owner'],kind=case['kind'],age45=case['age'],
                         root_center_distance_xy_m=float(np.linalg.norm(np.asarray(state['root'])[:2]-box[:2])),initial_foot_overlaps=overlaps))
    destination=source.with_name('initial_foot_overlap.json')
    destination.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('--asset',type=Path,default=Path('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml'))
    args=parser.parse_args();analyze(args.source,args.asset)
