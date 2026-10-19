"""Prepare original-scale local assets; flatten only fixed URDF joints and recenter frames."""
import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'runs/jointcarry_zeroshot/native_assets'


def xyz(node):
    if node is None: return np.zeros(3)
    if any(abs(float(v))>1e-9 for v in node.get('rpy','0 0 0').split()):
        raise ValueError('Rotated fixed link/geometry needs explicit transform handling')
    return np.array([float(v) for v in node.get('xyz','0 0 0').split()])


def prepare(source,name):
    source=Path(source).resolve(); tree=ET.parse(str(source)).getroot()
    dest=OUT/name
    if dest.exists(): raise ValueError('Preserve existing asset: '+str(dest))
    dest.mkdir(parents=True)
    links={n.get('name'):n for n in tree.findall('link')}
    positions={}; children=set()
    for joint in tree.findall('joint'):
        if joint.get('type')!='fixed': raise ValueError('Only rigid assets are supported')
        children.add(joint.find('child').get('link'))
    roots=set(links)-children
    assert len(roots)==1
    positions[next(iter(roots))]=np.zeros(3)
    remaining=list(tree.findall('joint'))
    while remaining:
        advanced=False
        for joint in remaining[:]:
            parent=joint.find('parent').get('link');child=joint.find('child').get('link')
            if parent in positions:
                positions[child]=positions[parent]+xyz(joint.find('origin'))
                remaining.remove(joint);advanced=True
        if not advanced: raise ValueError('Invalid joint tree')
    shapes=[];lower=[];upper=[];parts=[]
    for link_name,link in links.items():
        pos=positions[link_name]
        inertial=link.find('inertial'); mass=float(inertial.find('mass').get('value'))
        com=pos+xyz(inertial.find('origin'))
        inertia=inertial.find('inertia')
        tensor=None
        if inertia is not None:
            tensor=np.array([[float(inertia.get('i'+a+b,inertia.get('i'+b+a,'0'))) for b in 'xyz'] for a in 'xyz'])
        parts.append((mass,com,tensor))
        for kind in ['visual','collision']:
            for element in link.findall(kind):
                local=pos+xyz(element.find('origin')); geom=element.find('geometry'); shape=list(geom)[0]
                if shape.tag=='box':
                    half=np.array([float(v) for v in shape.get('size').split()])/2
                    lo,hi=local-half,local+half
                elif shape.tag=='cylinder':
                    half=np.array([float(shape.get('radius'))]*2+[float(shape.get('length'))/2]);lo,hi=local-half,local+half
                elif shape.tag=='mesh':
                    filename=shape.get('filename');mesh=(source.parent/filename).resolve()
                    vertices=np.array([[float(v) for v in line.split()[1:4]] for line in mesh.read_text().splitlines() if line.startswith('v ')])
                    scale=np.array([float(v) for v in shape.get('scale','1 1 1').split()])
                    vertices=vertices*scale;lo=vertices.min(0)+local;hi=vertices.max(0)+local
                    # Copy geometry and texture files, never run loaders against source data.
                    relative=Path(filename)
                    assert not relative.is_absolute() and '..' not in relative.parts
                    target=dest/relative.parent
                    if not target.exists(): shutil.copytree(mesh.parent,target)
                else: raise ValueError('Unsupported shape '+shape.tag)
                shapes.append((kind,copy.deepcopy(element),local))
                if kind=='collision':lower.append(lo);upper.append(hi)
    low=np.min(lower,axis=0);high=np.max(upper,axis=0);center=(low+high)/2;size=high-low
    robot=ET.Element('robot',name=name);flat=ET.SubElement(robot,'link',name='payload')
    total=sum(p[0] for p in parts);com=sum(p[0]*p[1] for p in parts)/total
    tensor=np.zeros((3,3));derived=False
    for mass,part_com,part_tensor in parts:
        if part_tensor is None:
            derived=True
            part_tensor=np.diag(mass*np.array([size[1]**2+size[2]**2,size[0]**2+size[2]**2,size[0]**2+size[1]**2])/12)
        delta=part_com-com
        tensor+=part_tensor+mass*(np.dot(delta,delta)*np.eye(3)-np.outer(delta,delta))
    inertial=ET.SubElement(flat,'inertial');ET.SubElement(inertial,'origin',xyz=' '.join(map(str,com-center)))
    ET.SubElement(inertial,'mass',value=str(total))
    ET.SubElement(inertial,'inertia',**{'i'+a+b:str(tensor[i,j]) for i,a in enumerate('xyz') for j,b in enumerate('xyz') if j>=i})
    for kind,element,origin in shapes:
        old=element.find('origin')
        if old is not None:element.remove(old)
        ET.SubElement(element,'origin',xyz=' '.join(map(str,origin-center)))
        flat.append(element)
    ET.ElementTree(robot).write(str(dest/'asset.urdf'),encoding='utf-8',xml_declaration=True)
    meta=dict(name=name,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),size=size.tolist(),mass=total,
              source_bbox_center=center.tolist(),source_links=len(links),normalization='none; bbox-center frame translation only',
              inertia='bbox approximation where source lacks inertia' if derived else 'source inertias combined with parallel-axis theorem',
              bbox_observation='actual collision-geometry bounds; 8 corners',native_scale=True)
    (dest/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


if __name__=='__main__':
    data=ROOT/'tokenhsi/data/dataset_dynamic_objects/test'
    selected=[(data/'Table_Circle/e446e818-363d-4bf5-b7dd-7e70bfca0b87/asset.urdf','tokenhsi_table_native'),
              (data/'ArmChair_Normal/002cd3a1-78b2-4a2e-aa01-5947dbe994f8/asset.urdf','tokenhsi_armchair_native')]
    selected += [(ROOT.parent/'TeamHOI/assets'/('%s.urdf'%name),'teamhoi_'+name) for name in
                 ['default_round','default_square','default_rectangle','large_round','large_square','large_rectangle']]
    catalog=[prepare(p,name) for p,name in selected]
    (OUT/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    for row in catalog: print(row['name'],row['size'],'mass',row['mass'])
