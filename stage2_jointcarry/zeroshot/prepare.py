"""Copy and normalize an OBJ into a local single-link payload asset (CPU only)."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def prepare(source, output, size=(.52, .8, .4)):
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    lines = source.read_text().splitlines()
    vertices = np.array([[float(v) for v in line.split()[1:4]]
                         for line in lines if line.startswith('v ')])
    low, high = vertices.min(0), vertices.max(0)
    extent = high-low
    if not np.isfinite(vertices).all() or (extent <= 0).any():
        raise ValueError('Invalid source mesh bounds')
    size = np.asarray(size, dtype=float)
    vertices = (vertices-(low+high)/2)*(size/extent)
    result=[]; cursor=0
    for line in lines:
        if line.startswith('v '):
            result.append('v '+' '.join('%.9g'%x for x in vertices[cursor])); cursor+=1
        elif line.startswith('f '):
            result.append('f '+' '.join(token.split('/')[0] for token in line.split()[1:]))
    (output/'mesh.obj').write_text('\n'.join(result)+'\n')
    mass=float(np.prod(size)*100.)
    robot=ET.Element('robot',name='zero_shot_payload')
    link=ET.SubElement(robot,'link',name='payload')
    inertial=ET.SubElement(link,'inertial')
    ET.SubElement(inertial,'mass',value=str(mass))
    inertia=mass*np.array([size[1]**2+size[2]**2,size[0]**2+size[2]**2,size[0]**2+size[1]**2])/12
    ET.SubElement(inertial,'inertia',ixx=str(inertia[0]),iyy=str(inertia[1]),izz=str(inertia[2]),ixy='0',ixz='0',iyz='0')
    for kind in ('visual','collision'):
        part=ET.SubElement(link,kind)
        geometry=ET.SubElement(part,'geometry')
        ET.SubElement(geometry,'mesh',filename='mesh.obj')
    ET.ElementTree(robot).write(str(output/'asset.urdf'),encoding='utf-8',xml_declaration=True)
    metadata=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_extent=extent.tolist(),size=size.tolist(),mass=mass,
                  normalization='anisotropic_bbox_fit',observation='original_8_bbox_corners',
                  reward='original_box_anchor_proxy',physics_validated=False)
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source'); parser.add_argument('output')
    args=parser.parse_args()
    print(json.dumps(prepare(args.source,args.output),indent=2))
