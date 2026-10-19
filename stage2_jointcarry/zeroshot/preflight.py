"""CPU-only bounds, topology and local-write checks; imports no CUDA libraries."""
import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

root=Path(__file__).resolve().parents[1]
for path in [root/'zeroshot/payload.py',root/'zeroshot/prepare.py',root/'tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py']:
    ast.parse(path.read_text())
for path in (root/'zeroshot/assets').glob('*/asset.urdf'):
    tree=ET.parse(str(path)).getroot()
    assert len(tree.findall('link'))==1 and not tree.findall('joint')
    manifest=json.loads((path.parent/'manifest.json').read_text())
    verts=np.array([[float(v) for v in line.split()[1:4]] for line in (path.parent/'mesh.obj').read_text().splitlines() if line.startswith('v ')])
    np.testing.assert_allclose(verts.max(0)-verts.min(0),manifest['size'],atol=1e-7)
    np.testing.assert_allclose(verts.max(0)+verts.min(0),0,atol=1e-7)
    for line in (path.parent/'mesh.obj').read_text().splitlines():
        if line.startswith('f '):
            ids=[int(v) for v in line.split()[1:]]
            assert all(1<=v<=len(verts) for v in ids)
    print(path.parent.name,'CPU_PREFLIGHT_OK',len(verts),'vertices')
