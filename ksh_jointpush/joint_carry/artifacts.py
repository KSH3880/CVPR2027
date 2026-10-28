import hashlib
from functools import lru_cache
from pathlib import Path
from tokenhsi.utils.grasp_geometry import ANCHOR_INSET


@lru_cache(None)
def contract_digest(source='single', inset=ANCHOR_INSET):
    digest = hashlib.sha256()
    digest.update(('source='+source+';inset='+str(inset)).encode())
    paths = ['joint_carry/build_size_rsi.py', 'joint_carry/size_candidates.py', 'joint_carry/correction.py',
             'joint_carry/kinematics.py', 'joint_carry/leg_correction.py', 'joint_carry/geometry.py', 'joint_carry/size_screen.py',
             'joint_carry/artifacts.py', 'tokenhsi/utils/grasp_geometry.py', 'joint_carry/single_source.py',
             'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml',
             'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml']
    if source == 'single':
        from joint_carry.single_source import source_paths
        paths += [str(path) for skill in ('pickUp', 'carryWith', 'putDown') for path in source_paths(skill)]
    else:
        paths += [str(path) for path in sorted(Path('joint_carry/source_rsi').rglob('cooperative_rsi_candidate.npz'))]
    for path in paths:
        digest.update(path.encode())
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()
