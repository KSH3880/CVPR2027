"""Evaluation-only chair mesh with the policy's original sampled AABB contract."""
import os
from pathlib import Path
from isaacgym import gymapi


def load_chair(gym, sim, size, options, output):
    source = Path(os.environ['CHAIR_MESH']) if os.environ.get('CHAIR_MESH') else Path(__file__).resolve().parents[2] / 'chair_assets/original.obj'
    lines = source.read_text().splitlines()
    vertices = [list(map(float, line.split()[1:4])) for line in lines if line.startswith('v ')]
    low = [min(v[a] for v in vertices) for a in range(3)]
    high = [max(v[a] for v in vertices) for a in range(3)]
    dimensions = [float(x) for x in size]
    folder = Path(output).resolve() / 'chair_assets' / source.parent.name / ('_'.join('%.3f' % x for x in dimensions))
    folder.mkdir(parents=True, exist_ok=True)
    mesh = []
    for v in vertices:
        mesh.append('v ' + ' '.join(str((v[a] - (low[a]+high[a])/2) * dimensions[a]/(high[a]-low[a])) for a in range(3)))
    for line in lines:
        if line.startswith('f '):
            mesh.append('f ' + ' '.join(x.split('/')[0] for x in line.split()[1:]))
    (folder/'chair.obj').write_text('\n'.join(mesh)+'\n')
    (folder/'asset.urdf').write_text('<robot name="chair"><link name="chair"><visual><geometry><mesh filename="chair.obj"/></geometry></visual><collision><geometry><mesh filename="chair.obj"/></geometry></collision></link></robot>')
    options.vhacd_enabled = True
    options.vhacd_params.resolution = 100000
    options.vhacd_params.max_convex_hulls = 32
    options.vhacd_params.max_num_vertices_per_ch = 64
    options.override_com = True
    options.override_inertia = True
    asset = gym.load_asset(sim, str(folder), 'asset.urdf', options)
    if asset is None or gym.get_asset_rigid_body_count(asset) != 1 or gym.get_asset_dof_count(asset) != 0:
        raise RuntimeError('Chair must be a single rigid body with no DOFs')
    print('[chair probe] payload AABB={} collision=VHACD path={}'.format(dimensions, folder), flush=True)
    return asset
