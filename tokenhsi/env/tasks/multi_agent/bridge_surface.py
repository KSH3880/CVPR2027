from pathlib import Path


def write_asset(directory, size, density, thickness, controlled=False):
    x, y, z = size
    mass = x*y*z*density
    inertia = [mass*(y*y+z*z)/12, mass*(x*x+z*z)/12, mass*(x*x+y*y)/12]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    kind = 'controlled' if controlled else 'push'
    key = '_'.join(f'{v:.9g}'.replace('.', 'p') for v in (*size, density, thickness))
    path = directory/f'bridge_{kind}_{key}.urdf'
    shape = lambda height: f'<geometry><box size="{x} {y} {height}"/></geometry>'
    path.write_text(f'<robot name="bridge_surface"><link name="bridge">'
        f'<inertial><mass value="{mass}"/><inertia ixx="{inertia[0]}" iyy="{inertia[1]}" izz="{inertia[2]}" ixy="0" ixz="0" iyz="0"/></inertial>'
        f'<visual>{shape(z)}</visual>'
        f'<collision name="base"><origin xyz="0 0 {-thickness/2}"/>{shape(z-thickness)}</collision>'
        f'<collision name="top_cap"><origin xyz="0 0 {(z-thickness)/2}"/>{shape(thickness)}</collision>'
        '</link></robot>')
    return path


def load_asset(gym, sim, directory, size, options, thickness, controlled=False):
    path = write_asset(directory, size, options.density, thickness, controlled)
    asset = gym.load_asset(sim, str(path.parent), path.name, options)
    if asset is None or gym.get_asset_rigid_body_count(asset) != 1 or gym.get_asset_rigid_shape_count(asset) != 2:
        raise RuntimeError('Bridge surface requires one body and two collision shapes')
    return asset
