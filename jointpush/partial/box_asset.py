"""One rigid body, two collision shapes: grippy body and low-friction sole."""
from pathlib import Path
import hashlib


def write_box_asset(directory, size, mass, thickness=.01):
    x, y, z = map(float, size)
    mass = float(mass)
    thickness = float(thickness)
    if min(x, y, z) <= 2 * thickness or mass <= 0:
        raise ValueError('Invalid partial box geometry/mass')
    key = hashlib.sha256(repr((x, y, z, mass, thickness)).encode()).hexdigest()
    path = Path(directory) / (key + '.urdf')
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit mass/COM/inertia preserve the full cuboid physical contract.
    ixx, iyy, izz = (mass*(y*y+z*z)/12, mass*(x*x+z*z)/12, mass*(x*x+y*y)/12)
    path.write_text(f'''<?xml version="1.0"?>
<robot name="partial_box"><link name="box">
<inertial><origin xyz="0 0 0"/><mass value="{mass}"/><inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/></inertial>
<visual><geometry><box size="{x} {y} {z}"/></geometry></visual>
<collision name="body"><origin xyz="0 0 {thickness/2}"/><geometry><box size="{x} {y} {z-thickness}"/></geometry></collision>
<collision name="bottom"><origin xyz="0 0 {-z/2+thickness/2}"/><geometry><box size="{x} {y} {thickness}"/></geometry></collision>
</link></robot>''')
    return path
