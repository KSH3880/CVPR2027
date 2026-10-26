from pathlib import Path


def plate_box_urdf(size, mass, plate):
    x, y, z = size
    thickness, inset = plate['thickness'], plate['inset']
    if not 0 < thickness < z or not 0 <= 2*inset < min(x, y):
        raise ValueError('PUSH bottom plate must fit inside the box bounds')
    ix, iy, iz = (mass*(y*y+z*z)/12, mass*(x*x+z*z)/12, mass*(x*x+y*y)/12)
    return f'''<robot name="push_plate_box">
  <link name="box">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ix}" ixy="0" ixz="0" iyy="{iy}" iyz="0" izz="{iz}"/>
    </inertial>
    <visual><geometry><box size="{x} {y} {z}"/></geometry></visual>
    <collision name="body">
      <origin xyz="0 0 {thickness/2}"/>
      <geometry><box size="{x} {y} {z-thickness}"/></geometry>
    </collision>
    <collision name="bottom_plate">
      <origin xyz="0 0 {-z/2+thickness/2}"/>
      <geometry><box size="{x-2*inset} {y-2*inset} {thickness}"/></geometry>
    </collision>
  </link>
</robot>
'''


def create_plate_box(gym, sim, size, mass, options, plate, directory, index):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'box_{index}.urdf'
    path.write_text(plate_box_urdf(size, mass, plate))
    asset = gym.load_asset(sim, str(directory.resolve()), path.name, options)
    if gym.get_asset_rigid_body_count(asset) != 1 or gym.get_asset_rigid_shape_count(asset) != 2:
        raise ValueError('PUSH plate box requires one rigid body and two collision shapes')
    return asset
