import torch
from isaacgym.torch_utils import quat_mul, quat_conjugate, quat_rotate, quat_from_angle_axis
from lpanlib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
from utils import torch_utils
from utils.motion_lib import MotionLib


def collision_shapes(asset, names, device):
    from pathlib import Path
    import xml.etree.ElementTree as ET
    asset = Path(asset)
    xml = ET.parse(asset).getroot()
    meshes = {mesh.attrib['name']: asset.parent/mesh.attrib['file'] for mesh in xml.findall('asset/mesh')}
    ids, points, radii = [], [], []
    for body in xml.findall('.//body'):
        for geom in body.findall('geom'):
            spec = geom.attrib
            radius = 0.
            offset = [float(v) for v in spec.get('pos', '0 0 0').split()]
            if spec.get('type') == 'mesh':
                vertices = [[float(v) for v in line.split()[1:4]] for line in meshes[spec['mesh']].read_text().splitlines() if line.startswith('v ')]
                vertices = [[v+a for v, a in zip(point, offset)] for point in vertices]
            elif 'fromto' in spec:
                values = [float(v) for v in spec['fromto'].split()]
                vertices = [values[:3], values[3:]]
                radius = float(spec['size'].split()[0])
            else:
                vertices = [offset]
                radius = float(spec['size'].split()[0])
            ids.extend([names.index(body.attrib['name'])]*len(vertices))
            points.extend(vertices)
            radii.extend([radius]*len(vertices))
    return (torch.tensor(ids, device=device), torch.tensor(points, device=device),
            torch.tensor(radii, device=device))


def face_retreat(positions, rotations, face, normal, shapes, clearance):
    ids, points, radii = shapes
    q = rotations[:, ids]
    world = positions[:, ids]+quat_rotate(q.reshape(-1, 4), points[None].expand(len(positions), -1, -1).reshape(-1, 3)).reshape(len(positions), -1, 3)
    support = ((world-face[:, None])*normal[:, None]).sum(-1)+radii
    return (support.amax(-1)+clearance).clamp_min(0)


class EvalPushMotionLib(MotionLib):
    def __init__(self, source_ids, *args, **kwargs):
        self.source_ids = source_ids
        super().__init__(*args, **kwargs)

    def _fetch_motion_files(self, motion_file, skill):
        groups = super()._fetch_motion_files(motion_file, skill)
        return tuple([group[index] for index in self.source_ids] if group else [] for group in groups)


def reference_state(lib, ids, times, dof_body_ids, dof_offsets, correct_hand_height=True):
    root, _, dof, _, _, dof_vel, _ = lib.get_motion_state(ids, times)
    positions, rotations, velocities, angular = lib.get_motion_state_max(ids, times)
    if not correct_hand_height:
        return positions, rotations, velocities, angular, dof, dof_vel
    # Match MotionLib's quaternion interpolation before the reset-only arm rotation.
    f0, f1, blend = lib._calc_frame_blend(times, lib._motion_lengths[ids], lib._motion_num_frames[ids], lib._motion_dt[ids])
    local = torch_utils.slerp(lib.lrs[lib.length_starts[ids]+f0], lib.lrs[lib.length_starts[ids]+f1], blend[:, None, None])
    original_tree = lib._motions[0].skeleton_tree
    tree = SkeletonTree(original_tree.node_names, original_tree.parent_indices.cpu(), original_tree.local_translation.to(root.device))
    state = SkeletonState.from_rotation_and_root_translation(tree, local, root, is_local=True)
    names = tree.node_names
    parents = tree.parent_indices.tolist()
    for side in ('right', 'left'):
        shoulder, hand = names.index(side+'_upper_arm'), names.index(side+'_hand')
        p, q = state.global_translation, state.global_rotation
        arm = p[:, hand]-p[:, shoulder]
        length = arm.norm(dim=-1).clamp_min(1e-6)
        axis = torch.stack((arm[:, 1], -arm[:, 0], torch.zeros_like(length)), -1)
        axis = axis/axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        angle = torch.asin(((.6175-p[:, shoulder, 2])/length).clamp(-1, 1))-torch.asin((arm[:, 2]/length).clamp(-1, 1))
        delta = quat_from_angle_axis(angle, axis)
        parent = q[:, parents[shoulder]]
        local_delta = quat_mul(quat_mul(quat_conjugate(parent), delta), parent)
        local[:, shoulder] = quat_mul(local_delta, local[:, shoulder])
        subtree = [shoulder]
        for index in range(shoulder+1, len(names)):
            if parents[index] in subtree:
                subtree.append(index)
        pivot_vel = velocities[:, shoulder:shoulder+1].clone()
        shoulder_omega = angular[:, shoulder:shoulder+1].clone()
        parent_omega = angular[:, parents[shoulder]:parents[shoulder]+1].clone()
        offsets = p[:, subtree]-p[:, shoulder:shoulder+1]
        turns = delta[:, None].expand(-1, len(subtree), -1).reshape(-1, 4)
        rotated_offsets = quat_rotate(turns, offsets.reshape(-1, 3)).reshape_as(offsets)
        internal_velocity = velocities[:, subtree]-pivot_vel-torch.cross(shoulder_omega.expand_as(offsets), offsets, dim=-1)
        velocities[:, subtree] = pivot_vel+torch.cross(parent_omega.expand_as(offsets), rotated_offsets, dim=-1)+quat_rotate(turns, internal_velocity.reshape(-1, 3)).reshape_as(offsets)
        angular[:, subtree] = parent_omega+quat_rotate(turns, (angular[:, subtree]-shoulder_omega).reshape(-1, 3)).reshape_as(offsets)
        slot = dof_body_ids.index(shoulder)
        dof_vel[:, dof_offsets[slot]:dof_offsets[slot+1]] = 0
        state = SkeletonState.from_rotation_and_root_translation(tree, local, root, is_local=True)
    dof = lib._local_rotation_to_dof(local)
    return state.global_translation, state.global_rotation, velocities, angular, dof, dof_vel
