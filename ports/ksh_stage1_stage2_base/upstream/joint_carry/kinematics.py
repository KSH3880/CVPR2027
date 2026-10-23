import numpy as np
from scipy.spatial.transform import Rotation as R

DOF_BODIES = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_SIZES = [3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3, 3]

def forward_kinematics(local_q, root, parents, offsets):
    xyz = np.empty(local_q.shape[:-1] + (3,))
    rotations = []
    for j, parent in enumerate(parents):
        local = R.from_quat(local_q[:, j])
        rotations.append(local if parent < 0 else rotations[parent] * local)
        xyz[:, j] = root if parent < 0 else xyz[:, parent] + rotations[parent].apply(offsets[j])
    return xyz, np.stack([r.as_quat() for r in rotations], axis=1)

def angular_velocity(q, fps, local=False):
    """Quaternion finite differences; local=True matches MotionLib DOF convention."""
    shape = q.shape
    a, b = R.from_quat(q[:-1].reshape(-1, 4)), R.from_quat(q[1:].reshape(-1, 4))
    delta = a.inv() * b if local else b * a.inv()
    v = delta.as_rotvec().reshape((shape[0] - 1,) + shape[1:-1] + (3,)) * fps
    return np.concatenate([v, v[-1:]], axis=0)

def pack(local_q, roots, parents, offsets, box, size, fps, names):
    positions, rotations = zip(*(forward_kinematics(local_q[:, a], roots[:, a], parents, offsets)
                                for a in range(roots.shape[1])))
    positions, rotations = np.stack(positions, 1), np.stack(rotations, 1)
    angles = R.from_quat(local_q.reshape(-1, 4)).as_rotvec().reshape(local_q.shape[:-1] + (3,))
    local_vel = angular_velocity(local_q, fps, local=True)
    dof = np.concatenate([angles[:, :, j] if n == 3 else angles[:, :, j, 1:2]
                          for j, n in zip(DOF_BODIES, DOF_SIZES)], axis=-1)
    dof_vel = np.concatenate([local_vel[:, :, j] if n == 3 else local_vel[:, :, j, 1:2]
                              for j, n in zip(DOF_BODIES, DOF_SIZES)], axis=-1)
    return dict(fps=np.asarray(fps), time=np.arange(len(box)) / fps,
                joint_names=np.asarray(names), parent_indices=parents, local_translation=offsets,
                local_rotation_xyzw=local_q, root_position=roots,
                root_rotation_xyzw=local_q[:, :, 0], dof_position=dof, dof_velocity=dof_vel,
                body_position=positions, body_rotation_xyzw=rotations,
                body_linear_velocity=np.gradient(positions, 1 / fps, axis=0),
                body_angular_velocity=angular_velocity(rotations, fps),
                box_position=box[:, :3], box_rotation_xyzw=box[:, 3:], box_size=np.asarray(size),
                box_linear_velocity=np.gradient(box[:, :3], 1 / fps, axis=0),
                box_angular_velocity=angular_velocity(box[:, 3:], fps),
                physics_validated=np.asarray(False), rsi_ready=np.asarray(False))
