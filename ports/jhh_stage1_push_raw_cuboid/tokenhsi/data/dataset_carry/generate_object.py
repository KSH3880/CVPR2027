import sys
sys.path.append("./")

import os
import os.path as osp
import torch
import trimesh
import numpy as np
import torchgeometry as tgm
import glob

from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from lpanlib.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from lpanlib.poselib.core.rotation3d import quat_mul, quat_from_angle_axis, quat_mul_norm, quat_rotate, quat_identity

from lpanlib.isaacgym_utils.vis.api import vis_hoi_use_scenepic_animation
from lpanlib.others.colors import name_to_rgb

if __name__ == '__main__':

    humanoid = "phys_humanoid_v3"

    all_files = glob.glob(osp.join(osp.dirname(__file__), "motions/*/*/{}/ref_motion.npy".format(humanoid)))

    humanoid_xml_path = osp.join(osp.dirname(__file__), "../assets/mjcf/{}.xml".format(humanoid))

    # selected motions
    candidates = {
        "carry": {
            "ACCAD+__+Female1General_c3d+__+A5_-_pick_up_box_stageii": {
                "box_size": [0.4, 0.4, 0.4],
                "moving_frames": [90, 140],
                "contact_offset": [0.0, 0.0, 0.0],
            },
            "ACCAD+__+Female1Walking_c3d+__+B19_-_walk_to_pick_up_box_stageii": {
                "box_size": [0.4, 0.4, 0.4],
                "moving_frames": [65, 95],
                "contact_offset": [0.0, 0.0, 0.0],
            },
            "ACCAD+__+Female1Walking_c3d+__+B20_-_walk_with_box_stageii": {
                "box_size": [0.4, 0.4, 0.4],
                "moving_frames": [0, 128],
                "contact_offset": [0.0, 0.0, 0.0],
            },
            "ACCAD+__+Female1Walking_c3d+__+B21_-_put_down_box_to_walk_stageii": {
                "box_size": [0.4, 0.4, 0.4],
                "moving_frames": [0, 130],
                "contact_offset": [-0.05, -0.05, 0.0],
            },
        },
    }

    for f in all_files:
        skill = f.split("/")[-4]
        seq_name = f.split("/")[-3]

        if skill in list(candidates.keys()) and seq_name in list(candidates[skill].keys()):

            print("processing [skill: {}] [seq_name: {}]".format(skill, seq_name))

            # load motion
            motion = SkeletonMotion.from_file(f)

            seq_label = candidates[skill][seq_name]

            # create box
            mesh = trimesh.creation.box(seq_label["box_size"])

            # fitting
            num_frames = motion.root_translation.shape[0]
            box_trans = np.zeros((num_frames, 3), dtype=np.float32)
            box_rot = np.zeros((num_frames, 4), dtype=np.float32)

            start = seq_label["moving_frames"][0]
            end = seq_label["moving_frames"][1]
            box_trans[start : end + 1] = (
                (motion.global_translation[start : end + 1, motion.skeleton_tree.index("left_hand")] + motion.global_translation[start : end + 1, motion.skeleton_tree.index("right_hand")]) / 2
            ).numpy()

            # making box on the ground plane
            min_box_trans = np.min(box_trans[start : end + 1, 2])
            if min_box_trans < seq_label["box_size"][2] / 2:
                height_offset = seq_label["box_size"][2] / 2 - min_box_trans
                box_trans[start : end + 1, 2] += height_offset
            
            box_trans[:start] = box_trans[start]
            box_trans[end:] = box_trans[end]

            # compute rotation of the box
            vec = (motion.global_translation[start : end + 1, motion.skeleton_tree.index("right_hand")] - motion.global_translation[start : end + 1, motion.skeleton_tree.index("left_hand")])
            vec_up_axis = np.zeros_like(vec)
            vec_up_axis[:, 2] = 1 # z up
            target_facing_dir = np.cross(vec_up_axis, vec)
            target_facing_dir[:, 2] = 0 # only consider direction on xy plane
            init_facing_dir = np.zeros_like(vec)
            init_facing_dir[:, 1] = 1 # unified to y axis as the default facing dir!!!! Dec 14 2023!!!

            angles_degree = []
            quats = []
            for i in range(init_facing_dir.shape[0]):
                cosine_angle = np.dot(target_facing_dir[i], init_facing_dir[i]) / (np.linalg.norm(target_facing_dir[i]) * np.linalg.norm(init_facing_dir[i]))
                angle_radian = np.arccos(cosine_angle)
                angles_degree.append(angle_radian * 180 / np.pi)

                if np.cross(init_facing_dir[i], target_facing_dir[i])[-1] > 0:
                    coeff = 1
                else:
                    coeff = -1
                
                aa = torch.tensor([0, 0, coeff * angle_radian])
                quat = tgm.angle_axis_to_quaternion(aa).numpy() # angle axis ---> quaternion
                quat = quat[[1, 2, 3, 0]] # switch quaternion order wxyz -> xyzw
                quats.append(quat)

            box_rot[start : end + 1] = np.array(quats)
            box_rot[:start] = box_rot[start]
            box_rot[end:] = box_rot[end]

            # making box collision-free with the human
            box_trans[:, 0] += seq_label["contact_offset"][0]
            box_trans[:, 1] += seq_label["contact_offset"][1]
            box_trans[:, 2] += seq_label["contact_offset"][2]

            box_motion = np.concatenate([box_trans, box_rot], axis=-1)
            save_path = osp.join(osp.dirname(f), "box_motion.npy")
            np.save(save_path, box_motion)

            # scenepic animation
            obj_meshes = [mesh]
            obj_global_pos = box_trans[:, np.newaxis, :] # (N_frames, N_objs, 3)
            obj_global_rot = box_rot[:, np.newaxis, :] # (N_frames, N_objs, 4)
            obj_colors = [name_to_rgb['LightYellow'] * 255]
            vis_hoi_use_scenepic_animation(
                asset_filename=humanoid_xml_path,
                rigidbody_global_pos=motion.global_translation,
                rigidbody_global_rot=motion.global_rotation,
                fps=motion.fps,
                up_axis="z",
                color=name_to_rgb['AliceBlue'] * 255,
                output_path=osp.join(osp.dirname(f), "box_motion_render.html"),
                obj_meshes=obj_meshes,
                obj_global_pos=obj_global_pos,
                obj_global_rot=obj_global_rot,
                obj_colors=obj_colors
            )
