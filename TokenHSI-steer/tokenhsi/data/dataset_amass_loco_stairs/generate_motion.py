import sys
sys.path.append("./")

import os
import os.path as osp
import torch
import numpy as np
import torchgeometry as tgm
from tqdm import tqdm

from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from lpanlib.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from lpanlib.poselib.core.rotation3d import quat_mul, quat_from_angle_axis, quat_mul_norm, quat_rotate, quat_identity

from body_models.model_loader import get_body_model

from lpanlib.isaacgym_utils.vis.api import vis_motion_use_scenepic_animation
from lpanlib.others.colors import name_to_rgb

from tokenhsi.data.data_utils import project_joints, project_joints_simple

joints_to_use = {
    "from_smpl_original_to_amp_humanoid": np.array([0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]),
    "from_smpl_original_to_phys_humanoid_v3": np.array([0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]),
}

if __name__ == "__main__":

    # load skeleton of smpl_humanoid
    smpl_humanoid_xml_path = osp.join(osp.dirname(__file__), "../assets/mjcf/smpl_humanoid.xml")
    smpl_humanoid_skeleton = SkeletonTree.from_mjcf(smpl_humanoid_xml_path)

    # load skeleton of amp_humanoid
    amp_humanoid_xml_path = osp.join(osp.dirname(__file__), "../assets/mjcf/amp_humanoid.xml")
    amp_humanoid_skeleton = SkeletonTree.from_mjcf(amp_humanoid_xml_path)

    # load skeleton of phys_humanoid_v3
    phys_humanoid_v3_xml_path = osp.join(osp.dirname(__file__), "../assets/mjcf/phys_humanoid_v3.xml")
    phys_humanoid_v3_skeleton = SkeletonTree.from_mjcf(phys_humanoid_v3_xml_path)

    # load skeleton of smpl_original
    bm = get_body_model("SMPL", "NEUTRAL", batch_size=1, debug=False)
    jts_global_trans = bm().joints[0, :24, :].cpu().detach().numpy()
    jts_local_trans = np.zeros_like(jts_global_trans)
    for i in range(jts_local_trans.shape[0]):
        parent = bm.parents[i]
        if parent == -1:
            jts_local_trans[i] = jts_global_trans[i]
        else:
            jts_local_trans[i] = jts_global_trans[i] - jts_global_trans[parent]

    skel_dict = smpl_humanoid_skeleton.to_dict()
    skel_dict["node_names"] = [
        "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine", "L_Ankle", "R_Ankle",
        "Chest", "L_Toe", "R_Toe", "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder", 
        "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
    ]
    skel_dict["parent_indices"]["arr"] = bm.parents.numpy()
    skel_dict["local_translation"]["arr"] = jts_local_trans
    smpl_original_skeleton = SkeletonTree.from_dict(skel_dict)

    # create tposes
    smpl_original_tpose = SkeletonState.zero_pose(smpl_original_skeleton)
    
    amp_humanoid_tpose = SkeletonState.zero_pose(amp_humanoid_skeleton)
    local_rotation = amp_humanoid_tpose.local_rotation
    local_rotation[amp_humanoid_skeleton.index("left_upper_arm")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
        local_rotation[amp_humanoid_skeleton.index("left_upper_arm")]
    )
    local_rotation[amp_humanoid_skeleton.index("right_upper_arm")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
        local_rotation[amp_humanoid_skeleton.index("right_upper_arm")]
    )

    phys_humanoid_v3_tpose = SkeletonState.zero_pose(phys_humanoid_v3_skeleton)
    local_rotation = phys_humanoid_v3_tpose.local_rotation
    local_rotation[phys_humanoid_v3_skeleton.index("left_upper_arm")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
        local_rotation[phys_humanoid_v3_skeleton.index("left_upper_arm")]
    )
    local_rotation[phys_humanoid_v3_skeleton.index("right_upper_arm")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
        local_rotation[phys_humanoid_v3_skeleton.index("right_upper_arm")]
    )

    # input/output dirs
    input_dir = osp.join(osp.dirname(__file__), "smpl_params")
    output_dir = osp.join(osp.dirname(__file__), "motions")

    os.makedirs(output_dir, exist_ok=True)

    data_list = os.listdir(input_dir)
    pbar = tqdm(data_list)
    for fname in pbar:
        pbar.set_description(fname)

        subset_name = fname.split("+__+")[0]
        subject = fname.split("+__+")[1]
        action = fname.split("+__+")[2][:-4]

        curr_output_dir = osp.join(output_dir, subset_name, fname[:-4])

        os.makedirs(curr_output_dir, exist_ok=True)

        # load SMPL params
        raw_params = np.load(osp.join(input_dir, fname), allow_pickle=True).item()
        poses = torch.tensor(raw_params["poses"], dtype=torch.float32)
        trans = torch.tensor(raw_params["trans"], dtype=torch.float32)
        fps = raw_params["fps"]

        # compute world absolute position of root joint
        trans = bm(
            global_orient=poses[:, 0:3], 
            body_pose=poses[:, 3:72],
            transl=trans[:, :],
        ).joints[:, 0, :].cpu().detach()

        poses = poses.reshape(-1, 24, 3)

        # angle axis ---> quaternion
        poses_quat = tgm.angle_axis_to_quaternion(poses.reshape(-1, 3)).reshape(poses.shape[0], -1, 4)

        # switch quaternion order
        # wxyz -> xyzw
        poses_quat = poses_quat[:, :, [1, 2, 3, 0]]

        # generate motion
        skeleton_state = SkeletonState.from_rotation_and_root_translation(smpl_original_skeleton, poses_quat, trans, is_local=True)
        motion = SkeletonMotion.from_skeleton_state(skeleton_state, fps=fps)

        # plot_skeleton_motion_interactive(motion)

        ################ retarget ################

        configs = {
            "amp_humanoid": {
                "skeleton": amp_humanoid_skeleton,
                "xml_path": amp_humanoid_xml_path,
                "tpose": amp_humanoid_tpose,
                "joints_to_use": joints_to_use["from_smpl_original_to_amp_humanoid"],
                "root_height_offset": 0.05,
            },
            "phys_humanoid_v3": {
                "skeleton": phys_humanoid_v3_skeleton,
                "xml_path": phys_humanoid_v3_xml_path,
                "tpose": phys_humanoid_v3_tpose,
                "joints_to_use": joints_to_use["from_smpl_original_to_phys_humanoid_v3"],
                "root_height_offset": 0.07,
            },
        }

        ###### retargeting ######
        for k, v in configs.items():

            target_origin_global_rotation = v["tpose"].global_rotation.clone()

            target_aligned_global_rotation = quat_mul_norm( 
                torch.tensor([-0.5, -0.5, -0.5, 0.5]), target_origin_global_rotation
            )

            target_final_global_rotation = quat_mul_norm(
                skeleton_state.global_rotation.clone()[..., v["joints_to_use"], :], target_aligned_global_rotation.clone()
            )
            target_final_root_translation = skeleton_state.root_translation.clone()

            new_skeleton_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree=v["skeleton"],
                r=target_final_global_rotation,
                t=target_final_root_translation,
                is_local=False,
            ).local_repr()
            new_motion = SkeletonMotion.from_skeleton_state(new_skeleton_state, fps=fps)

            new_motion_params_root_trans = new_motion.root_translation.clone()
            new_motion_params_local_rots = new_motion.local_rotation.clone()

            # check foot-ground penetration
            if "stair" not in fname:
                min_h = torch.min(new_motion.global_translation[:, :, 2], dim=-1)[0].mean()
            else:
                min_h = torch.min(new_motion.global_translation[:, :, 2], dim=-1)[0].min()

            for i in range(new_motion.global_translation.shape[0]):
                new_motion_params_root_trans[i, 2] += -min_h

            # adjust the height of the root to avoid ground penetration
            root_height_offset = v["root_height_offset"]
            new_motion_params_root_trans[:, 2] += root_height_offset

            # update new_motion
            new_skeleton_state = SkeletonState.from_rotation_and_root_translation(v["skeleton"], new_motion_params_local_rots, new_motion_params_root_trans, is_local=True)
            new_motion = SkeletonMotion.from_skeleton_state(new_skeleton_state, fps=fps)

            if k == "amp_humanoid":
                new_motion = project_joints(new_motion)
            elif k == "phys_humanoid" or k == "phys_humanoid_v2" or k == "phys_humanoid_v3":
                new_motion = project_joints_simple(new_motion)
            else:
                pass

            # save retargeted motion
            save_dir = osp.join(curr_output_dir, k)
            os.makedirs(save_dir, exist_ok=True)
            save_path = osp.join(save_dir, "ref_motion.npy")
            new_motion.to_file(save_path)

            # plot_skeleton_motion_interactive(new_motion)

            # scenepic animation
            vis_motion_use_scenepic_animation(
                asset_filename=v["xml_path"],
                rigidbody_global_pos=new_motion.global_translation,
                rigidbody_global_rot=new_motion.global_rotation,
                fps=fps,
                up_axis="z",
                color=name_to_rgb['AliceBlue'] * 255,
                output_path=osp.join(save_dir, "ref_motion_render.html"),
            )
