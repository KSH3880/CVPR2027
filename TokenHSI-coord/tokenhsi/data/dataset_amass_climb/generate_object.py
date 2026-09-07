import sys
sys.path.append("./")

import os
import os.path as osp
import numpy as np
import glob
import trimesh
import yaml
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from tokenhsi.utils import torch_utils

import torch
import torchgeometry as tgm
import PIL
import json

from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

from lpanlib.isaacgym_utils.vis.api import vis_hoi_use_scenepic_animation_climb
from lpanlib.others.colors import name_to_rgb

from tokenhsi.data.dataset_sit.generate_object import get_heightmap_raytrace

# load humanoid motions
all_files = glob.glob(osp.join(osp.dirname(__file__), "motions/*/*/phys_humanoid_v3/ref_motion.npy"))
humanoid_xml_path = osp.join(osp.dirname(__file__), "../assets/mjcf/phys_humanoid_v3.xml")

motion_file = osp.join(osp.dirname(__file__), "dataset_amass_climb.yaml")
ext = os.path.splitext(motion_file)[1]
if (ext == ".yaml"):

    with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
        motion_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    candidates = [data['file'].split("/")[2] for data in motion_config["motions"]["climb"]] # only process these seqs

else:
    raise NotImplementedError

# load object meshes
dataset = "objects"
dataset_dir = osp.join(osp.dirname(__file__), dataset)
obj_sets = os.listdir(dataset_dir) # train or test
for set in obj_sets:
    set_dir = osp.join(dataset_dir, set)
    categories = os.listdir(set_dir)
    objects = {}
    for cat in categories:
        print("Loading object... set: {} category {}".format(set, cat))
        cat_dir = osp.join(set_dir, cat)
        obj_list = os.listdir(cat_dir)
        for obj_name in obj_list:
            
            curr_dir = osp.join(cat_dir, obj_name)
            mesh_with_texture = trimesh.load(osp.join(curr_dir, "geom/mesh.obj"), process=False)
            mesh_wo_texture = trimesh.Trimesh(vertices=mesh_with_texture.vertices, faces=mesh_with_texture.faces)
            objects["{}_{}_{}".format(set, cat, obj_name)] = mesh_wo_texture

for f in all_files:
    skill = f.split("/")[-4]
    seq_name = f.split("/")[-3]

    if seq_name in candidates:

        print("processing [skill: {}] [seq_name: {}]".format(skill, seq_name))
    
        # load motion
        motion = SkeletonMotion.from_file(f)

        left_foot_pos = motion.global_translation[-1, motion.skeleton_tree.index("left_foot")]
        right_foot_pos = motion.global_translation[-1, motion.skeleton_tree.index("right_foot")]

        object_center = (left_foot_pos + right_foot_pos) / 2
        object_center[-1] = 0.0

        # compute rotation of the object
        humanoid_root_rotation = torch_utils.calc_heading_quat(motion.global_root_rotation[-1].unsqueeze(0))
        humanoid_init_facing_dir = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0)
        humanoid_curr_facing_dir = torch_utils.quat_rotate(humanoid_root_rotation, humanoid_init_facing_dir).squeeze(0)

        if seq_name == "CMU+__+40+__+40_08_stageii":
            object_center += 0.3 * humanoid_curr_facing_dir
            object_center[-1] = 0.0
        elif seq_name == "CMU+__+107+__+107_10_stageii":
            object_center += 0.3 * humanoid_curr_facing_dir
            object_center[-1] = 0.0
        elif seq_name == "Eyes_Japan_Dataset+__+ichige+__+jump-13-matrix-ichige_stageii":
            object_center += 0.3 * humanoid_curr_facing_dir
            object_center[-1] = 0.0
        elif seq_name == "CMU+__+36+__+36_04_stageii":
            object_center += 0.25 * humanoid_curr_facing_dir
            object_center[-1] = 0.0
        else:
            pass

        object_init_facing_dir = np.array([0.0, 1.0, 0.0]) # unified to y axis as the default facing dir!!!! Dec 14 2023!!!
        object_target_facing_dir = -1 * humanoid_curr_facing_dir.numpy()

        cosine_angle = np.dot(object_target_facing_dir, object_init_facing_dir) / (np.linalg.norm(object_target_facing_dir) * np.linalg.norm(object_init_facing_dir))
        angle_radian = np.arccos(cosine_angle)
        angles_degree = angle_radian * 180 / np.pi

        if np.cross(object_init_facing_dir, object_target_facing_dir)[-1] > 0:
            coeff = 1
        else:
            coeff = -1
        
        aa = torch.tensor([0, 0, coeff * angle_radian])
        quat = tgm.angle_axis_to_quaternion(aa).numpy() # angle axis ---> quaternion
        quat = quat[[1, 2, 3, 0]] # switch quaternion order wxyz -> xyzw

        object_state = np.concatenate([object_center, quat], axis=-1)
        save_path = osp.join(osp.dirname(f), "object_state.npy")
        np.save(save_path, object_state)

        # scenepic animation
        obj_meshes = list(objects.values())
        obj_names = list(objects.keys())

        obj_global_pos = []
        for i in range(len(obj_names)):
            pos3d = object_center.clone().numpy()
            pos3d[-1] = (obj_meshes[i].vertices[:, -1].max() - obj_meshes[i].vertices[:, -1].min()) / 2.0
            obj_global_pos.append(pos3d)

        obj_global_rot = [quat for i in range(len(obj_names))] # (N_objs, 4)
        obj_colors = [name_to_rgb['LightYellow'] * 255 for i in range(len(obj_names))]
        vis_hoi_use_scenepic_animation_climb(
            asset_filename=humanoid_xml_path,
            rigidbody_global_pos=motion.global_translation,
            rigidbody_global_rot=motion.global_rotation,
            fps=motion.fps,
            up_axis="z",
            color=name_to_rgb['AliceBlue'] * 255,
            output_path=osp.join(osp.dirname(f), "object_state_render.html"),
            obj_meshes=obj_meshes,
            obj_global_pos=obj_global_pos,
            obj_global_rot=obj_global_rot,
            obj_colors=obj_colors,
            obj_names=obj_names
        )


# generate object config

for set in obj_sets:
    obj_dir = osp.join(dataset_dir, set)
    categories = os.listdir(set_dir)
    
    for categeory in categories:

        obj_ids = os.listdir(osp.join(obj_dir, categeory))
        for id in obj_ids:
            curr_dir = osp.join(obj_dir, categeory, id)
            mesh = trimesh.load(osp.join(curr_dir, "geom/mesh.obj"), process=False)

            cfg = {}

            # generate bbox points
            bbox_length = np.max(mesh.vertices, axis=0) - np.min(mesh.vertices, axis=0)
            bbox_center = (np.max(mesh.vertices, axis=0) + np.min(mesh.vertices, axis=0)) / 2
            assert bbox_center.sum() == 0.0
            cfg["bbox"] = bbox_length.tolist()
            cfg["center"] = bbox_center.tolist()
            cfg["facing"] = [0, 1, 0]

            # generate height map
            height_map = get_heightmap_raytrace(mesh.copy(), HEIGHT_MAP_DIM=128)
            curr_output_dir = osp.join(curr_dir, "height_map")
            os.makedirs(curr_output_dir, exist_ok=True)
            np.save(osp.join(curr_output_dir, "map2d.npy"), height_map)

            height_map2d_int = (height_map * 255).round().astype(np.uint8) # height map image
            img = PIL.Image.fromarray(height_map2d_int)
            # img.show()
            img.save(osp.join(curr_output_dir, "map2d.png"))

            # generate target climb position
            tar_root_height = height_map[128 // 2, 128 // 2]
            tar_climb_pos = [0.0, 0.0, tar_root_height - bbox_length[-1] / 2] # always place the sit point on the (0, 0)

            cfg["tarSitPos"] = tar_climb_pos # still use the name of tarSitPos to simplify the env code

            # generate config file
            with open(os.path.join(curr_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            # generate visualization
            components = []

            components.append(mesh)

            bbox = cfg["bbox"] # bbox
            line_x = trimesh.creation.box([bbox[0], 0.02, 0.02])
            line_y = trimesh.creation.box([0.02, bbox[1], 0.02])
            line_z = trimesh.creation.box([0.02, 0.02, bbox[2]])

            line_x.visual.vertex_colors[:, :3] = [0, 255, 0]
            line_y.visual.vertex_colors[:, :3] = [0, 255, 0]
            line_z.visual.vertex_colors[:, :3] = [0, 255, 0]

            components.append(line_x.copy().apply_translation([0,      bbox[1]/2, -1 * bbox[2]/2]))
            components.append(line_x.copy().apply_translation([0,      bbox[1]/2,      bbox[2]/2]))
            components.append(line_x.copy().apply_translation([0, -1 * bbox[1]/2,      bbox[2]/2]))
            components.append(line_x.copy().apply_translation([0, -1 * bbox[1]/2, -1 * bbox[2]/2]))
            components.append(line_y.copy().apply_translation([     bbox[0]/2, 0, -1 * bbox[2]/2]))
            components.append(line_y.copy().apply_translation([-1 * bbox[0]/2, 0, -1 * bbox[2]/2]))
            components.append(line_y.copy().apply_translation([-1 * bbox[0]/2, 0,      bbox[2]/2]))
            components.append(line_y.copy().apply_translation([     bbox[0]/2, 0,      bbox[2]/2]))
            components.append(line_z.copy().apply_translation([     bbox[0]/2,      bbox[1]/2, 0]))
            components.append(line_z.copy().apply_translation([     bbox[0]/2, -1 * bbox[1]/2, 0]))
            components.append(line_z.copy().apply_translation([-1 * bbox[0]/2, -1 * bbox[1]/2, 0]))
            components.append(line_z.copy().apply_translation([-1 * bbox[0]/2,      bbox[1]/2, 0]))

            # ground plane
            ground = trimesh.creation.box([2, 2, 0.001])
            ground.apply_translation([0, 0, -1 * bbox[2] / 2])
            components.append(ground)

            merged_mesh = trimesh.util.concatenate(components)

            # save
            save_dir = osp.join(obj_dir, categeory, id, "vis")
            os.makedirs(save_dir, exist_ok=True)
            merged_mesh.export(osp.join(save_dir, "mesh.obj"))

            print("Saving processed object mesh at {}".format(save_dir))
