import sys
sys.path.append("./")

import os
import os.path as osp
import numpy as np
import trimesh
import yaml
import json
import PIL.Image

import torch
import torchgeometry as tgm

def get_heightmap_raytrace(mesh, HEIGHT_MAP_DIM: int=100):
    min_x, min_y, min_z = mesh.vertices[:, 0].min(), mesh.vertices[:, 1].min(), mesh.vertices[:, 2].min()
    max_x, max_y, max_z = mesh.vertices[:, 0].max(), mesh.vertices[:, 1].max(), mesh.vertices[:, 2].max()

    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    center_z = (max_z + min_z) / 2
    assert (center_x < 1e-3) and (center_y < 1e-3) and (center_z < 1e-3)

    mesh.apply_translation([0.0, 0.0, -1 * min_z]) # place the object on the ground, we assume that the up axis is Z

    x = np.linspace(min_x, max_x, HEIGHT_MAP_DIM)
    y = np.linspace(max_y, min_y, HEIGHT_MAP_DIM) # be careful with this!!! to align coordinate
    xx, yy = np.meshgrid(x, y)
    pos2d = np.concatenate([xx[..., None], yy[..., None]], axis=-1)

    origins = np.zeros((HEIGHT_MAP_DIM * HEIGHT_MAP_DIM, 3))
    vectors = np.zeros((HEIGHT_MAP_DIM * HEIGHT_MAP_DIM, 3))

    origins[:, 0] = pos2d.reshape(-1, 2)[:, 0]
    origins[:, 1] = pos2d.reshape(-1, 2)[:, 1]
    origins[:, 2] = source_height = mesh.vertices[:, 2].max() + 1.0

    vectors[:, 2] = -1

    # do the actual ray- mesh queries
    points, index_ray, index_tri = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=False
    )

    # for each hit, find the distance along its vector
    depth = trimesh.util.diagonal_dot(points - origins[index_ray], vectors[index_ray])
    # convert depth to height
    height = source_height - depth

    height_map2d = np.zeros((HEIGHT_MAP_DIM, HEIGHT_MAP_DIM))
    for idx, h in zip(index_ray, height):
        row = idx // HEIGHT_MAP_DIM
        col = idx - row * HEIGHT_MAP_DIM
        height_map2d[row, col] = h

    return height_map2d

if __name__ == "__main__":

    motion_file = osp.join(osp.dirname(__file__), "dataset_sit.yaml")
    with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
        motion_config = yaml.load(f, Loader=yaml.SafeLoader)

        candidates = []        
        for data in motion_config["motions"]["sit"]:
            candidates.append(data['file'])
        
    for seq_name in candidates:

        if "chair_mo" in seq_name:
            object_center = np.array([0.03428713232278818, 0.9752678275108337 + 0.2, 0.0]) # copied from InterScene
        
        elif "armchair" in seq_name:
            if ("armchair_stageII" in seq_name) or ("001" in seq_name) or ("002" in seq_name) or ("003" in seq_name) or ("004" in seq_name) or ("005" in seq_name) or ("006" in seq_name) or ("007" in seq_name):
                object_center = np.array([0.03428713232278818, 0.9752678275108337 - 0.2, 0.0]) # copied from InterScene
            elif ("008" in seq_name) or ("009" in seq_name):
                object_center = np.array([0.03428713232278818, 0.9752678275108337 - 0.35, 0.0]) # copied from InterScene
            else:
                object_center = np.array([0.03428713232278818, 0.9752678275108337 - 0.25, 0.0]) # copied from InterScene
        else:
            raise NotImplementedError

        aa = torch.tensor([0, 0, 3.14])
        quat = tgm.angle_axis_to_quaternion(aa).numpy() # angle axis ---> quaternion
        quat = quat[[1, 2, 3, 0]] # switch quaternion order wxyz -> xyzw

        object_state = np.concatenate([object_center, quat], axis=-1)
        save_path = osp.join(osp.dirname(__file__), osp.dirname(seq_name), "object_state.npy")
        np.save(save_path, object_state)
    
    obj_dir = osp.join(osp.dirname(__file__), "objects")
    obj_sets = os.listdir(obj_dir) # train or test

    for set in obj_sets:

        obj_categories = os.listdir(osp.join(obj_dir, set))

        # generate object config

        for categeory in obj_categories:

            obj_ids = os.listdir(osp.join(obj_dir, set, categeory))
            for id in obj_ids:
                curr_dir = osp.join(obj_dir, set, categeory, id)
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

                # generate target sit position
                offset = 0.1
                sit_height = (height_map[128 // 2, 128 // 2] + height_map[128 // 2 - 5, 128 // 2] + height_map[128 // 2 + 5, 128 // 2] + height_map[128 // 2 - 10, 128 // 2] + height_map[128 // 2 + 10, 128 // 2]) / 5.0 + offset
                tar_sit_pos = [0.0, 0.0, sit_height - bbox_length[-1] / 2] # always place the sit point on the (0, 0)

                cfg["tarSitPos"] = tar_sit_pos

                # generate config file
                with open(os.path.join(curr_dir, "config.json"), "w") as f:
                    json.dump(cfg, f)

                # generate visualization
                components = []

                marker = trimesh.creation.icosphere(subdivisions=3, radius=0.05) # tarSitPos
                marker.apply_translation(cfg["tarSitPos"])
                marker.visual.vertex_colors[:, :3] = [255, 0, 0]

                components.append(mesh)
                components.append(marker)

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
                save_dir = osp.join(obj_dir, set, categeory, id, "vis")
                os.makedirs(save_dir, exist_ok=True)
                merged_mesh.export(osp.join(save_dir, "mesh.obj"))

                print("Saving processed object mesh at {}".format(save_dir))
