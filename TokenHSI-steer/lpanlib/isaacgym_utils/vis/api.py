import os
import os.path as osp
import trimesh

from .utils.sp_animation import sp_animation
from .utils.xml_parser import parse_geom_elements_from_xml
from .utils.body_builder import build_complete_body, state2mat

def vis_motion_use_scenepic_animation(
        asset_filename,
        rigidbody_global_pos, # (N_frames, N_bodies, 3)
        rigidbody_global_rot, # (N_frames, N_bodies, 4)
        fps,
        up_axis,
        color,
        output_path,
    ):

    # create scenepic animator
    animator = sp_animation(framerate=fps)

    # add ground plane
    plane = trimesh.load(osp.join(osp.dirname(osp.abspath(__file__)), f"data/plane_{up_axis.lower()}_up.obj"), process=False)
    animator.add_static_mesh(plane, 'plane')

    # add human meshes for the motion sequence
    rigidbody_names, rigidbody_meshes = parse_geom_elements_from_xml(asset_filename)
    
    num_frames = rigidbody_global_pos.shape[0]
    for i in range(num_frames):
        human_mesh = build_complete_body(rigidbody_global_pos[i], rigidbody_global_rot[i], rigidbody_meshes)
        human_mesh.visual.vertex_colors[:, :3] = color
        animator.add_frame([human_mesh], ['human'])

    # save
    animator.save_animation(output_path)

def vis_hoi_use_scenepic_animation(
        asset_filename,
        rigidbody_global_pos, # (N_frames, N_bodies, 3)
        rigidbody_global_rot, # (N_frames, N_bodies, 4)
        fps,
        up_axis,
        color,
        output_path,
        obj_meshes,
        obj_global_pos, # (N_frames, N_objs, 3)
        obj_global_rot, # (N_frames, N_objs, 4)
        obj_colors,
    ):

    # create scenepic animator
    animator = sp_animation(framerate=fps)

    # add ground plane
    plane = trimesh.load(osp.join(osp.dirname(osp.abspath(__file__)), f"data/plane_{up_axis.lower()}_up.obj"), process=False)
    animator.add_static_mesh(plane, 'plane')

    # add human meshes for the motion sequence
    rigidbody_names, rigidbody_meshes = parse_geom_elements_from_xml(asset_filename)
    
    num_frames = rigidbody_global_pos.shape[0]
    for i in range(num_frames):

        sp_meshes = []
        sp_layers = []

        human_mesh = build_complete_body(rigidbody_global_pos[i], rigidbody_global_rot[i], rigidbody_meshes)
        human_mesh.visual.vertex_colors[:, :3] = color
        sp_meshes.append(human_mesh)
        sp_layers.append('human')

        num_objs = len(obj_meshes)
        for j in range(num_objs):
            obj_mesh = obj_meshes[j].copy()
            matrix = state2mat(obj_global_pos[i, j], obj_global_rot[i, j])
            obj_mesh.apply_transform(matrix)
            obj_mesh.visual.vertex_colors[:, :3] = obj_colors[j]
            sp_meshes.append(obj_mesh),
            sp_layers.append('obj_{:02d}'.format(j))
        
        animator.add_frame(sp_meshes, sp_layers)

    # save
    animator.save_animation(output_path)

def vis_hoi_use_scenepic_animation_climb(
        asset_filename,
        rigidbody_global_pos, # (N_frames, N_bodies, 3)
        rigidbody_global_rot, # (N_frames, N_bodies, 4)
        fps,
        up_axis,
        color,
        output_path,
        obj_meshes,
        obj_global_pos, # (N_objs, 3)
        obj_global_rot, # (N_objs, 4)
        obj_colors,
        obj_names,
    ):

    # create scenepic animator
    animator = sp_animation(framerate=fps)

    # add ground plane
    plane = trimesh.load(osp.join(osp.dirname(osp.abspath(__file__)), f"data/plane_{up_axis.lower()}_up.obj"), process=False)
    animator.add_static_mesh(plane, 'plane')

    num_objs = len(obj_meshes)
    for j in range(num_objs):
        obj_mesh = obj_meshes[j].copy()
        obj_mesh.visual.vertex_colors[:, :3] = obj_colors[j]
        matrix = state2mat(obj_global_pos[j], obj_global_rot[j])
        obj_mesh.apply_transform(matrix)
        animator.add_static_mesh(obj_mesh, obj_names[j])

    # add human meshes for the motion sequence
    rigidbody_names, rigidbody_meshes = parse_geom_elements_from_xml(asset_filename)
    
    num_frames = rigidbody_global_pos.shape[0]
    for i in range(num_frames):

        sp_meshes = []
        sp_layers = []

        human_mesh = build_complete_body(rigidbody_global_pos[i], rigidbody_global_rot[i], rigidbody_meshes)
        human_mesh.visual.vertex_colors[:, :3] = color
        sp_meshes.append(human_mesh)
        sp_layers.append('human')
        
        animator.add_frame(sp_meshes, sp_layers)

    # save
    animator.save_animation(output_path)
