"""Separate native object categories for SIT/CLIMB, evaluation only."""
import json,os
from pathlib import Path
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate

def assign(task):
    if not task.cfg['args'].test or task._task_graph_preset not in ('sit','climb') or task.num_agents!=1 or task.num_objects!=1:
        raise ValueError('Object comparison requires single-agent SIT/CLIMB evaluation')
    root=Path(__file__).resolve().parents[2]/'comparison_assets'
    category=os.environ['OBJECT_CATEGORY']
    catalog=json.loads((root/'catalog.json').read_text())[category]
    ids=torch.multinomial(torch.full((len(catalog),),1/len(catalog),device=task.device),task.num_envs,replacement=True)
    dtype=task._box_size.dtype
    def values(key):return torch.tensor([e[key] for e in catalog],device=task.device,dtype=dtype)[ids,None]
    task._box_size[:]=values('bbox').reshape(-1,3)
    task._native_category=category
    task._native_chair_ids=ids;task._native_chair_catalog=catalog;task._native_chair_cache={}
    task._native_chair_seat=values('sit_surface' if task._task_graph_preset=='sit' else 'climb_surface')
    task._native_chair_extent=values('support_extent')
    task._rsi_asset_sizes=torch.full((task.num_envs,1,3),.45,dtype=dtype,device=task.device)
    task._object_sampling=dict(category=category,method='uniform_replacement',model_count=len(catalog),model_counts=torch.bincount(ids,minlength=len(catalog)).cpu().tolist())
    print('[object comparison]',category,task._task_graph_preset,task._object_sampling,flush=True)

def load(task,index,options):
    mid=int(task._native_chair_ids[index])
    if mid not in task._native_chair_cache:
        folder=Path(task._native_chair_catalog[mid]['path'])
        options.fix_base_link=True;options.use_mesh_materials=True
        options.mesh_normal_mode=gymapi.COMPUTE_PER_VERTEX;options.vhacd_enabled=True
        options.vhacd_params=gymapi.VhacdParams();options.vhacd_params.resolution=100000
        options.vhacd_params.max_convex_hulls=128;options.vhacd_params.max_num_vertices_per_ch=64
        asset=task.gym.load_asset(task.sim,str(folder),'asset.urdf',options)
        if asset is None or task.gym.get_asset_rigid_body_count(asset)!=1 or task.gym.get_asset_dof_count(asset)!=0:raise ValueError('Invalid rigid native object')
        task._native_chair_cache[mid]=asset
    return task._native_chair_cache[mid]
load_chair=load

def seat_support(task,objects,sizes,env_ids):
    objects=objects.clone()
    seats=task._logical_box_values(task._native_chair_seat,env_ids)
    extents=task._logical_box_values(task._native_chair_extent,env_ids)
    objects[...,:3]+=quat_rotate(objects[...,3:7].reshape(-1,4),seats.reshape(-1,3)).reshape_as(seats)
    objects[...,2]-=.01
    return objects,torch.cat([extents,torch.full_like(extents[...,:1],.02)],-1)
