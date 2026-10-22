"""Native original CLIMB test mixture; no changes to SIT or training."""
import json
from collections import Counter
from pathlib import Path
import torch
from isaacgym import gymapi
from utils.climb_mixed_sampling import CATEGORIES,model_weights,sample_ids

def assign(task):
    if not task.cfg['args'].test or task._task_graph_preset!='climb' or task.num_agents!=1 or task.num_objects!=1:
        raise ValueError('Mixed object probe requires single-agent CLIMB evaluation')
    root=Path(__file__).resolve().parents[2]/'climb_mixed_assets'
    catalog=json.loads((root/'catalog.json').read_text())
    ids=sample_ids(catalog,task.num_envs,task.device)
    dtype=task._box_size.dtype
    task._box_size[:]=torch.tensor([e['bbox'] for e in catalog],dtype=dtype,device=task.device)[ids]
    task._native_table_ids=ids
    task._native_table_catalog=catalog
    task._native_table_root=root
    task._native_table_cache={}
    task._native_table_meta_z=torch.tensor([e['tarSitPos'][2] for e in catalog],dtype=dtype,device=task.device)[ids,None]
    # The task is fixed to CLIMB. Do not exclude native objects by training-size eligibility.
    task._rsi_asset_sizes=torch.full((task.num_envs,1,3),.45,dtype=dtype,device=task.device)
    selected=ids.detach().cpu().tolist()
    task._climb_sampling=dict(method='torch.multinomial_replacement',categories=list(CATEGORIES),category_probabilities={c:1/3 for c in CATEGORIES},models_per_category=dict(Counter(e['category'] for e in catalog)),model_probabilities=model_weights(catalog).tolist(),category_counts=dict(Counter(catalog[i]['category'] for i in selected)),model_ids=selected)
    print('[mixed CLIMB]',task._climb_sampling['category_counts'],flush=True)

def load(task,index,options):
    mid=int(task._native_table_ids[index])
    if mid not in task._native_table_cache:
        entry=task._native_table_catalog[mid]
        folder=Path(entry['path'])
        options.fix_base_link=True
        options.use_mesh_materials=True
        options.mesh_normal_mode=gymapi.COMPUTE_PER_VERTEX
        options.vhacd_enabled=True
        options.vhacd_params=gymapi.VhacdParams()
        options.vhacd_params.resolution=100000
        options.vhacd_params.max_convex_hulls=128
        options.vhacd_params.max_num_vertices_per_ch=64
        options.replace_cylinder_with_capsule=False
        asset=task.gym.load_asset(task.sim,str(folder),'asset.urdf',options)
        if asset is None or task.gym.get_asset_rigid_body_count(asset)!=1 or task.gym.get_asset_dof_count(asset)!=0:
            raise ValueError('Expected native single-rigid-body asset: '+entry['id'])
        task._native_table_cache[mid]=asset
    return task._native_table_cache[mid]
