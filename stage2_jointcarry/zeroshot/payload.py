"""Inference-only single-link payload swap; existing bbox policy interface retained."""
import os
from pathlib import Path


def load_payload(task, index, options):
    path = os.environ.get('JOINT_ZERO_SHOT_URDF')
    if not path or index % task.num_objects != 0:
        return None
    if task._mode != 'test':
        raise ValueError('Zero-shot payload replacement is inference-only')
    meta = native_metadata()
    if meta and meta.get('variants'):
        path = meta['variants'][(index // task.num_objects) % len(meta['variants'])]['urdf']
    from isaacgym import gymapi
    import xml.etree.ElementTree as ET
    path = Path(path).resolve(strict=True)
    root = ET.parse(str(path)).getroot()
    if len(root.findall('link')) != 1 or root.findall('joint'):
        raise ValueError('Payload must have exactly one rigid link and zero joints')
    if not root.find('link').findall('collision'):
        raise ValueError('Payload requires collision geometry')
    # Mesh sources live in the copied checkout; Isaac Gym uses its own VHACD cache.
    options.vhacd_enabled = True
    options.vhacd_params.max_convex_hulls = 16
    options.vhacd_params.max_num_vertices_per_ch = 64
    options.vhacd_params.resolution = 100000
    options.override_com = native_metadata() is None
    options.override_inertia = native_metadata() is None
    options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset = task.gym.load_asset(task.sim, str(path.parent), path.name, options)
    if asset is None:
        raise RuntimeError('Payload asset failed to load')
    if task.gym.get_asset_rigid_body_count(asset) != 1 or task.gym.get_asset_dof_count(asset) != 0:
        raise ValueError('Loaded payload violates the actor tensor layout')
    return asset


def native_metadata():
    import json
    path=os.environ.get('JOINT_ZERO_SHOT_NATIVE_CONFIG')
    return json.loads(Path(path).read_text()) if path else None


def apply_native_size(task):
    meta=native_metadata()
    if meta is None:
        return
    if task._task_graph_preset != 'joint_carry_at':
        raise ValueError('Native-size payloads currently support joint_carry_at only')
    import torch
    size=torch.tensor(meta['size'],device=task.device,dtype=task._box_size.dtype)
    if size.shape!=(3,) or not torch.isfinite(size).all() or (size<=0).any():
        raise ValueError('Invalid native bbox')
    if meta.get('variants'):
        sizes=torch.tensor([meta['variants'][e % len(meta['variants'])]['size'] for e in range(task.num_envs)],device=task.device,dtype=task._box_size.dtype)
        if not torch.isfinite(sizes).all() or (sizes<=0).any():
            raise ValueError('Invalid batch sizes')
        task._box_size.reshape(task.num_envs,task.num_objects,3)[:,0]=sizes
        import json
        mapping=[dict(env=e,name=meta['variants'][e % len(meta['variants'])]['name'],size=meta['variants'][e % len(meta['variants'])]['size']) for e in range(task.num_envs)]
        Path(task.cfg['args'].output_path,'native_env_assets.json').write_text(json.dumps(mapping,indent=2)+'\n')
    else:
        task._box_size.reshape(task.num_envs,task.num_objects,3)[:,0]=size
    task._box_scale=task._box_size/torch.tensor(task._build_base_size,device=task.device)
    print('NATIVE_PAYLOAD_BBOX', meta['name'], meta['size'], 'mass', meta['mass'], flush=True)



def verify_direct_gpu6_context(output):
    """Require a direct CUDA context on physical GPU 6 before constructing PhysX."""
    import json
    import subprocess
    import xml.etree.ElementTree as ET
    import torch
    if os.environ.get('CUDA_MPS_PIPE_DIRECTORY') != '':
        raise ValueError('GPU PhysX probe must explicitly bypass the training MPS server')
    expected=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected:
        raise ValueError('GPU probe must expose physical GPU 6 by UUID only')
    probe=torch.zeros(1,device='cuda:0');torch.cuda.synchronize()
    tree=ET.fromstring(subprocess.check_output(['nvidia-smi','-i','6','-q','-x'],text=True))
    entries=[node for node in tree.findall('.//process_info') if node.findtext('pid')==str(os.getpid())]
    if len(entries)!=1 or entries[0].findtext('type')!='C':
        raise RuntimeError('Expected a direct compute context (C), not an MPS client; PhysX was not started')
    report=dict(pid=os.getpid(),physical_gpu=6,uuid=expected,process_type='C',mps_pipe='',physics='gpu',pipeline='gpu',policy='cuda:0')
    Path(output,'gpu_context.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DIRECT_GPU6_CONTEXT',json.dumps(report),flush=True)
