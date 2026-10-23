"""Select native RSI frames by hand midpoint height; never retarget any pose."""
from isaacgym import gymapi
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push

args=get_args();cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
assert cfg['env']['pushTask']['rsi_retarget']=='none'
set_seed(42);torch.set_num_threads(1)
task=HumanoidMAStage1Push(cfg,parse_sim_params(args,cfg,train),args.physics_engine,args.device,args.device_id,True)
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
try:
    source=Path(cfg['env']['pushTask']['rsi_cache_file'])
    with np.load(source,allow_pickle=False) as data:payload={key:data[key].copy() for key in data.files}
    lib=task._motion_lib['push'];geometry=task._push_reset_geometry();assert geometry is not None
    from utils.open_push_rsi import align_push_pose
    heights=[];placed=[]
    for clip in range(lib.num_motions()):
        for start in range(0,int(lib._motion_num_frames[clip]),512):
            frame=torch.arange(start,min(start+512,int(lib._motion_num_frames[clip])),device=task.device)
            clips=torch.full_like(frame,clip);times=frame*lib._motion_dt[clip]
            body,rotation,velocity,angular=lib.get_motion_state_max(clips,times)
            original=body[:,task._key_body_ids[:2],2].mean(-1)
            targets=body.new_tensor([[.2,-.9,1.],[-.2,-.9,1.]]).expand(len(body),-1,-1)
            body,rotation,_,_,_=align_push_pose(body,rotation,velocity,angular,targets,task._key_body_ids[:2].cpu().tolist(),cfg['env']['pushTask']['stand_off_m'])
            shift=geometry.floor_shift(body,rotation)
            heights.append(original.cpu());placed.append((original+shift).cpu())
    reference=torch.cat(heights).numpy();final=torch.cat(placed).numpy()
    # Small reserve absorbs float32 yaw/translation roundoff during actual placement.
    eligible=np.isfinite(reference)&np.isfinite(final)&(reference<=1.-1e-3)&(final<=1.-1e-3)
    original_valid=payload['valid'].copy();payload['valid'] &= eligible[None]
    assert payload['valid'].any(axis=1).all()
    meta=json.loads(str(payload['metadata']))
    meta['height_selection']=dict(kind='native_hand_midpoint_z',max_m=1.,include_safe_floor_shift=True)
    meta['height_filter_source_cache_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    meta['height_filter_numeric_reserve_m']=1e-3
    payload['metadata']=np.array(json.dumps(meta));payload['reference_hand_midpoint_z_m']=reference;payload['placed_hand_midpoint_z_m']=final
    target=out/'stage1_push_depth180_rsi_le1m_cache.npz';assert not target.exists()
    np.savez_compressed(target,**payload)
    offsets=payload['clip_offsets'];counts=[]
    for clip in range(lib.num_motions()):
        mask=payload['valid'][:,offsets[clip]:offsets[clip+1]]
        counts.append(dict(motion_id=clip,valid_size_frames=int(mask.sum()),profiles_with_frames=int(mask.any(axis=1).sum())))
    selected=payload['valid'].any(axis=0)
    report=dict(limit_m=1.,source_cache=str(source),source_cache_sha256=meta['height_filter_source_cache_sha256'],
        source_valid_size_frames=int(original_valid.sum()),selected_size_frames=int(payload['valid'].sum()),
        profiles=len(payload['sizes']),per_profile=payload['valid'].sum(axis=1).tolist(),per_motion=counts,
        selected_motion_count=sum(r['valid_size_frames']>0 for r in counts),
        selected_reference_height_m=[float(reference[selected].min()),float(reference[selected].max())],
        selected_placed_height_m=[float(final[selected].min()),float(final[selected].max())],
        pose_retargeted=False,dof_or_velocity_changed=False,source_cache_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==meta['height_filter_source_cache_sha256'],
        filtered_cache_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (out/'selection.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
finally:task.gym.destroy_sim(task.sim)
