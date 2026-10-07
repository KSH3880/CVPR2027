"""Shared reference pose pools and geometric object bindings."""
import math
import torch
from utils.torch_utils import quat_mul,quat_rotate,exp_map_to_quat

def asset_reference_pose(task, root, rotation, dof):
    """Use the simulated skeleton lengths/joint axes, not motion-file offsets."""
    if not hasattr(task,'_rsi_asset_skeleton'):
        import xml.etree.ElementTree as ET
        from pathlib import Path
        asset=task.cfg['env']['asset']
        tree=ET.parse(str(Path(asset['assetRoot'])/asset['assetFileName']))
        nodes=[];parents=[]
        def walk(node,parent):
            index=len(nodes);nodes.append(node);parents.append(parent)
            for child in node.findall('body'):walk(child,index)
        walk(tree.find('worldbody/body'),-1)
        if len(nodes)!=task.num_bodies:raise ValueError('RSI asset body order mismatch')
        offsets=root.new_tensor([[float(v) for v in node.get('pos','0 0 0').split()] for node in nodes])
        axes={i:root.new_tensor([float(v) for v in node.find('joint').get('axis','0 0 1').split()]) for i,node in enumerate(nodes) if len(node.findall('joint'))==1}
        task._rsi_asset_skeleton=(parents,offsets,axes)
    parents,offsets,axes=task._rsi_asset_skeleton
    local=torch.zeros(len(root),task.num_bodies,4,device=root.device);local[...,3]=1
    for j,index in enumerate(task._dof_body_ids):
        start,end=task._dof_offsets[j:j+2]
        if end-start==3:local[:,index]=exp_map_to_quat(dof[:,start:end])
        else:
            angle=dof[:,start]/2
            local[:,index,:3]=axes[index]*angle.sin()[:,None]
            local[:,index,3]=angle.cos()
    positions=[root];rotations=[rotation]
    for i in range(1,task.num_bodies):
        parent=parents[i]
        positions.append(positions[parent]+quat_rotate(rotations[parent],offsets[i].expand(len(root),3)))
        rotations.append(quat_mul(rotations[parent],local[:,i]))
    return torch.stack(positions,1),torch.stack(rotations,1)


def push_hand_normal(relative_xy):
    """Face normal perpendicular to the hand pair, so both palms reach the face."""
    midpoint=relative_xy.mean(-2)
    separation=relative_xy[...,1,:]-relative_xy[...,0,:]
    normal=torch.stack([-separation[...,1],separation[...,0]],-1)
    normal=torch.where(((normal*midpoint).sum(-1)<0)[...,None],-normal,normal)
    normal=torch.where((separation.norm(dim=-1)<1e-5)[...,None],midpoint,normal)
    return normal/normal.norm(dim=-1,keepdim=True).clamp_min(1e-6)


def frame_clearance(task, points):
    """Conservative body/limb envelopes against the three fixed frame boxes.

    Points are body origins in door-base coordinates. Foot spheres include toes;
    torso/head spheres include their offset geometry. Segment samples include a
    half-spacing margin so joints alone cannot miss a crossing limb.
    """
    spec=task._door_spec
    radius=points.new_tensor([.12,.25,.28,.07,.06,.05,.07,.06,.05,.07,.065,.23,.07,.065,.23])
    parent=[0,0,1,1,3,4,1,6,7,0,9,10,0,12,13]
    fraction=points.new_tensor([.25,.5,.75])
    start=points[...,parent,:]
    segments=start.unsqueeze(-2)+(points-start).unsqueeze(-2)*fraction[None,:,None]
    spacing=(points-start).norm(dim=-1)/8
    segment_radius=radius.clone();segment_radius[[1,2,3,6,9,12]]=.075
    radii=torch.cat([radius.expand(points.shape[:-1]),(segment_radius+spacing).unsqueeze(-1).expand(*spacing.shape,3).reshape(*spacing.shape[:-1],-1)],-1)
    samples=torch.cat([points,segments.reshape(*points.shape[:-2],-1,3)],-2)
    height=spec.height+spec.bottom_gap+spec.frame_gap
    side=spec.width/2+spec.frame_gap+spec.frame_width/2
    centers=points.new_tensor([[0,-side,height/2],[0,side,height/2],[0,0,height+spec.frame_width/2]])
    halves=points.new_tensor([[spec.frame_depth/2,spec.frame_width/2,height/2],[spec.frame_depth/2,spec.frame_width/2,height/2],[spec.frame_depth/2,side+spec.frame_width/2,spec.frame_width/2]])
    delta=(samples.unsqueeze(-2)-centers).abs()-halves
    sdf=delta.clamp_min(0).norm(dim=-1)+delta.amax(-1).clamp_max(0)
    return (sdf-radii.unsqueeze(-1)).amin(dim=(-1,-2))

def safe_door_candidates(task, root, body):
    """Return aligned transforms and safe angle masks before sampling a reset."""
    cfg=task._interaction['task_rsi'];dev=body.device
    rel=body-root[:,None,:]
    hands=rel[:,task._key_body_ids[:2]]
    heights=body[:,task._key_body_ids[:2],2]
    eligible=hands[...,:2].norm(dim=-1)>cfg['min_extension']
    pick=(heights-task._door_spec.handle_height).abs().masked_fill(~eligible,float('inf')).argmin(-1)
    hand=hands[torch.arange(len(root),device=dev),pick]
    low,high=cfg['door_angle_degrees']
    angles=torch.deg2rad(torch.linspace(low,high,64,device=dev))
    yaw=angles[None,:]-torch.atan2(hand[:,1],hand[:,0])[:,None]
    q=torch.zeros(len(root),len(angles),4,device=dev);q[...,2]=torch.sin(yaw/2);q[...,3]=torch.cos(yaw/2)
    rotated=quat_rotate(q[:,:,None,:].expand(-1,-1,body.shape[1],-1).reshape(-1,4),rel[:,None,:,:].expand(-1,len(angles),-1,-1).reshape(-1,3)).view(len(root),len(angles),body.shape[1],3)
    hand_rotated=rotated[torch.arange(len(root),device=dev)[:,None],torch.arange(len(angles),device=dev)[None,:],task._key_body_ids[:2][pick][:,None]]
    aq=torch.zeros(len(angles),4,device=dev);aq[:,2]=torch.sin(angles/2);aq[:,3]=torch.cos(angles/2)
    handle=body.new_tensor(task._door_spec.hinge)+quat_rotate(aq,body.new_tensor(task._door_spec.front_handle_local).expand(len(angles),3))
    origins=root[:,None,:].expand(-1,len(angles),-1).clone()
    normal=torch.stack([angles.cos(),angles.sin()],-1)
    origins[...,:2]=handle[None,:,:2]-normal[None,:,:]*cfg['hand_gap']-hand_rotated[...,:2]
    clearance=frame_clearance(task,rotated+origins[:,:,None,:])
    return angles,q,origins,clearance>=cfg.get('frame_clearance',.03)

def build_pool(task,skill):
    cfg=task._interaction['task_rsi'];lib=task._motion_lib[skill]
    lo,hi=cfg['push_phase' if skill=='push' else 'door_phase']
    mids=[];times=[]
    for i,length in enumerate(lib._motion_lengths):
        if lib._motion_weights[i] <= 0:continue
        t=torch.arange(float(length)*lo,float(length)*hi,float(lib._motion_dt[i]),device=task.device)
        if len(t):mids.append(torch.full_like(t,i,dtype=torch.long));times.append(t)
    mids=torch.cat(mids);times=torch.cat(times)
    root,rot,dof,rv,ra,dv,key=lib.get_motion_state(mids,times)
    body,brot=asset_reference_pose(task,root,rot,dof)
    hands=body[:,task._key_body_ids[:2]]
    extension=torch.linalg.vector_norm(hands[...,:2]-root[:,None,:2],dim=-1)
    if skill=='push':
        low,high=cfg['push_hand_height']
        rel=hands[...,:2]-root[:,None,:2]
        normal=push_hand_normal(rel)
        tangent=torch.stack([-normal[:,1],normal[:,0]],-1)
        forward=(rel*normal[:,None,:]).sum(-1)
        lateral=((rel-rel.mean(1,keepdim=True))*tangent[:,None,:]).sum(-1).abs()
        valid=((hands[...,2]>=low)&(hands[...,2]<=high)).all(-1)&(forward.max(-1).values>.3)&(lateral.max(-1).values<task._box_size[1]/2-.01)
    else:
        near=(hands[...,2]-task._door_spec.handle_height).abs()<cfg['door_height_tolerance']
        valid=(near&(extension>cfg['min_extension'])).any(-1)
        indices=valid.nonzero().flatten()
        for chunk in indices.split(128):
            valid[chunk]&=safe_door_candidates(task,root[chunk],body[chunk])[3].any(-1)
    early_safe=torch.zeros_like(valid)
    if skill=='doorOpen':
        for chunk in valid.nonzero().flatten().split(128):
            angles,_,_,safe=safe_door_candidates(task,root[chunk],body[chunk])
            early_safe[chunk]=(safe & (angles<=math.radians(cfg['door_early_max_degrees']))).any(-1)
        if not early_safe.any():raise ValueError('No frame-safe small-angle door RSI poses')
    if not valid.any():raise ValueError('No physically bindable RSI reference poses for '+skill)
    mids=mids[valid];times=times[valid]
    count=torch.bincount(mids,minlength=len(lib._motion_lengths)).clamp_min(1)
    weights=lib._motion_weights[mids]/count[mids]
    phase=times/lib._motion_lengths[mids]
    late=phase>lo+.5*(hi-lo)
    # Explicitly allocate 70% probability to the later half when both halves exist.
    if late.any() and (~late).any():
        weights=torch.where(late,weights/weights[late].sum()*cfg['late_fraction'],weights/weights[~late].sum()*(1-cfg['late_fraction']))
    weights/=weights.sum()
    print('[task RSI pool]',skill,'eligible_frames',len(times),'late_probability',float(weights[late].sum()),flush=True)
    return mids,times,weights,early_safe[valid]

def sample_reference(task,tasks):
    cfg=task._interaction['task_rsi'];n=tasks.numel();dev=task.device
    selected=torch.rand(n,device=dev)<cfg['probability']
    loco=task._motion_lib['loco'];ids=loco.sample_motions(n)
    t=torch.full((n,),task.dt*(task._num_amp_obs_steps-1),device=dev)
    root,rot,dof,*_=loco.get_motion_state(ids,t)
    body,brot,_,_=loco.get_motion_state_max(ids,t)
    if not hasattr(task,'_task_rsi_pools'):
        task._task_rsi_pools={s:build_pool(task,s) for s in ('push','doorOpen')}
    phases=torch.zeros(n,device=dev)
    early=torch.zeros(n,device=dev,dtype=torch.bool)
    for uid,skill in enumerate(('push','doorOpen')):
        mask=selected&(tasks.flatten()==uid);size=int(mask.sum())
        if not size:continue
        lib=task._motion_lib[skill];mid,tm,weights,early_safe=task._task_rsi_pools[skill]
        indices=torch.multinomial(weights,size,replacement=True)
        if skill=='doorOpen':
            want_early=torch.rand(size,device=dev)<cfg['door_early_fraction']
            # Sample the pose from the small-angle-compatible pool FIRST.
            # Angle rejection must not bias the requested mixture toward wide-open starts.
            early_weights=weights*early_safe
            phase=tm/lib._motion_lengths[mid]
            lo,hi=cfg['door_phase'];late=phase>lo+.5*(hi-lo)
            if early_weights[late].sum()>0 and early_weights[~late].sum()>0:
                early_weights=torch.where(late,early_weights/early_weights[late].sum()*cfg['late_fraction'],early_weights/early_weights[~late].sum()*(1-cfg['late_fraction']))
            indices[want_early]=torch.multinomial(early_weights,int(want_early.sum()),replacement=True) if want_early.any() else indices[want_early]
            early[mask]=want_early
        mid=mid[indices];tm=tm[indices]
        r,q,d,*_=lib.get_motion_state(mid,tm)
        b,bq=asset_reference_pose(task,r,q,d)
        root[mask]=r;rot[mask]=q;dof[mask]=d;body[mask]=b;brot[mask]=bq
        phases[mask]=tm/lib._motion_lengths[mid]
    task._last_rsi_door_early=early.view_as(tasks)
    task._last_rsi_selected=selected.view_as(tasks)
    task._last_rsi_phase=phases.view_as(tasks)
    return root,rot,dof,body,brot,selected

def align_task_reference(task,env_ids,tasks,selected,refroot,body,unheading,newroot):
    cfg=task._interaction['task_rsi'];E,M=tasks.shape;flat_tasks=tasks.flatten()
    relhands=body[:,task._key_body_ids[:2]]-refroot[:,None,:]
    rot=unheading[:,None,:].expand(-1,2,-1)
    hands=quat_rotate(rot.reshape(-1,4),relhands.reshape(-1,3)).view(-1,2,3)
    output=newroot.reshape(-1,3).clone()
    rotation=unheading.clone()
    push=selected&(flat_tasks==0);door=selected&(flat_tasks==1)
    boxes=task._box_states[env_ids].reshape(-1,13).clone()
    goals=task._targets[env_ids].reshape(-1,3).clone()
    if push.any():
        # Align BOTH hand centers to the rear face rather than only the nearer hand.
        h=push_hand_normal(hands[push,...,:2])
        boxyaw=2*torch.atan2(boxes[push,5],boxes[push,6])
        yaw=boxyaw-torch.atan2(h[:,1],h[:,0])
        q=torch.zeros(len(yaw),4,device=task.device);q[:,2]=torch.sin(yaw/2);q[:,3]=torch.cos(yaw/2)
        rotation[push]=quat_mul(q,unheading[push])
        rh=quat_rotate(q[:,None,:].expand(-1,2,-1).reshape(-1,4),hands[push].reshape(-1,3)).view(-1,2,3)
        normal=torch.stack([torch.cos(boxyaw),torch.sin(boxyaw)],-1)
        tangent=torch.stack([-normal[:,1],normal[:,0]],-1)
        reach=(rh[...,:2]*normal[:,None,:]).sum(-1).max(-1).values
        sideways=(rh[...,:2]*tangent[:,None,:]).sum(-1).mean(-1)
        output[push,:2]=boxes[push,:2]-normal*(reach+task._box_size[0]/2+cfg['push_hand_radius']+cfg['push_surface_gap'])[:,None]-tangent*sideways[:,None]
        low,high=cfg['push_remaining'];distance=low+torch.rand(len(yaw),device=task.device)*(high-low)
        goal_yaw=boxyaw+torch.deg2rad((torch.rand(len(yaw),device=task.device)*2-1)*20)
        goals[push,:2]=boxes[push,:2]+distance[:,None]*torch.stack([torch.cos(goal_yaw),torch.sin(goal_yaw)],-1)
    if door.any():
        angles,qs,origins,safe=safe_door_candidates(task,refroot[door],body[door])
        if not safe.any(-1).all():raise RuntimeError('Door RSI pool contains a pose without frame clearance')
        requested_early=task._last_rsi_door_early.flatten()[door]
        small=angles<=math.radians(cfg['door_early_max_degrees'])
        safe=torch.where(requested_early[:,None],safe & small[None,:],safe)
        choice=torch.multinomial(safe.float(),1).squeeze(-1)
        row=torch.arange(len(choice),device=task.device)
        angle=angles[choice]
        rotation[door]=qs[row,choice]
        bases=task._root_states[task.doors.actor_indices.long()].view(task.num_envs,task.num_agents,13)[env_ids].reshape(-1,13)[door,:3]
        output[door,:2]=origins[row,choice,:2]+bases[:,:2]
        state=task._door_state_view[env_ids].clone().reshape(-1,2);state[door,0]=angle;state[door,1]=0
        task._door_state_view[env_ids]=state.view(E,M,2)
        task._best_angle[env_ids]=task._door_state_view[env_ids,:,0].clamp_min(0)
    task._targets[env_ids]=goals.view(E,M,3)
    task._best_push[env_ids]=-torch.linalg.vector_norm(task._box_states[env_ids,...,:2]-task._targets[env_ids,...,:2],dim=-1)
    return rotation,output.view(E,M,3)
