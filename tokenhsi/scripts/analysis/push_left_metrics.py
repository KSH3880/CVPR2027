"""Force-driven wrong-direction left pushing before the far partner arrives."""
import numpy as np
from push_comparison_select import first_run


def left_push_metrics(trace,index,initial,env,a,sideways_case=False):
    other=1-a
    position=trace['boxes'][:,index,:2].numpy()
    root=trace['roots'][:,index,:,:2].numpy()
    alive=trace['alive'][:,index].all(-1).numpy()
    distances=trace['surface_distance'][:,index].numpy()
    b_gap=distances[:,other].min(-1)
    contact=(distances[:,a].max(-1)<=.18)&alive
    cfg=initial['config']['env']['jointPush']['contact']
    if 'hand_force_xy_norm' in trace:
        force=(trace['hand_force_xy_norm'][:,index,a].numpy().min(-1)>=cfg['min_hand_force'])
        force &= trace['box_contact_xy_norm'][:,index].numpy()>=cfg['min_box_force']
    else:force=trace['force_contact'][:,index,a].numpy()
    active=contact&force
    first=first_run(contact);b_arrival=first_run((b_gap<=.3)&alive)
    result=dict(left_push_pass=False,left_push_before_b_m=0.,left_push_capture_step=-1,
        physical_a_contact_step=first,physical_b_near_step=b_arrival,rejection=[])
    if first<0:
        result['rejection'].append('A never establishes physical hand contact');return result
    end=b_arrival if b_arrival>=0 else len(position)
    frames=np.arange(len(position))
    first_force=first_run(active&(frames<end))
    if first_force<0 or first_force>first+15:
        result['rejection'].append('No prompt sustained force-driven A push')
    direction=initial['center_goals'][env,0,:2].numpy()-position[0]
    direction/=np.linalg.norm(direction).clip(1e-8)
    left=np.array([-direction[1],direction[0]])
    delta=position-position[first]
    sideways=delta@left;forward=delta@direction
    angle=np.degrees(np.arctan2(sideways,forward))
    speed=np.zeros(len(position));speed[3:]=(position[3:]-position[:-3])@left
    distance=np.linalg.norm(root[:,other]-position[first],axis=-1)
    progress=distance[first]-distance
    step=np.zeros_like(root[:,other]);step[3:]=root[3:,other]-root[:-3,other]
    toward=position-root[:,other];toward/=np.linalg.norm(toward,axis=-1,keepdims=True).clip(1e-8)
    gap_threshold=.7
    if first_force>=0 and min(float(b_gap[first]),float(b_gap[first_force]))<.7:gap_threshold=.5
    if first_force>=0 and min(float(b_gap[first]),float(b_gap[first_force]))<gap_threshold:
        result['rejection'].append('B is already next to the box when A pushing starts')
    startup_end=min(first+15,max(first,end-1))
    startup=float(np.linalg.norm(delta[first:startup_end+1],axis=-1).max())
    if startup<.1:result['rejection'].append('No immediate box motion after A contact')
    candidate=active&(b_gap>=.5)&(progress>=.15)&((step*toward).sum(-1)>=.015)
    if sideways_case:
        candidate &= (frames>=first+6)&(frames<=end-6)
        usable=np.flatnonzero(candidate)
        early_frame=int(usable[np.argmax(np.linalg.norm(delta[usable],axis=-1))]) if len(usable) else -1
        if early_frame<0:result['rejection'].append('No verified A pushing while B is far and approaching')
        else:
            count=int(active[first:early_frame+1].sum())
            result.update(clear_partner_early_motion_m=float(np.linalg.norm(delta[early_frame])),
                early_capture_step=early_frame,first_force_push_step=first_force,
                solo_force_contact_frames=count,solo_force_contact_fraction=count/(early_frame-first+1),
                early_partner_min_hand_gap_m=float(b_gap[early_frame]))
            if count<6 or count/(early_frame-first+1)<.25:result['rejection'].append('Insufficient sustained solo pushing force')
        result['premature_push_verified']=not result['rejection']
        possible=np.flatnonzero(alive&(frames>=first))
        signed=(position-position[0])@left
        peak=int(possible[np.argmax(np.abs(signed[possible]))])
        offset=float(abs(signed[peak]))
        result.update(sideways_path_offset_m=offset,sideways_path_signed_m=float(signed[peak]),
            sideways_path_side='left' if signed[peak]>=0 else 'right',sideways_peak_step=peak,
            startup_box_motion_m=startup,minimum_partner_gap_m=gap_threshold)
        if offset<2.:result['rejection'].append('Box position never deviates >=2m to either side of the initial goal line')
        result['sideways_case_pass']=not result['rejection']
        return result
    candidate &= (frames>=first+6)&(frames<=end-6)&(sideways>0)&(angle>=60)&(speed>=.009)
    usable=np.flatnonzero(candidate)
    if len(usable):
        capture=int(usable[np.argmax(sideways[usable])]);count=int(active[first:capture+1].sum())
        result.update(left_push_before_b_m=float(sideways[capture]),left_push_capture_step=capture,
            left_push_forward_motion_m=float(forward[capture]),left_push_direction_degrees=float(angle[capture]),
            b_hand_gap_at_left_push_m=float(b_gap[capture]),solo_force_contact_frames=count,
            solo_force_contact_fraction=count/(capture-first+1),first_force_push_step=first_force,
            startup_box_motion_m=startup,minimum_partner_gap_m=gap_threshold)
        if count<6 or count/(capture-first+1)<.25:result['rejection'].append('Insufficient sustained solo pushing force')
    if result['left_push_before_b_m']<2.:
        result['rejection'].append('No >=2m physical left push while B is still far and approaching')
    result['left_push_pass']=not result['rejection']
    return result
