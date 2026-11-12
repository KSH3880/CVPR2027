"""Measure path bending separately from goal error and box yaw rotation."""
import numpy as np

VERSION='path_curve_v2'


def path_metrics(trace,index,initial,env,start=0,success=-1):
    position=trace['boxes'][:,index,:3].numpy()
    alive=trace['alive'][:,index].all(-1).numpy()
    times=np.arange(len(position))
    mask=alive & (times>=max(start,0))
    if success>=max(start,0):mask &= times<=success
    frames=np.flatnonzero(mask)
    result=dict(path_curve=False,path_straight=False,path_bow_m=0.,path_turn_degrees=0.,
        path_goal_line_lateral_m=0.,path_length_m=0.,path_peak_step=-1,box_yaw_change_degrees=0.)
    if len(frames)<2:return result
    points=position[frames,:2]
    goal=initial['center_goals'][env,0,:2].numpy()
    axis=goal-position[0,:2];axis/=np.linalg.norm(axis).clip(1e-8)
    delta=points-position[0,:2]
    lateral=np.abs(delta[:,0]*axis[1]-delta[:,1]*axis[0])
    # Spatial sampling ignores stationary contact jitter and retains real turns.
    keep=[0]
    for k in range(1,len(points)):
        if np.linalg.norm(points[k]-points[keep[-1]])>=.05:keep.append(k)
    if keep[-1]!=len(points)-1:keep.append(len(points)-1)
    sampled=points[keep]
    lengths=np.linalg.norm(np.diff(sampled,axis=0),axis=-1)
    arc=np.r_[0.,np.cumsum(lengths)]
    length=float(arc[-1])
    chord=points[-1]-points[0];norm=float(np.linalg.norm(chord))
    if norm>=.2:
        unit=chord/norm;offset=points-points[0]
        bow=np.abs(offset[:,0]*unit[1]-offset[:,1]*unit[0])
        peak=int(bow.argmax());bow_max=float(bow[peak])
    else:
        peak=int(np.linalg.norm(points-points[0],axis=-1).argmax())
        bow_max=float(np.linalg.norm(points[peak]-points[0]))
    turn=0.
    if length>=.8:
        grid=np.arange(0,length+.00001,.2)
        uniform=np.stack([np.interp(grid,arc,sampled[:,d]) for d in (0,1)],-1)
        segments=uniform[2:]-uniform[:-2]
        segments=segments[np.linalg.norm(segments,axis=-1)>.15]
        if len(segments):
            headings=np.unwrap(np.arctan2(segments[:,1],segments[:,0]))
            turn=float(np.degrees(headings.max()-headings.min()))
        # Compare 40cm travel segments, rather than noisy frame velocities.
        for k in range(2,len(uniform)-2):
            before=uniform[k]-uniform[k-2];after=uniform[k+2]-uniform[k]
            scale=np.linalg.norm(before)*np.linalg.norm(after)
            if scale>.02:
                turn=max(turn,float(np.degrees(np.arccos(np.clip(before@after/scale,-1,1)))))
    q=trace['boxes'][frames,index,3:7].numpy()
    yaw=np.unwrap(np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2)))
    lateral_max=float(lateral.max())
    result.update(path_bow_m=bow_max,path_turn_degrees=turn,path_goal_line_lateral_m=lateral_max,
        path_length_m=length,path_peak_step=int(frames[peak]),
        box_yaw_change_degrees=float(np.degrees(np.abs(yaw-yaw[0]).max())),
        path_curve=bool(length>=.8 and bow_max>=.4 and lateral_max>=.4 and turn>=20.),
        path_straight=bool(norm>=.5 and bow_max<=.15 and lateral_max<=.15))
    return result
