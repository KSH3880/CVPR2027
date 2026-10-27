"""Opt-in full Stage1 RSI with graph-first BEFORE placement and bounded fallback."""
import torch
from utils.before_full_rsi import SKILLS, evaluation_weights
from utils.before_layout import transform
from utils.shared_support import rotate
from utils.before_sampling import sample_follower_modes, shorten_leader_xy, sample_before_goal_xy
from utils.stage1_sampling import sample_goal_xy
from utils.size_rsi import resolve_skills


def allocate(t):
    n,m=t.num_envs,t.num_agents
    t._full_rsi_skill=torch.full((n,m),-1,dtype=torch.long,device=t.device)
    t._full_rsi_allowed=torch.ones(n,m,len(SKILLS),dtype=torch.bool,device=t.device)
    t._full_rsi_attempts=torch.zeros(n,m,dtype=torch.long,device=t.device)
    t._full_rsi_invalid=torch.zeros(n,m,dtype=torch.bool,device=t.device)
    t._full_rsi_requested=torch.zeros(3,5,len(SKILLS),device=t.device)
    t._full_rsi_executed=torch.zeros_like(t._full_rsi_requested)
    t._full_rsi_fallbacks=torch.zeros(2,3,5,device=t.device)
    t._full_rsi_frame_counts=torch.zeros(2,device=t.device)


def roles(t,rows,owners):
    pre=t._before_predecessor[rows]
    follower=pre.gather(1,owners[:,None]).flatten()>=0
    leader=(pre==owners[:,None]).any(-1)
    return torch.where(follower,2,leader.long())


def new_graph(t,ids):
    t._full_rsi_skill[ids]=-1;t._full_rsi_allowed[ids]=True
    t._full_rsi_attempts[ids]=0;t._full_rsi_invalid[ids]=False


def reset_ref(t,rows,owners):
    mode=getattr(t.cfg['args'],'before_full_rsi','loco') if t._is_eval else 'mix'
    mixed=mode!='loco'
    templates=t._templates[rows,owners]
    weights=t._template_rsi_weights[templates].clone()
    if t._is_eval:
        weights=evaluation_weights(weights,mode,t._before_predecessor,t._templates,rows,owners)
    if mixed and not hasattr(t,'_size_rsi_cache'):
        from env.tasks.multi_agent.before_full_rsi_cache import BeforeFullRsiCache
        t._size_rsi_cache=BeforeFullRsiCache(t)
    current=t._full_rsi_skill[rows,owners]
    first=current<0
    available=(t._size_rsi_cache.availability(rows,owners,templates) if mixed else weights.bool())
    available &= t._full_rsi_allowed[rows,owners]
    available[:,0]=True
    if first.any():
        requested=torch.multinomial(weights[first],1).flatten()
        selected,fallback=resolve_skills(requested,weights[first],available[first],0)
        current[first]=selected
        rr=roles(t,rows[first],owners[first]);tt=templates[first]
        t._full_rsi_requested.index_put_((rr,tt,requested),torch.ones_like(tt,dtype=torch.float),accumulate=True)
        t._full_rsi_fallbacks[0].index_put_((rr,tt),fallback.float(),accumulate=True)
        if not t._is_eval:
            t._size_rsi_requested.index_put_((tt,requested),torch.ones_like(tt,dtype=torch.float),accumulate=True)
            t._size_rsi_fallbacks.index_add_(0,tt,fallback.float())
    # Placement failure removes only the exhausted skill, then uses Stage1 weights.
    retry=~available.gather(1,current[:,None]).flatten()
    if retry.any():
        selected,_=resolve_skills(current[retry],weights[retry],available[retry],0)
        current[retry]=selected
        rr=roles(t,rows[retry],owners[retry]);tt=templates[retry]
        t._full_rsi_fallbacks[1].index_put_((rr,tt),torch.ones_like(tt,dtype=torch.float),accumulate=True)
    t._full_rsi_skill[rows,owners]=current
    for j,skill in enumerate(SKILLS):
        choose=current==j
        if not choose.any():continue
        r,a=rows[choose],owners[choose];lib=t._motion_lib[skill]
        if mixed:clips,times=t._size_rsi_cache.sample(skill,r,a,templates[choose])
        else:clips=lib.sample_motions(len(r));times=lib.sample_time_rsi(clips)
        root,rot,dofs,vel,ang,dv,_=lib.get_motion_state(clips,times)
        offset=t._env_origins[r]+t._agent_spawn_offsets[a]
        t._set_env_state(r,a,root+offset,rot,dofs,vel,ang,dv)
        pos,quat,linear,angular=lib.get_motion_state_max(clips,times)
        t._kinematic_humanoid_rigid_body_states[r,a]=torch.cat((pos+offset[:,None],quat,linear,angular),-1)
        t._every_env_init_dof_pos[r,a]=dofs
        t._reset_ref_slots[skill]=(r,a);t._reset_ref_motion_ids[skill]=clips;t._reset_ref_motion_times[skill]=times


def move(t,rows,owners,q,shift,payload=False):
    t._humanoid_root_states[rows,owners]=transform(t._humanoid_root_states[rows,owners],q,shift)
    t._kinematic_humanoid_rigid_body_states[rows,owners]=transform(t._kinematic_humanoid_rigid_body_states[rows,owners],q,shift)
    if payload:
        source=t._scenario_physical_object(rows,owners)
        t._box_states[rows,source]=transform(t._box_states[rows,source],q,shift)


def yaw(q):
    x,y,z,w=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def reset_boxes(t,ids,base):
    t._full_rsi_invalid[ids]=False
    # A shared SIT/CLIMB reference must never overwrite the leader-owned box.
    saved=t._reset_ref_slots.copy()
    for skill,(r,a) in saved.items():
        keep=~((t._before_predecessor[r,a]>=0)&(t._templates[r,a]<3))
        t._reset_ref_slots[skill]=(r[keep],a[keep])
        # Base code indexes reference ids alongside these slots.
    saved_ids=t._reset_ref_motion_ids.copy();saved_times=t._reset_ref_motion_times.copy()
    for skill,(r,a) in saved.items():
        keep=~((t._before_predecessor[r,a]>=0)&(t._templates[r,a]<3))
        t._reset_ref_motion_ids[skill]=saved_ids[skill][keep];t._reset_ref_motion_times[skill]=saved_times[skill][keep]
    base(ids)
    t._reset_ref_slots=saved;t._reset_ref_motion_ids=saved_ids;t._reset_ref_motion_times=saved_times
    rr,b=(t._before_predecessor[ids]>=0).nonzero(as_tuple=True);r=ids[rr];a=t._before_predecessor[r,b]
    modes=t._before_distance_modes[r,b];mix=t.cfg['env']['before']['distance_mixture']
    short=(modes==2)&(t._full_rsi_skill[r,a]==0)
    e,h=r[short],a[short]
    delta=shorten_leader_xy(t._humanoid_root_states[e,h,:2],t._box_states[e,h,:2])-t._humanoid_root_states[e,h,:2]
    t._humanoid_root_states[e,h,:2]+=delta;t._kinematic_humanoid_rigid_body_states[e,h,:,:2]+=delta[:,None]
    skill=t._full_rsi_skill[r,b];template=t._templates[r,b]
    free=(skill==0)|((template==4)&((skill==5)|(skill==6)))
    e,h=r[free],b[free]
    desired=sample_follower_modes(t._box_states[r[free],a[free],:2],template[free],modes[free],mix)
    q=torch.zeros(len(e),4,device=t.device);q[:,3]=1
    shift=torch.zeros(len(e),3,device=t.device);shift[:,:2]=desired-t._humanoid_root_states[e,h,:2]
    # Locomotion source is resampled below; advanced payload follows the actor.
    move(t,e,h,q,shift,payload=False)
    advanced=skill[free]!=0
    t._box_states[e[advanced],h[advanced],:2]+=shift[advanced,:2]
    ground=(template[free]==4)&~advanced
    t._box_states[e[ground],h[ground],:2]=sample_follower_modes(desired[ground],template[free][ground],modes[free][ground],mix,payload=True)
    # Shared-target RSI is aligned in XY/yaw only. Never lift/tilt the humanoid to fit an airborne support.
    for name in ('sit','climb','putDown'):
        if name not in saved:continue
        es,hs=saved[name];sel=(t._before_predecessor[es,hs]>=0)
        if name=='putDown':sel &= t._templates[es,hs]==4
        es,hs=es[sel],hs[sel]
        if not len(es):continue
        lead=t._before_predecessor[es,hs];target=t._box_states[es,lead]
        lib=t._motion_lib[name];clips=saved_ids[name][sel]
        if name=='putDown':
            pos,_=lib.get_obj_motion_state(clips,lib.get_motion_length(clips))
            rot=torch.zeros(len(es),4,device=t.device);rot[:,3]=1
        else:pos,rot=lib.get_obj_motion_state_single_frame(clips)
        pivot=pos+t._env_origins[es]+t._agent_spawn_offsets[hs]
        angle=yaw(target[:,3:7])-yaw(rot)
        q=torch.zeros(len(es),4,device=t.device);q[:,2]=torch.sin(angle/2);q[:,3]=torch.cos(angle/2)
        shift=target[:,:3]-rotate(q,pivot);shift[:,2]=0
        move(t,es,hs,q,shift,payload=name=='putDown')
        upright=(target[:,3:5].square().sum(-1)<1e-4)
        grounded=(target[:,2]-t._box_size[es,lead,2]/2).abs()<.02
        t._full_rsi_invalid[es,hs]=~(upright&grounded)


def reset_goals(t,ids,base):
    base(ids)
    rr,b=(t._before_predecessor[ids]>=0).nonzero(as_tuple=True);r=ids[rr];a=t._before_predecessor[r,b]
    keep=t._full_rsi_skill[r,a]!=7;r,a,b=r[keep],a[keep],b[keep]
    goal=t._scenario_goal(r,a)
    t._tar_pos[r,goal,:2]=sample_before_goal_xy(t._humanoid_root_states[r,a,:2],t._box_states[r,a,:2],
        t._before_distance_modes[r,b],t.cfg['env']['before']['distance_mixture'])


def ontop_targets(t,ids):
    rr,a=((t._templates[ids]==4)&(t._before_predecessor[ids]<0)).nonzero(as_tuple=True);r=ids[rr]
    put=t._full_rsi_skill[r,a]==7
    e,h=r[~put],a[~put]
    t._box_states[e,3*t.num_agents+h,:2]=sample_goal_xy(t._humanoid_root_states[e,h,:2],t._box_states[e,h,:2],t._env_origins[e,:2])
    if 'putDown' in t._reset_ref_slots:
        es,hs=t._reset_ref_slots['putDown'];sel=(t._templates[es,hs]==4)&(t._before_predecessor[es,hs]<0)
        e,h=es[sel],hs[sel];lib=t._motion_lib['putDown'];clips=t._reset_ref_motion_ids['putDown'][sel]
        if len(e):
            pos,_=lib.get_obj_motion_state(clips,lib.get_motion_length(clips))
            support=3*t.num_agents+h
            t._box_states[e,support,:2]=(pos+t._env_origins[e]+t._agent_spawn_offsets[h])[:,:2]
            t._box_states[e,support,2]=t._box_size[e,support,2]/2
            t._box_states[e,support,3:7]=t._box_states.new_tensor([0.,0.,0.,1.])
            t._box_states[e,support,7:13]=0
    t._place_distractors(ids)


def screen(t,ids,base):
    bad=base(ids)|t._full_rsi_invalid[ids].any(-1)
    body=t._kinematic_humanoid_rigid_body_states[ids,:,:,:3]
    bad |= (body[:,:,t._key_body_ids[-2:],2]<-.02).any(-1).any(-1)
    # Count only final accepted choices; retries never redraw graph/mode/initial request.
    r,a=(~bad[:,None].expand(-1,t.num_agents)).nonzero(as_tuple=True);es=ids[r]
    t._full_rsi_executed.index_put_((roles(t,es,a),t._templates[es,a],t._full_rsi_skill[es,a]),torch.ones_like(a,dtype=torch.float),accumulate=True)
    if getattr(t, '_is_eval', False) and getattr(t.cfg['args'], 'before_full_rsi', 'loco') == 'both_carry':
        good = ids[~bad]
        row, follower = ((t._before_predecessor[good] >= 0) & (t._templates[good] == 4)).nonzero(as_tuple=True)
        if len(row):
            envs = good[row]; leader = t._before_predecessor[envs, follower]
            lc = t._full_rsi_skill[envs, leader] == 6
            fc = t._full_rsi_skill[envs, follower] == 6
            print(f'[BEFORE both_carry] pairs={len(row)} leader={int(lc.sum())} follower={int(fc.sum())} both={int((lc & fc).sum())}; others fell back to loco', flush=True)
    explicit=t._full_rsi_invalid[ids]
    candidates=torch.where(explicit.any(-1)[:,None],explicit,bad[:,None].expand(-1,t.num_agents))
    r,a=(candidates&(t._full_rsi_skill[ids]>0)).nonzero(as_tuple=True);es=ids[r]
    t._full_rsi_attempts[es,a]+=1
    exhausted=t._full_rsi_attempts[es,a]>=t.cfg['env']['before']['full_rsi']['placement_retry_limit']
    es,a=es[exhausted],a[exhausted]
    t._full_rsi_allowed[es,a,t._full_rsi_skill[es,a]]=False
    t._full_rsi_attempts[es,a]=0
    return bad


def diagnostics(t):
    result={}
    for role,label in enumerate(('solo','leader','follower')):
        for template,name in enumerate(('holding','sit','climb','at','ontop')):
            for skill in (0,1,2,5,6,7):
                key=f'rsi/{label}/{name}/{SKILLS[skill]}'
                result[key+'_requested']=t._full_rsi_requested[role,template,skill].clone()
                result[key+'_executed']=t._full_rsi_executed[role,template,skill].clone()
            for i,reason in enumerate(('size_fallback','placement_fallback')):
                result[f'rsi/{label}/{name}/{reason}']=t._full_rsi_fallbacks[i,role,template].clone()
    result['rsi/late_branch_fraction']=t._full_rsi_frame_counts[1]/t._full_rsi_frame_counts.sum().clamp_min(1)
    for value in (t._full_rsi_requested,t._full_rsi_executed,t._full_rsi_fallbacks,t._full_rsi_frame_counts):value.zero_()
    return result
