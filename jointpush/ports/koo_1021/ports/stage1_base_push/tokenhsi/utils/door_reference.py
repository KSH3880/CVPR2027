"""Geometry-screened existing manipulation references; these are Door proxies."""
import math
import torch


def yaw_from_quat(q):
    x,y,z,w=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def rotate(q,v):
    uv=torch.cross(q[...,:3],v,dim=-1)
    return v+2*(q[...,3:]*uv+torch.cross(q[...,:3],uv,dim=-1))


def multiply(a,b):
    av,bv=a[...,:3],b[...,:3]
    return torch.cat((a[...,3:]*bv+b[...,3:]*av+torch.cross(av,bv,dim=-1),
        a[...,3:]*b[...,3:]-(av*bv).sum(-1,keepdim=True)),-1)


def candidate_mask(positions,root_rot,root_vel,hands,feet,spec):
    root=positions[:,0]
    hand=positions[:,hands]
    yaw=yaw_from_quat(root_rot)
    forward=torch.stack((torch.cos(yaw),torch.sin(yaw)),-1)
    reach=((hand[...,:2]-root[:,None,:2])*forward[:,None]).sum(-1)
    hand_ok=((hand[...,2]-spec['handle_height']).abs()<=spec['height_tolerance']) & (reach>=spec['min_forward_reach'])
    return ((root[:,2]>=spec['min_root_height']) &
        (root_vel[:,:2].norm(dim=-1)<=spec['max_root_speed']) &
        (positions[:,feet,2].amax(-1)<=spec['max_foot_height']) & hand_ok.any(-1))


class DoorReferenceBank:
    def __init__(self,libraries,hands,feet,spec,history_duration):
        self.spec=spec
        self.pools={};self.counts={}
        for skill in spec['interaction_skills']:
            lib=libraries[skill]
            ids=[];times=[];weights=[];late_weights=[]
            for motion_id in range(lib.num_motions()):
                start=int(lib.length_starts[motion_id]);n=int(lib._motion_num_frames[motion_id])
                sl=slice(start,start+n)
                valid=candidate_mask(lib.gts[sl],lib.grs[sl,0],lib.grvs[sl],hands,feet,spec)
                dt=float(lib._motion_dt[motion_id])
                history=math.ceil(history_duration/dt)
                # Each complete AMP history window must satisfy the same screen.
                window=history+1
                eligible=torch.zeros_like(valid)
                if n>=window:
                    eligible[history:]=valid.unfold(0,window,1).all(-1)
                index=eligible.nonzero().flatten()
                if not len(index):continue
                count=len(index);weight=float(lib._motion_weights[motion_id])
                ids.append(torch.full_like(index,motion_id))
                times.append(index.float()*dt)
                weights.append(torch.full_like(index,weight/count,dtype=torch.float))
                late=torch.zeros(count,device=index.device)
                begin=int(count*(1-spec['late_fraction']))
                late[begin:]=weight/(count-begin)
                late_weights.append(late)
            if not ids:
                raise ValueError('No geometry-valid Door proxy reference windows: '+skill)
            self.pools[skill]=(torch.cat(ids),torch.cat(times),torch.cat(weights),torch.cat(late_weights))
            self.counts[skill]=len(self.pools[skill][0])

    def sample(self,skill,count,late=False):
        ids,times,weights,late_weights=self.pools[skill]
        if late:
            choice=torch.multinomial(weights,count,replacement=True)
            requested=torch.rand(count,device=weights.device)<self.spec['late_probability']
            if requested.any():
                choice[requested]=torch.multinomial(late_weights,int(requested.sum()),replacement=True)
        else:
            choice=torch.multinomial(weights,count,replacement=True)
        return ids[choice],times[choice]


def align_reference(body_pos,body_rot,body_vel,body_ang,root_rot,target,hands,stand_off,min_forward_reach=.2):
    """Yaw toward +Y and align the closest-height hand XY in front of own Door."""
    yaw=math.pi/2-yaw_from_quat(root_rot)
    q=torch.zeros_like(root_rot);q[:,2]=torch.sin(yaw/2);q[:,3]=torch.cos(yaw/2)
    root=body_pos[:,0]
    relative=rotate(q[:,None].expand_as(body_rot),body_pos-root[:,None])
    reach=relative[:,hands,1]
    score=(body_pos[:,hands,2]-target[:,None,2]).abs()
    score=score.masked_fill(reach<min_forward_reach,float('inf'))
    if not torch.isfinite(score.amin(-1)).all():
        raise ValueError('Door RSI sample has no forward-reaching hand')
    hand=score.argmin(-1)
    hand_ids=torch.tensor(hands,device=root.device)[hand]
    rows=torch.arange(len(root),device=root.device)
    new_root=root.clone()
    new_root[:,:2]=target[:,:2]-relative[rows,hand_ids,:2]
    new_root[:,1]-=stand_off
    return (relative+new_root[:,None],multiply(q[:,None].expand_as(body_rot),body_rot),
        rotate(q[:,None].expand_as(body_rot),body_vel),
        rotate(q[:,None].expand_as(body_rot),body_ang),q)
