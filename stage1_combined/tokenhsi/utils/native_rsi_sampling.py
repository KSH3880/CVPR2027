"""Full-safe/late-phase RSI pools matching the baseline's 70:30 selection."""
import numpy as np
import torch


def audited_candidates(data, full_motion=False):
    valid=data['static_pass']&data['physics_short_pass']&data['physics_hold_pass']
    if not full_motion:return valid&data['phase']
    # Preserve approach/lowering, but exclude post-interaction frames as baseline does.
    for clip in np.unique(data['clip']):
        same=data['clip']==clip;phase=same&data['phase']
        if phase.any():valid&=~same|(data['frame']<=data['frame'][phase].max())
    return valid


def pack_pools(clips, frames, phase, weights):
    starts=np.zeros((len(weights),2),dtype=np.int64);counts=starts.copy();flat=[];offset=0
    for clip in range(len(weights)):
        ids=np.flatnonzero(clips==clip);ids=ids[np.argsort(frames[ids])]
        late=ids[phase[ids]];late=late[-max(1,(len(late)*3+9)//10):]
        if not len(late):late=ids
        for kind,pool in enumerate([ids,late]):
            starts[clip,kind]=offset;counts[clip,kind]=len(pool);flat.extend(pool.tolist());offset+=len(pool)
    weight=np.asarray(weights,dtype=np.float32)*(counts[:,0]>0)
    assert weight.sum()>0
    return starts,counts,np.asarray(flat,dtype=np.int64),weight/weight.sum()


def sample_pools(starts,counts,flat,weights,n,late_probability):
    clip=torch.multinomial(weights,n,replacement=True)
    late=(torch.rand(n,device=weights.device)<late_probability).long()
    count=counts[clip,late]
    index=starts[clip,late]+(torch.rand(n,device=weights.device)*count).long()
    return flat[index],late.bool()
