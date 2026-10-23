"""PUSH adaptation of SizeRsiCache's per-size valid-frame and late-frame pools."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from utils.size_rsi import late_frames, size_key
from utils.push_reset_geometry import reset_geometry_contract


class PushNativeRsiCache:
    def __init__(self, task):
        self.task=task;self.spec=task.cfg['env']['pushTask'];self.lib=task._motion_lib['push']
        path=Path(self.spec['rsi_cache_file'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.spec['rsi_cache_sha256']:
            raise ValueError('PUSH native RSI cache differs from checkpoint/config contract')
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']))
            contract=reset_geometry_contract(task.cfg['env'])
            if contract is not None and meta.get('reset_geometry')!=contract:
                raise ValueError('PUSH native RSI reset-collider contract differs')
            if (meta['motion_hashes']!=self.spec['motion_hashes'] or meta['box']!=task.cfg['env']['pushBox']
                    or meta['sim']!=task.cfg['sim']):
                raise ValueError('PUSH native RSI cache motion/physical contract differs')
            limit=self.spec.get('rsi_max_hand_midpoint_height_m')
            if limit is not None:
                expected=dict(kind='native_hand_midpoint_z',max_m=limit,include_safe_floor_shift=True)
                if meta.get('height_selection')!=expected:
                    raise ValueError('PUSH RSI height-selection cache contract differs')
                if 'placed_hand_midpoint_z_m' not in data or 'reference_hand_midpoint_z_m' not in data:
                    raise ValueError('PUSH RSI height-selection measurements are missing')
                selected=data['valid'].any(axis=0)
                if (not np.isfinite(data['placed_hand_midpoint_z_m'][selected]).all()
                        or not np.isfinite(data['reference_hand_midpoint_z_m'][selected]).all()
                        or (data['placed_hand_midpoint_z_m'][selected]>limit).any()
                        or (data['reference_hand_midpoint_z_m'][selected]>limit).any()):
                    raise ValueError('PUSH RSI cache contains frames above the height limit')
            valid=data['valid'];self.keys=[size_key(s) for s in data['sizes']]
            offsets=data['clip_offsets']
        self.lookup={key:i for i,key in enumerate(self.keys)}
        keys=torch.tensor(self.keys,dtype=torch.long)
        self.profile_min=keys.amin(0).to(task.device)
        self.profile_max=keys.amax(0).to(task.device)
        shape=tuple((keys.amax(0)-keys.amin(0)+1).tolist())
        self.profile_grid=torch.full(shape,-1,device=task.device,dtype=torch.long)
        for i,key in enumerate(keys):self.profile_grid[tuple((key-keys.amin(0)).tolist())]=i
        profiles=len(self.keys);clips=self.lib.num_motions()
        starts=torch.zeros(profiles,clips,2,dtype=torch.long);counts=torch.zeros_like(starts)
        flat=[];total=0
        for i in range(profiles):
            for clip in range(clips):
                mask=torch.from_numpy(valid[i,offsets[clip]:offsets[clip+1]])
                # CARRY marks the entire motion as interaction phase; PUSH does
                # the same. Use the exact existing last-30%-of-valid helper.
                pools=(mask.nonzero().flatten(),late_frames(mask,torch.ones_like(mask)))
                for tail,frames in enumerate(pools):
                    starts[i,clip,tail]=total;counts[i,clip,tail]=len(frames)
                    flat.append(frames);total+=len(frames)
        self.starts=starts.to(task.device);self.counts=counts.to(task.device)
        self.frames=torch.cat(flat).to(task.device)
        self.weights=(self.counts[...,0]>0)*self.lib._motion_weights[None]
        if self.spec.get('rsi_max_hand_midpoint_height_m') is not None and (self.weights.sum(-1)<=0).any():
            raise ValueError('PUSH RSI height filter leaves an empty size profile; no fallback is permitted')

    def profiles(self,sizes):
        quantized=(sizes*20).round().long()
        if ((quantized<self.profile_min)|(quantized>self.profile_max)).any():raise ValueError('PUSH RSI size is outside the screened grid')
        index=quantized-self.profile_min
        result=self.profile_grid[index[:,0],index[:,1],index[:,2]]
        if (result<0).any():raise ValueError('PUSH RSI size has no cache profile')
        return result

    def available(self,sizes):
        return self.weights[self.profiles(sizes)].sum(-1)>0

    def sample(self,sizes):
        profile=self.profiles(sizes)
        clip=torch.multinomial(self.weights[profile],1).flatten()
        late=torch.rand(len(profile),device=sizes.device)<self.spec['late_probability']
        count=self.counts[profile,clip,late.long()]
        offset=self.starts[profile,clip,late.long()]
        frame=self.frames[offset+(torch.rand(len(profile),device=sizes.device)*count).long()]
        times=frame*self.lib._motion_dt[clip];phase=times/self.lib._motion_lengths[clip]
        return dict(motion_id=clip,time=times,phase=phase,box_progress=phase*self.spec['rsi_box_progress_scale'],late_requested=late)
