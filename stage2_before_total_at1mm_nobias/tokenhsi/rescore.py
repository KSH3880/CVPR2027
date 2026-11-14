"""Recompute point/region and BEFORE metrics from a saved trace without simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from utils.evaluation_protocol import spatial_success, coupled_current_success, coupled_scores


def rescore(path):
    torch.set_num_threads(1)
    with np.load(path/'trace.npz') as archive:
        data = {name: archive[name] for name in archive.files}
        packet=torch.from_numpy(data['packet'])
        sizes=torch.from_numpy(data['sizes'])
        goals=torch.from_numpy(data['goals'])
        parent=torch.from_numpy(data['predecessor'])
        dependent=parent >= 0
        n,m=parent.shape
        rows=torch.arange(n)[:,None]
        ever=torch.zeros(n,m,2,dtype=torch.bool)
        simultaneous=torch.zeros_like(ever)
        ever32=torch.zeros_like(ever)
        simultaneous32=torch.zeros_like(ever)
        previous=torch.ones(n,m,dtype=torch.bool)
        recorded=json.loads((path/'result.json').read_text())
        individual_goals=recorded.get('push_goal_mode')=='individual'
        if individual_goals:
            groups=torch.from_numpy(data['groups'])
            group_ever=torch.zeros_like(ever)
        for frame in range(len(data['roots'])):
            valid=torch.from_numpy(data['valid'][frame])
            assert not (valid & ~previous).any(), 'A fallen person was reactivated'
            current=spatial_success(packet,torch.from_numpy(data['roots'][frame]),
                torch.from_numpy(data['boxes'][frame]),sizes,goals) & valid[...,None]
            current32=spatial_success(packet,torch.from_numpy(data['roots'][frame]),
                torch.from_numpy(data['boxes'][frame]),sizes,goals,dtype=torch.float32) & valid[...,None]
            ever32 |= current32
            simultaneous32 |= current32 & current32[rows,parent.clamp_min(0)] & dependent[...,None]
            ever |= current
            if individual_goals:
                group_current=coupled_current_success(current,groups)
                group_ever |= group_current
            simultaneous |= current & current[rows,parent.clamp_min(0)] & dependent[...,None]
            previous=valid
        both=ever & ever[rows,parent.clamp_min(0)] & dependent[...,None]
        for name,value in [('ever',ever),('both_ever',both),('simultaneous',simultaneous)]:
            assert np.array_equal(value.numpy(),data[name]), f'{name} CPU/GPU mismatch'
        steps=len(data['roots'])
        recorded_horizon=json.loads((path/'result.json').read_text())['horizon']
        valid=data['valid']
        falls=data['fall_step']
        # A fall's terminal frame is included; later frames are excluded.
        for env,agent in np.argwhere(falls >= 0):
            step=falls[env,agent]
            assert valid[step-1,env,agent]
            assert not valid[step:,env,agent].any()
        assert steps==recorded_horizon or not valid[-1].any() or np.all((falls>=0)&(falls<=steps))
        recorded=json.loads((path/'result.json').read_text())
        if individual_goals:
            assert recorded['coupled_scores']==coupled_scores(groups,ever,current,group_ever,group_current)
        for label,idx in [('point',0),('region',1)]:
            for kind,name in enumerate(('SIT','CLIMB','AT','ON_TOP','PUSH')):
                if name not in recorded['scores'][label]:
                    assert not ((packet[...,1]==kind) & packet[...,0].bool()).any()
                    continue
                mask=(packet[...,1]==kind) & packet[...,0].bool()
                pair_mask=mask & dependent
                expected=dict(trials=int(mask.sum()),success=int(ever[...,idx][mask].sum()),
                    pair_trials=int(pair_mask.sum()),both_ever_success=int(both[...,idx][pair_mask].sum()),
                    simultaneous_success=int(simultaneous[...,idx][pair_mask].sum()),
                    follower_individual_success=int(ever[...,idx][pair_mask].sum()))
                for key,value in expected.items():
                    if key in recorded['scores'][label][name]:
                        assert recorded['scores'][label][name][key]==value,(label,name,key)
                if expected['trials']:
                    assert abs(recorded['scores'][label][name]['rate']-expected['success']/expected['trials']) < 1e-12
        result=dict(run=str(path),frames=steps,agents=n*m,cpu_gpu_exact=True,json_counts_exact=True,
                    float32_float64_ever_differences=int((ever32 != ever).sum()),
                    float32_float64_simultaneous_differences=int((simultaneous32 != simultaneous).sum()),
                    finite=bool(np.isfinite(data['roots']).all() and np.isfinite(data['boxes']).all()))
        assert result['finite']
    (path/'rescore.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path)
    args=parser.parse_args()
    paths=[args.path] if (args.path/'trace.npz').exists() else sorted(p.parent for p in args.path.glob('*/trace.npz'))
    for path in paths:rescore(path)
