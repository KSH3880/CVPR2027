"""Recompute point/region and BEFORE metrics from a saved trace without simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from utils.evaluation_protocol import spatial_success, coupled_current_success, coupled_scores
from utils.evaluation_protocol import shared_center_success, shared_center_scores, shared_centers_from_tokens
from utils.evaluation_protocol import interaction_group_scores
from utils.evaluation_errors import error_scores, update_error_minima


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
        groups=torch.from_numpy(data['groups'])
        has_coupled=bool(((groups[:,:,None]==groups[:,None,:]).sum(-1)>1).any())
        minimum_errors=torch.full((n,m,2),float('inf'),dtype=torch.float64)
        center_minimum=torch.full((n,m),float('inf'),dtype=torch.float64) if has_coupled else None
        error_center_ever=torch.zeros(n,m,dtype=torch.bool) if has_coupled else None
        error_centers=error_goals=None
        if individual_goals:
            groups=torch.from_numpy(data['groups'])
            group_ever=torch.zeros_like(ever)
            center_ever=torch.zeros(n,m,dtype=torch.bool)
            center_ever32=torch.zeros_like(center_ever)
            center_precision_differences=0
            physical_centers='shared_centers' in data
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
                if physical_centers:
                    centers=torch.from_numpy(data['shared_centers'][frame])
                    center_goals=torch.from_numpy(data['shared_center_goals'])
                else:
                    centers,center_goals=shared_centers_from_tokens(packet,
                        torch.from_numpy(data['boxes'][frame]),goals,groups)
                center_current=shared_center_success(centers,center_goals,groups,valid)
                center_current32=shared_center_success(centers,center_goals,groups,valid,dtype=torch.float32)
                center_ever |= center_current
                center_ever32 |= center_current32
                center_precision_differences += int((center_current32 != center_current).sum())
            if has_coupled:
                if individual_goals:
                    error_centers,error_goals=centers,center_goals
                else:
                    error_centers,error_goals=shared_centers_from_tokens(packet,
                        torch.from_numpy(data['boxes'][frame]),goals,groups)
                error_center_ever |= shared_center_success(error_centers,error_goals,groups,valid)
            update_error_minima(minimum_errors,packet,torch.from_numpy(data['roots'][frame]),
                torch.from_numpy(data['boxes'][frame]),sizes,goals,valid,
                center_minimum,error_centers,error_goals,groups)
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
            enabling_success=recorded.get('enabling_success','both_ever')
            interactions=interaction_group_scores(packet[...,0].bool(),parent,groups,ever,center_ever,
                simultaneous,enabling_success=enabling_success)
            if 'interaction_group_scores' in recorded:
                assert recorded['interaction_group_scores']==interactions
            center_scores=shared_center_scores(groups,center_ever,center_current)
            if 'shared_center_scores' in recorded:
                assert recorded['shared_center_scores']==center_scores
            if physical_centers:
                assert np.array_equal(center_ever.numpy(),data['shared_center_ever'])
                assert np.array_equal(center_current.numpy(),data['shared_center_final'])
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
        if individual_goals and physical_centers:
            result['finite'] &= bool(np.isfinite(data['shared_centers']).all() and np.isfinite(data['shared_center_goals']).all())
        assert result['finite']
        errors=error_scores(packet,parent,groups,ever,simultaneous,minimum_errors,
                            error_center_ever,center_minimum)
        if 'errors' in recorded:
            for section in ('per_task','coupled_by_size'):
                for name,row in errors[section].items():
                    expected=recorded['errors'][section][name]
                    assert row['successes']==expected['successes']
                    for metric in ('point_error_cm','region_error_cm'):
                        assert (row[metric] is None and expected[metric] is None) or (
                            row[metric] is not None and expected[metric] is not None
                            and abs(row[metric]-expected[metric])<=1e-6), (section,name,metric)
            assert recorded['errors']['protocol']==errors['protocol']
        if 'minimum_errors' in data:
            assert np.allclose(minimum_errors.numpy(),data['minimum_errors'],rtol=0,atol=1e-8)
        if 'shared_center_minimum_error' in data:
            assert np.allclose(center_minimum.numpy(),data['shared_center_minimum_error'],rtol=0,atol=1e-8)
        result['errors']=errors
        result['errors_match_recorded']=True if 'errors' in recorded else None
        if individual_goals:
            result.update(enabling_success=enabling_success,
                interaction_group_scores=interactions, shared_center_scores=center_scores,
                shared_center_source='physical_trace' if physical_centers else 'legacy_tokens_mean',
                shared_center_cpu_gpu_exact=True if physical_centers else None,
                shared_center_float32_float64_frame_differences=center_precision_differences,
                shared_center_float32_float64_ever_differences=int((center_ever32 != center_ever).sum()))
    (path/'rescore.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path)
    args=parser.parse_args()
    paths=[args.path] if (args.path/'trace.npz').exists() else sorted(p.parent for p in args.path.glob('*/trace.npz'))
    for path in paths:rescore(path)
