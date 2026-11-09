"""CPU geometry and temporal audit for saved multi-person random demonstrations."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from utils.evaluation_protocol import spatial_success
from learning.multi_agent.random_demo import candidates, near_waits


def validate(directory):
    directory=Path(directory);meta=json.loads((directory/'metadata.json').read_text());x=np.load(directory/'trajectory.npz')
    t,m=len(x['roots']),meta['num_agents'];order=x['logical_box_order']
    packet=torch.tensor(x['packet'])[None].expand(t,-1,-1)
    spatial=spatial_success(packet,torch.tensor(x['roots'][:,:m,:3]),torch.tensor(x['roots'][:,m+order]),
        torch.tensor(x['sizes_physical'][order])[None].expand(t,-1,-1),torch.tensor(x['goals'])[None].expand(t,-1,-1))[...,1].numpy()
    assert np.array_equal(spatial,x['spatial'])
    if meta.get('continue_after_individual_failure'):
        assert not (x['native'] & ~x['agent_valid']).any()
        assert not ((~x['agent_valid'][:-1]) & x['agent_valid'][1:]).any()
    if not meta.get('continue_after_individual_failure',False) and 'agent_valid' in x:
        assert x['agent_valid'].all(), 'Selected scene includes a fallen agent'
    pairs=np.column_stack((np.zeros(len(x['pairs']),int),x['pairs']))
    ranked=candidates(spatial[:,None],x['native'][:,None],x['valid'][:,None],x['pair_success'],x['initial_gate'],pairs,meta['dt'],meta.get('minimum_success'),meta.get('minimum_pairs',1))
    assert ranked and ranked[0]['count']==meta['success_count']
    assert ranked[0]['successful']==meta['successful_agents']
    assert ranked[0]['frame']+1==meta['success_step']
    report=dict(cpu_geometry_verified=True,temporal_verified=True,all_before_pairs_success=True,
        success_count=meta['success_count'],total=m,before_pairs=meta['before_pairs'],successful_agents=meta['successful_agents'],
        failed_agents=sorted(set(range(m))-set(meta['successful_agents'])),success_seconds=meta['success_seconds'])
    names=['SIT','CLIMB','AT','ON_TOP','PUSH']
    report['tasks']=[dict(agent=a,task=names[int(x['packet'][a,1])],success=a in meta['successful_agents']) for a in range(m)]
    if meta.get('wait_radius',0):
        waits=near_waits(x['roots'],x['native'],x['valid'],x['packet'],x['goals'],x['pairs'],meta['dt'],meta['wait_radius'],meta.get('wait_seconds',1.),meta.get('wait_extent',0.),meta.get('wait_require_walk',False),x['initial_roots'])
        assert waits is not None
        assert waits==meta['near_waits']
        report['near_waits_verified']=waits
    (directory/'validation.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');args=p.parse_args()
    print(json.dumps(validate(args.case),indent=2))
