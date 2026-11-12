"""Gate unmodified natural scenes in TokenHSI -> FSM -> Ours order."""
import argparse
import json
import os
from pathlib import Path
import isaacgym
import numpy as np
import torch
import search_genuine_push_cases as genuine
from push_comparison_select import analyze, first_run
from push_path_metrics import path_metrics, VERSION
from push_left_metrics import left_push_metrics

SIDEWAYS_PROFILE=os.environ.get('PUSH_STAGE_PROFILE')=='sideways2'
LEFT_PROFILE=os.environ.get('PUSH_STAGE_PROFILE') in ('left2','sideways2')
STAGE_VERSION='fixed_sideways2_v2' if SIDEWAYS_PROFILE else 'fixed_left_push2_v1' if LEFT_PROFILE else VERSION


def select_stage(base, seed, method):
    source=base/f'source/seed{seed}'
    folder=base/f'rollouts/seed{seed}/{method}'
    initial=torch.load(source/'initial.pt',map_location='cpu',weights_only=False)
    placement=json.loads((source/'initial_prefilter.json').read_text())
    roles={r['env_id']:r for r in placement['cases']}
    previous=placement if method=='tokenhsi' else json.loads((base/f'rollouts/seed{seed}/{"tokenhsi" if method=="fsm" else "fsm"}/selection.json').read_text())
    expected=set(previous['env_ids'])
    run=torch.load(folder/'rollout.pt',map_location='cpu',weights_only=False)
    assert all(run['verification']['initial_exact'].values())
    passed=[];evidence=[];ours_candidates=[]
    for index,env in enumerate(run['env_ids']):
        if env not in expected:
            continue
        a=roles[env]['A'];trace=run['trajectory']
        info=analyze(trace,index,initial,env)
        if method=='tokenhsi':
            if LEFT_PROFILE:
                info.update(left_push_metrics(trace,index,initial,env,a,sideways_case=SIDEWAYS_PROFILE))
                info['native_first_person']=info['first_person']
                info['first_person']=a
                info['first_step']=info['physical_a_contact_step']
                if not SIDEWAYS_PROFILE:info['early_capture_step']=info['left_push_capture_step']
                info['genuine_early_push']=info.get('premature_push_verified',info['left_push_pass'])
            else:
                genuine.MIN_PARTNER_GAP=.7
                info=genuine.genuine_metrics(trace,index,initial,env)
                if not info['genuine_early_push']:
                    genuine.MIN_PARTNER_GAP=.5
                    info=genuine.genuine_metrics(trace,index,initial,env)
                info['minimum_partner_gap_m']=genuine.MIN_PARTNER_GAP
            info.update(path_metrics(trace,index,initial,env,info['first_step'],info['success_step']))
            ok=info.get('sideways_case_pass',False) if SIDEWAYS_PROFILE else info['left_push_pass'] if LEFT_PROFILE else (info['genuine_early_push'] and info['first_person']==a and info['path_curve'])
        elif method=='fsm':
            info.update(path_metrics(trace,index,initial,env,info['release_step'],info['success_step']))
            ok=(info['first_frozen_step']>=0 and info['release_step']>info['first_frozen_step']
                and info['joint_contact_fraction']<.85 and info['first_person']==a and info['path_curve'])
            if ok:
                frozen=trace['frozen'][:,index].numpy()
                ok=bool(frozen[info['first_frozen_step'],a])
        else:
            assert method=='ours'
            first,second=info['arrivals'][a],info['arrivals'][1-a]
            position=trace['boxes'][:,index,:3].numpy()
            face=trace['on_face'][:,index].numpy()
            motion=float(np.linalg.norm(position[first:second+1,:2]-position[first,:2],axis=-1).max()) if first>=0 and second>=first else 100.
            early_contact=float(face[first:second+1,a].mean()) if first>=0 and second>=first else 0.
            initial_distance=float(np.linalg.norm(position[0]-initial['center_goals'][env,0].numpy()))
            info.update(path_metrics(trace,index,initial,env,info['all_contact_step'],info['success_step']))
            ok=(info['first_person']==a and first>=0 and second-first>=15 and motion<=.15
                and early_contact>=.7 and info['success_step']>=second
                and info['joint_contact_fraction']>=.95 and initial_distance>=1. and info['path_straight'])
            info.update(ours_pattern=bool(ok),wait_box_motion_m=motion,wait_seconds=(second-first)*initial['dt'],
                phase_steps=[max(first,0)+3,max(info['all_contact_step'],0)+3,max(info['success_step'],0)+10])
            ours_candidates.append(dict(seed=seed,env_id=env,first_person=a,qualifies=bool(ok),
                first_contact_step=first,second_contact_step=second,joint_contact_step=info['all_contact_step'],
                wait_steps=second-first,wait_seconds=info['wait_seconds'],wait_box_displacement_max_m=motion,
                waiting_person_contact_fraction=early_contact,joint_contact_fraction_until_success=info['joint_contact_fraction'],
                success_step=info['success_step'],initial_goal_distance_m=initial_distance,
                minimum_goal_distance_m=info['minimum_goal_error_m']))
        if ok:passed.append(env)
        evidence.append(dict(env_id=env,passes=bool(ok),initial=roles[env],metrics=info))
    report=dict(seed=seed,num_envs=initial['num_envs'],method=method,env_ids=passed,criteria_version=STAGE_VERSION,
        evaluated=len(evidence),passed=len(passed),cases=evidence,initial_state_modified=bool(initial.get('fixed_layout')))
    (folder/'selection.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item())+'\n')
    if method=='ours':
        (source/'candidates.json').write_text(json.dumps(dict(seed=seed,num_envs=initial['num_envs'],candidates=ours_candidates),indent=2)+'\n')
    print(method,seed,'evaluated',report['evaluated'],'passed',len(passed),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--method',choices=['tokenhsi','fsm','ours'],required=True)
    a=p.parse_args();select_stage(a.root,a.seed,a.method)
