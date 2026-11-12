"""Review recorded seed/env cases, including rejected FSM runs; no policy runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'tokenhsi'),str(Path(__file__).parent)]
import isaacgym
import torch
from push_comparison_select import analyze
from push_path_metrics import path_metrics


def build(args):
    base=args.root.resolve();source=base/'source';roll=base/'rollouts'
    available={}
    for proof in roll.glob('seed*/*/verification.json'):
        if proof.parent.name in ('tokenhsi','fsm','ours'):
            available[(int(proof.parent.parent.name[4:]),proof.parent.name)]=proof.parent/'rollout.pt'
    requested=[tuple(map(int,k.split(':'))) for k in args.cases] if args.cases else None
    if args.seed is not None or args.env is not None:
        if args.seed is None or args.env is None:raise ValueError('Use --seed and --env together')
        requested=[(args.seed,args.env)]
    needed_seeds={s for s,e in requested} if requested is not None else {s for s,m in available if m=='fsm'}
    runs={key:torch.load(path,map_location='cpu',weights_only=False,mmap=True) for key,path in available.items() if key[0] in needed_seeds}
    if requested is None:
        requested=[]
        for (seed,method),run in runs.items():
            if method!='fsm' or (seed,'tokenhsi') not in runs:continue
            requested.extend((seed,e) for e in run['env_ids'] if e in runs[(seed,'tokenhsi')]['env_ids'])
        requested=sorted(set(requested),key=lambda k:(k!=(8,930),k[0],k[1]))
    if not requested:
        print('Waiting for completed TokenHSI/FSM recordings',flush=True);return False
    cases=[]
    for seed,env in requested:
        initial=torch.load(source/f'seed{seed}/initial.pt',map_location='cpu',weights_only=False)
        roles=json.loads((source/f'seed{seed}/initial_prefilter.json').read_text())['cases']
        early=next(r['A'] for r in roles if r['env_id']==env)
        methods={};labels=[];pending={}
        for method in ('tokenhsi','fsm','ours'):
            run=runs.get((seed,method))
            if run is None or env not in run['env_ids']:
                pending[method]='검사 대기 또는 아직 완료 기록 없음';continue
            k=run['env_ids'].index(env);t=run['trajectory']
            info=analyze(t,k,initial,env)
            begin=info['release_step'] if method=='fsm' else info['all_contact_step'] if method=='ours' else info['first_step']
            info.update(path_metrics(t,k,initial,env,begin,info['success_step']))
            selection=roll/f'seed{seed}/{method}/selection.json'
            row=None
            if selection.exists():
                row=next((c for c in json.loads(selection.read_text())['cases'] if c['env_id']==env),None)
            if row:
                info.update(row['metrics']);info['selection_passed']=row['passes']
            last=int(t['alive'][:,k].all(-1).nonzero()[-1])
            if method=='tokenhsi':
                p0=info.get('first_force_push_step',info['first_step'])+1
                p1=info.get('early_capture_step',info['phase_steps'][1])
            elif method=='fsm':p0=info['first_frozen_step']+3;p1=info['release_step']+3
            else:p0=info['arrivals'][early]+3;p1=info['all_contact_step']+3
            p0=max(0,min(p0,last-2));p1=max(p0+1,min(p1,last-1))
            p2=info['success_step']+10 if method=='ours' and info['success_step']>=0 else info['path_peak_step']
            p2=max(p1+1,min(max(p2,p1+15),last))
            info['phase_steps']=[p0,p1,p2];methods[method]=info
            name={'tokenhsi':'Token','fsm':'FSM','ours':'Ours'}[method]
            if method=='tokenhsi' and 'sideways_path_offset_m' in info:
                side='왼쪽' if info['sideways_path_side']=='left' else '오른쪽'
                labels.append(f'Token {side} 이탈 {info["sideways_path_offset_m"]:.2f}m')
            elif method=='tokenhsi' and 'left_push_before_b_m' in info:
                labels.append(f'Token 왼쪽 밀기 {info["left_push_before_b_m"]:.2f}m')
            else:
                labels.append(f'{name} 굽힘 {info["path_bow_m"]:.2f}m')
            if method=='fsm' and row is not None:labels.append('FSM 선별 통과' if row['passes'] else 'FSM 선별 미통과')
        if not methods:raise ValueError(f'No completed recording for seed {seed} env {env}')
        cases.append(dict(seed=seed,env_id=env,early=early,matches=False,methods=methods,
            label=' · '.join(labels),pending_methods=pending))
    review=base/'review';review.mkdir(exist_ok=True)
    ranking=review/'viewer_cases.json';ranking.write_text(json.dumps(dict(cases=cases),indent=2,default=lambda x:x.item())+'\n')
    output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name('.building_'+output.name)
    env=dict(os.environ,COMPARE_SOURCE_ROOT=str(source),COMPARE_OUTPUT_ROOT=str(roll))
    command=['python',str(ROOT/'tokenhsi/scripts/analysis/build_push_comparison_html.py'),
        '--allow-missing-methods','--ranking-file',str(ranking),'--cases',
        *[f'{s}:{e}' for s,e in requested],'--output',str(temporary)]
    subprocess.run(command,cwd=ROOT,env=env,check=True)
    temporary.replace(output)
    proof=json.loads(temporary.with_suffix('.verification.json').read_text());proof['html']=str(output)
    proof['available_methods_by_case']={f'{c["seed"]}:{c["env_id"]}':list(c['methods']) for c in cases}
    output.with_suffix('.verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    temporary.with_suffix('.verification.json').unlink()
    print('REVIEW READY',output,'cases',len(cases),flush=True);return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'output_etc/check/push_natural_prefilter_20261112')
    p.add_argument('--output',type=Path,default=ROOT/'output_etc/check/push_comparison_html_20261112/push_review.html')
    p.add_argument('--cases',nargs='+');p.add_argument('--seed',type=int);p.add_argument('--env',type=int)
    p.add_argument('--watch',action='store_true');a=p.parse_args();previous=None
    while True:
        signature=tuple(sorted((str(f),f.stat().st_mtime_ns) for f in (a.root/'rollouts').glob('seed*/*/verification.json')))
        if signature!=previous:
            build(a);previous=signature
        if not a.watch:return
        time.sleep(5)


if __name__=='__main__':main()
