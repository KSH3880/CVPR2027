"""Bounded seed workers; every seed gates initial -> TokenHSI -> FSM -> Ours."""
import fcntl
import json
import os
import shutil
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(os.environ.get('NATURAL_PUSH_ROOT',str(ROOT/'output_etc/check/push_natural_prefilter_20261112'))).resolve()
OUT.relative_to(ROOT/'output_etc/check')
WRAPPER=os.environ.get('NATURAL_PUSH_WRAPPER','tokenhsi/scripts/analysis/natural_push_search.sh')
METHODS=('tokenhsi','fsm','ours')
MAX_ROLLOUTS=int(os.environ.get('NATURAL_PUSH_WORKERS','4'))
CONTINUE_SEARCH=os.environ.get('NATURAL_PUSH_CONTINUE','0')=='1'
NEW_NUM_ENVS=int(os.environ.get('NATURAL_PUSH_NUM_ENVS','1024'))
CRITERIA_VERSION='fixed_sideways2_v2' if os.environ.get('PUSH_STAGE_PROFILE')=='sideways2' else 'fixed_left_push2_v1' if os.environ.get('PUSH_STAGE_PROFILE')=='left2' else 'path_curve_v2'


def active_rollouts():
    result=set()
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:
            args=(d/'cmdline').read_bytes().split(b'\0')
            if any(a.endswith((b'/search_push_wait.py',b'/push_comparison_rollout.py',b'/push_fixed_rollout.py')) for a in args):
                decoded=[a.decode() for a in args if a]
                if '--output_path' not in decoded:continue
                try:
                    Path(decoded[decoded.index('--output_path')+1]).resolve().relative_to(ROOT/'output_etc/check')
                except (ValueError,IndexError):continue
                result.add(int(d.name))
        except OSError:pass
    return result


def running_seed_jobs():
    jobs={}
    for pid in active_rollouts():
        try:
            args=[a.decode() for a in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if a]
            output=Path(args[args.index('--output_path')+1]).resolve()
            try:
                relative=output.relative_to(OUT/'rollouts')
                if len(relative.parts)==2:
                    jobs[int(relative.parts[0][4:])]=dict(pid=pid,method=relative.parts[1])
            except ValueError:
                relative=output.relative_to(OUT/'source')
                if len(relative.parts)==1:
                    jobs[int(relative.parts[0][4:])]=dict(pid=pid,method='initial')
        except (OSError,ValueError,IndexError):pass
    return jobs


def token_bend_priority(method,spec):
    if method=='tokenhsi':return 0.
    path=OUT/f'rollouts/seed{spec["seed"]}/tokenhsi/selection.json'
    if not path.exists():return 0.
    rows=json.loads(path.read_text())['cases']
    eligible=None
    if method=='ours':
        eligible=set(json.loads((OUT/f'rollouts/seed{spec["seed"]}/fsm/selection.json').read_text())['env_ids'])
    return max([r['metrics'].get('sideways_path_offset_m',r['metrics'].get('left_push_before_b_m',r['metrics'].get('path_bow_m',0.))) for r in rows if r['passes'] and (eligible is None or r['env_id'] in eligible)]+[0.])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def read_initial():
        result=[]
        for path in (OUT/'source').glob('seed*/initial_prefilter.json'):
            try:result.append(json.loads(path.read_text()))
            except (FileNotFoundError,json.JSONDecodeError):continue
        result.sort(key=lambda r:(-len(r['env_ids'])/r['num_envs'],r['seed']))
        return result
    first_seed=int(os.environ.get('NATURAL_PUSH_START_SEED','18'))
    next_seed=max([int(p.name[4:]) for p in (OUT/'source').glob('seed*') if p.name[4:].isdigit()]+[first_seed-1])+1
    env=dict(os.environ,TOKENHSI_GPU='7')
    children={};failed=[];verified=set();found=[]
    while True:
        for seed,(process,log,method) in list(children.items()):
            code=process.poll()
            if code is None:continue
            log.close();del children[seed]
            if code:failed.append(dict(seed=seed,method=method,exit_code=code))
        initial=read_initial()
        status_rows=[]
        ready=[]
        for spec in initial:
            seed=spec['seed'];current='no_initial_match';done=not spec['env_ids'];passed=len(spec['env_ids'])
            if any(f['seed']==seed for f in failed):
                status_rows.append(dict(seed=seed,state='failed'));continue
            if spec['env_ids']:
                expected=set(spec['env_ids'])
                for method in METHODS:
                    folder=OUT/f'rollouts/seed{seed}/{method}'
                    select_path=folder/'selection.json'
                    verification_path=folder/'verification.json'
                    if verification_path.exists():
                        actual=set(json.loads(verification_path.read_text())['env_ids'])
                        if not expected.issubset(actual):
                            if seed in children or seed in running_seed_jobs():
                                current=method;break
                            archived=OUT/f'archive/{CRITERIA_VERSION}/seed{seed}/{method}_{time.time_ns()}'
                            archived.parent.mkdir(parents=True,exist_ok=True)
                            shutil.move(str(folder),str(archived))
                            print('retained previous subset',seed,method,'rerun for',sorted(expected-actual),flush=True)
                    stale=select_path.exists() and json.loads(select_path.read_text()).get('criteria_version')!=CRITERIA_VERSION
                    if verification_path.exists() and (not select_path.exists() or stale):
                        if stale:
                            shutil.copy2(select_path,folder/'selection.previous_goal_error.json')
                        command=['bash','-c','source tokenhsi/scripts/multi_agent/runtime_env.sh && exec python tokenhsi/scripts/analysis/natural_push_stage_select.py --root "$1" --seed "$2" --method "$3"',
                            'natural-select',str(OUT),str(seed),method]
                        with (OUT/f'{method}_seed{seed}_select.log').open('w') as log:
                            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                        if result.returncode:
                            failed.append(dict(seed=seed,method=method+'_select',exit_code=result.returncode));current='failed';done=True;break
                    if select_path.exists():
                        selection=json.loads(select_path.read_text());passed=selection['passed']
                        if not passed:current=method+'_rejected';done=True;break
                        expected=set(selection['env_ids'])
                        if method=='ours':
                            current='complete_match';done=True
                            found.extend(dict(seed=seed,env_id=e) for e in selection['env_ids'])
                        continue
                    current=method
                    ready.append((method,spec))
                    break
            status_rows.append(dict(seed=seed,state=current,done=done,passed=passed))
        # Complete the same scene's baseline comparison before starting a new
        # seed's TokenHSI run. No barrier over all prefiltered scenes.
        manual_reviews=[]
        for request_path in sorted((OUT/'manual_requests').glob('*.json')):
            request=json.loads(request_path.read_text());seed=request['seed'];method=request['method']
            proof=OUT/f'rollouts/seed{seed}/{method}/verification.json'
            complete=proof.exists() and set(request['env_ids']).issubset(json.loads(proof.read_text())['env_ids'])
            state='complete' if complete else 'pending'
            if not complete and (seed in children or seed in running_seed_jobs()):
                state='running'
            elif not complete and len(active_rollouts() | {p.pid for p,_,_ in children.values()})<MAX_ROLLOUTS:
                log=(OUT/f'{method}_seed{seed}_manual_review.log').open('w')
                process=subprocess.Popen(['bash','tokenhsi/scripts/analysis/push_comparison_rollout.sh',method,str(seed)],
                    cwd=ROOT,env=dict(env,COMPARE_NUM_ENVS=str(request['num_envs']),
                        COMPARE_SOURCE_ROOT=str(OUT/'source'),COMPARE_OUTPUT_ROOT=str(OUT/'rollouts'),
                        COMPARE_SELECTION_FILE=str(request_path)),stdout=log,stderr=subprocess.STDOUT)
                children[seed]=(process,log,method);state='running'
                print('started explicit manual review',method,seed,request['env_ids'],process.pid,flush=True)
            manual_reviews.append(dict(seed=seed,method=method,env_ids=request['env_ids'],state=state))
        ready.sort(key=lambda item:(-METHODS.index(item[0]),-token_bend_priority(*item)))
        for method,spec in ready:
            seed=spec['seed']
            if seed in children or seed in running_seed_jobs() or found:
                continue
            if len(active_rollouts() | {p.pid for p,_,_ in children.values()})>=MAX_ROLLOUTS:
                break
            log=(OUT/f'{method}_seed{seed}.log').open('w')
            process=subprocess.Popen(['bash',WRAPPER,method,str(seed)],
                cwd=ROOT,env=dict(env,COMPARE_NUM_ENVS=str(spec['num_envs'])),stdout=log,stderr=subprocess.STDOUT)
            children[seed]=(process,log,method)
            print('started',method,seed,process.pid,'initial_filter_passed',len(spec['env_ids']),flush=True)
        # Only fill spare slots after all ready FSM/Ours/TokenHSI work has priority.
        # Fresh resets are inspected before any policy action is executed.
        while CONTINUE_SEARCH and not found and len(active_rollouts() | {p.pid for p,_,_ in children.values()})<MAX_ROLLOUTS:
            seed=next_seed;next_seed+=1
            if (OUT/f'source/seed{seed}').exists() or seed in running_seed_jobs():
                continue
            log=(OUT/f'initial_seed{seed}.log').open('w')
            process=subprocess.Popen(['bash',WRAPPER,'initial',str(seed)],
                cwd=ROOT,env=dict(env,COMPARE_NUM_ENVS=str(NEW_NUM_ENVS)),stdout=log,stderr=subprocess.STDOUT)
            children[seed]=(process,log,'initial')
            print('started natural initial-only reset',seed,process.pid,flush=True)
        found=list({(r['seed'],r['env_id']):r for r in found}.values())
        done=not children and all(r['state']=='complete' for r in manual_reviews) and (bool(found) or (not CONTINUE_SEARCH and all(r.get('done') or r['state']=='failed' for r in status_rows)))
        jobs=running_seed_jobs()
        jobs.update({s:dict(pid=p.pid,method=m) for s,(p,_,m) in children.items()})
        report=dict(pid=os.getpid(),state='complete' if done else 'running',gpu=7,max_rollouts=MAX_ROLLOUTS,
            criteria_version=CRITERIA_VERSION,
            candidate_priority='follow-up stages first, then larger TokenHSI path bow',
            continue_new_seeds=CONTINUE_SEARCH,next_seed=next_seed,new_seed_num_envs=NEW_NUM_ENVS,
            order=['initial_distance_and_facing','tokenhsi','fsm','ours'],initial_state_modified=CRITERIA_VERSION.startswith('fixed_'),
            initial_scenes=sum(r['num_envs'] for r in initial),prefilter_matches=sum(len(r['env_ids']) for r in initial),
            active={str(s):job for s,job in jobs.items()},
            manual_reviews=manual_reviews,
            seeds=status_rows,failures=failed,complete_cases=found)
        (OUT/'status.json').write_text(json.dumps(report,indent=2)+'\n')
        if done:
            print('finished',json.dumps(report),flush=True);return
        time.sleep(3)


if __name__=='__main__':main()
