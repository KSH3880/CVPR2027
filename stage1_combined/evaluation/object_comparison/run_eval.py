"""Independent Box/Chair/Cabinet/Table x SIT/CLIMB paired policy evaluation."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import numpy as np
from prepare import prepare,DEFAULT_WORKSPACE
CATEGORIES=['Box','Chair','Cabinet','Table'];ACTIONS=['SIT','CLIMB']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=DEFAULT_WORKSPACE)
    parser.add_argument('--categories',nargs='+',choices=CATEGORIES,default=CATEGORIES)
    parser.add_argument('--actions',nargs='+',choices=ACTIONS,default=ACTIONS)
    parser.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parents[2]/'checkpoints/Stage1V2Fixed_00073000.pth')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='4 env x seed0, full900step, no performance statistics')
    args=parser.parse_args()
    if len(set(args.categories))!=len(args.categories) or len(set(args.actions))!=len(args.actions):parser.error('Duplicate cases')
    source=prepare(args.workspace)
    if args.prepare_only:print(source);return
    ck=args.checkpoint.resolve()
    if not ck.is_file():parser.error('Checkpoint missing: '+str(ck))
    sys.path.insert(0,str(source/'tokenhsi'));sys.path.insert(0,str(source.parent))
    from utils.single_match_metrics import aggregate
    from rescore import validate
    count=4 if args.smoke else 512;seeds=[0] if args.smoke else [0,1,2]
    result=source.parent/('eval_'+time.strftime('%Y%m%d_%H%M%S')+('_smoke' if args.smoke else ''));result.mkdir()
    (result/'run_manifest.json').write_text(json.dumps(dict(protocol='separate_object_sit_climb_v1',categories=args.categories,actions=args.actions,num_envs=count,seeds=seeds,gpu=5,smoke=args.smoke,box_sizes='original TokenHSI CLIMB test18',sampling='uniform replacement within each independent category',checkpoint=str(ck)),indent=2))
    summary={};checks=[]
    for category in args.categories:
        summary[category]={}
        for action in args.actions:
            summary[category][action]={}
            for policy in ['ours','tokenhsi']:
                records=[]
                for seed in seeds:
                    out='output_etc/check/'+result.name+'/'+category+'/'+action+'/'+policy+'/seed_'+str(seed)
                    env=os.environ.copy()
                    for key in ['CHAIR_ONTOP','CHAIR_MESH','CLIMB_TABLE_TEST','CLIMB_TABLE_BOX_CONTROL','EPISODE_LENGTH','STAGE1_V2_FIXED_ENV_CONFIG','TASK_GRAPH','CHECK','MAX_ITERATIONS','PUBLIC_TOKENHSI_POLICY']:
                        env.pop(key,None)
                    env.update(TOKENHSI_GPU='5',TOKENHSI_CONDA_ENV='tokenhsi_jhh',CHAIR_NATIVE_TEST='1',OBJECT_CATEGORY=category,PUBLIC_TOKENHSI_POLICY='1' if policy=='tokenhsi' else '0',OUTPUT_PATH=out,PYTHONDONTWRITEBYTECODE='1')
                    print('START',category,action,policy,seed,flush=True)
                    with (result/(category+'_'+action+'_'+policy+'_seed'+str(seed)+'.log')).open('w') as log:
                        rc=subprocess.run(['bash',str(source/'tokenhsi/scripts/multi_agent/single_match_test.sh'),str(ck),action,str(seed),str(count)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
                    if rc:
                        (result/'failed.json').write_text(json.dumps(dict(category=category,action=action,policy=policy,seed=seed,rc=rc)))
                        raise SystemExit('Failed: '+str(result))
                    folder=source/out;validate(folder)
                    if policy=='tokenhsi':
                        ref=Path(next(x['run'] for x in summary[category][action]['ours']['records'] if x['seed']==seed))
                        with np.load(folder/'initial_0.npz') as a,np.load(ref/'initial_0.npz') as b:
                            assert set(a.files)==set(b.files)
                            for key in b.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
                        checks.append(dict(category=category,action=action,seed=seed,initial_arrays_exact=True))
                        (result/'matched_initial_validation.json').write_text(json.dumps(checks,indent=2))
                    rec=json.loads((folder/'results.json').read_text());rec['run']=str(folder);records.append(rec)
                    print('DONE',category,action,policy,seed,rec['point_ever'],rec['region_ever'],flush=True)
                entry={'records':records}
                if not args.smoke:entry['statistics']=aggregate(records,count)
                summary[category][action][policy]=entry
                (result/'summary.json').write_text(json.dumps(summary,indent=2))
    if not args.smoke:
        lines=['# Separate native-object SIT/CLIMB evaluation','', '| Object | Task | TokenHSI point % | Our73k point % | TokenHSI region % | Our73k region % |','|---|---|---:|---:|---:|---:|']
        def cell(s,k):return '{:.2f} ± {:.2f}'.format(s[k]['mean_percent'],s[k]['sample_std_percentage_points'])
        for category in args.categories:
            for action in args.actions:
                x=summary[category][action];p=x['tokenhsi']['statistics'];o=x['ours']['statistics']
                lines.append('| {} | {} | {} | {} | {} | {} |'.format(category,action,cell(p,'point_ever'),cell(o,'point_ever'),cell(p,'region_ever'),cell(o,'region_ever')))
        lines+=['','Each object category is evaluated independently. Uniform replacement over original test models. Box uses original TokenHSI18 sizes. 512env x seeds0/1/2, mean±sample std; first-episode30s ever. Chair SIT/CLIMB use seat+.10/.94; other SIT uses bbox top+.10, other CLIMB metadata height+.94. Point pelvis/root distance≤.20; region inner80%XY and z±.20.','', 'Same initial scenes/DOF/object IDs between frozen public40k and our73k. Category-specific tasks include cross-object generalization cases absent from original benchmark.']
        (result/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (result/'completed.json').write_text(json.dumps(dict(rc=0,smoke=args.smoke,all_rescored=True,paired_initial_arrays_exact=True,cases=len(checks)),indent=2));print('RESULT',result,flush=True)

if __name__=='__main__':main()
