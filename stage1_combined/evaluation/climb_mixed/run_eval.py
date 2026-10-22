"""Run paired frozen policies on TokenHSI's native mixed CLIMB distribution."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import numpy as np
from prepare import prepare,DEFAULT_WORKSPACE,DEFAULT_TEMPLATE,DEFAULT_ASSETS

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=DEFAULT_WORKSPACE)
    parser.add_argument('--source-template',type=Path,default=DEFAULT_TEMPLATE)
    parser.add_argument('--asset-root',type=Path,default=DEFAULT_ASSETS)
    parser.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parents[2]/'checkpoints/Stage1V2Fixed_00073000.pth')
    parser.add_argument('--smoke',action='store_true',help='16 env x seed0; implementation check, not a performance result')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    source=prepare(args.workspace,args.source_template,args.asset_root)
    if args.prepare_only:print(source);return
    checkpoint=args.checkpoint.resolve()
    if not checkpoint.is_file():parser.error('Checkpoint missing: '+str(checkpoint))
    sys.path.insert(0,str(source/'tokenhsi'));sys.path.insert(0,str(source.parent))
    from utils.single_match_metrics import aggregate
    from rescore import validate
    count=16 if args.smoke else 512;seeds=[0] if args.smoke else [0,1,2]
    result=source.parent/('eval_'+time.strftime('%Y%m%d_%H%M%S')+('_smoke' if args.smoke else ''))
    result.mkdir()  # Refuse any existing output tag, including simultaneous launch collisions.
    manifest=dict(protocol='single_match_mixed_climb_metadata_v1',checkpoint=str(checkpoint),num_envs=count,seeds=seeds,action='CLIMB',gpu=5,smoke=args.smoke,categories={'Box':1/3,'Cabinet':1/3,'Table_Square':1/3},model_sampling='uniform within category, replacement',target='object_root_z + tarSitPos.z + 0.94',sit_changed=False)
    (result/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    summary={};checks=[]
    for policy in ['ours','tokenhsi']:
        records=[]
        for seed in seeds:
            out='output_etc/check/'+result.name+'/'+policy+'/seed_'+str(seed)
            env=os.environ.copy()
            for key in ['CHAIR_NATIVE_TEST','CHAIR_ONTOP','CHAIR_MESH','CLIMB_TABLE_BOX_CONTROL','EPISODE_LENGTH','STAGE1_V2_FIXED_ENV_CONFIG','TASK_GRAPH','CHECK','MAX_ITERATIONS','PUBLIC_TOKENHSI_POLICY']:
                env.pop(key,None)
            env.update(TOKENHSI_GPU='5',TOKENHSI_CONDA_ENV='tokenhsi_jhh',CLIMB_TABLE_TEST='1',PUBLIC_TOKENHSI_POLICY='1' if policy=='tokenhsi' else '0',OUTPUT_PATH=out,PYTHONDONTWRITEBYTECODE='1')
            print('START',policy,seed,'envs',count,flush=True)
            with (result/(policy+'_seed'+str(seed)+'.log')).open('w') as log:
                rc=subprocess.run(['bash',str(source/'tokenhsi/scripts/multi_agent/single_match_test.sh'),str(checkpoint),'CLIMB',str(seed),str(count)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            if rc:
                (result/'failed.json').write_text(json.dumps(dict(policy=policy,seed=seed,rc=rc)))
                raise SystemExit('Failed evaluation: '+str(result))
            folder=source/out;validate(folder)
            if policy=='tokenhsi':
                ref=Path(next(x['run'] for x in summary['ours']['records'] if x['seed']==seed))
                with np.load(folder/'initial_0.npz') as a,np.load(ref/'initial_0.npz') as b:
                    assert set(a.files)==set(b.files)
                    for key in b.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
                checks.append(dict(seed=seed,all_initial_arrays_exact=True))
                (result/'matched_initial_validation.json').write_text(json.dumps(checks,indent=2))
            record=json.loads((folder/'results.json').read_text());record['run']=str(folder);records.append(record)
            print('DONE',policy,seed,'counts',record['category_counts'],record['point_ever'],record['region_ever'],flush=True)
        summary[policy]={'records':records}
        if not args.smoke:
            summary[policy]['statistics']=aggregate(records,count)
            summary[policy]['category_statistics']={}
            for category in manifest['categories']:
                entries=[next(c for c in rec['per_category'] if c['category']==category) for rec in records]
                category_stats={}
                for metric in ['point_ever','region_ever']:
                    values=[100*c[metric+'_count']/c['num_envs'] if c['num_envs'] else None for c in entries]
                    category_stats[metric]={'seed_percent':values,'mean_percent':float(np.mean(values)) if None not in values else None,'sample_std_percentage_points':float(np.std(values,ddof=1)) if None not in values else None}
                summary[policy]['category_statistics'][category]=category_stats
        (result/'summary.json').write_text(json.dumps(summary,indent=2))
    if not args.smoke:
        lines=['# Mixed CLIMB metadata evaluation','', 'Box/Cabinet/Table_Square prior = 1/3 each; models uniform within category, replacement.','', '| Policy | Point ever % | Region ever % |','|---|---:|---:|']
        for policy in ['tokenhsi','ours']:
            stat=summary[policy]['statistics']
            lines.append('| {} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} |'.format(policy,stat['point_ever']['mean_percent'],stat['point_ever']['sample_std_percentage_points'],stat['region_ever']['mean_percent'],stat['region_ever']['sample_std_percentage_points']))
        lines+=['','512 environments x independent seeds0/1/2; mean ± sample standard deviation. First episode 30s ever, pelvis point <=20cm; region inner80% XY and z±20cm. Metadata target = object_z + tarSitPos.z + .94. SIT unchanged.','', 'Realized category counts and per-category statistics: summary.json. Matched initial scene/DOF arrays: matched_initial_validation.json. This is the common evaluation protocol with the original object prior, not a full reproduction of every paper benchmark setting.']
        (result/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (result/'completed.json').write_text(json.dumps(dict(rc=0,smoke=args.smoke,all_rescored=True,matched_initial_arrays_exact=True),indent=2))
    print('RESULT',result,flush=True)

if __name__=='__main__':main()
