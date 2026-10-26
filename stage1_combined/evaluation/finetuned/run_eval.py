from pathlib import Path
import argparse,datetime,json,os,subprocess,hashlib,shutil
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parents[2]

def catalog(category):
    if category=='Mixed':
        result=[]
        for group in ['Box','Cabinet','Table']:
            entries=catalog(group)
            for e in entries:
                e['sampling_weight']=1/(3*len(entries))
                e['feature']=[1.,0.,0.,0.,0.,0.]  # Match the existing CLIMB policy inputs.
            result.extend(entries)
        return result
    if category=='Chair':
        entries=json.loads((ROOT/'chair_assets/catalog.json').read_text())['test']
        return [dict(id=e['id'],path=e['path'],bbox=e['bbox'],seat=e['seat'],extent=e['seat_extent'],feature=e['feature'],metadata_height=json.loads((Path(e['path'])/'config.json').read_text())['tarSitPos'][2]) for e in entries]
    folder=ROOT/'tokenhsi/data/dataset_amass_climb/objects/test'/('Table_Square' if category=='Table' else category)
    result=[]
    for path in sorted(folder.iterdir()):
        if not path.is_dir():continue
        d=json.loads((path/'config.json').read_text());seat=[0,0,d['bbox'][2]/2]
        result.append(dict(id=path.name,path=str(path.resolve()),bbox=d['bbox'],seat=seat,extent=d['bbox'][:2],feature=[1,0,0,*seat],metadata_height=d['tarSitPos'][2]))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--original30d',action='store_true');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--gpu',default='0');p.add_argument('--categories',nargs='+',choices=['Chair','Table','Box','Cabinet','Mixed'],default=['Chair','Table']);p.add_argument('--actions',nargs='+',choices=['SIT','CLIMB'],default=['SIT','CLIMB']);a=p.parse_args()
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    out=a.output or ROOT/'output_etc/check'/('finetuned_eval_'+stamp+('_smoke' if a.smoke else ''))
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    ck=a.checkpoint.resolve();shutil.copy2(ck,out/'checkpoint.pth');ck=out/'checkpoint.pth'
    weights=torch.load(ck,map_location='cpu',weights_only=False)
    if not a.original30d:assert weights['chair_ft_contract']['shape_encoder']=='bbox35_mlp'
    else:
        assert weights['model']['a2c_network.actor_encoder.tokenizers.1.0.weight'].shape[1]==30
        assert not any('.tokenizers.1.chair.' in k or '.tokenizers.1.base.' in k for k in weights['model'])
    count=4 if a.smoke else 512;seeds=[0] if a.smoke else [0,1,2]
    manifest=dict(checkpoint=str(a.checkpoint.resolve()),sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),epoch=weights['epoch'],num_envs=count,seeds=seeds,gpu=a.gpu,protocol='single_person_native_object_point_region_ever',scope='finetuned_only_no_baseline_comparison',max_steps=900)
    manifest['architecture']='original30d_no_expansion' if a.original30d else 'previous35d'
    manifest['code_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    manifest['core_goal_sha256']=hashlib.sha256((ROOT/'tokenhsi/env/tasks/multi_agent/humanoid_chair_ft.py').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    results={}
    for category in a.categories:
        entries=catalog(category);assert len(entries)==dict(Chair=26,Table=8,Box=18,Cabinet=9,Mixed=35)[category]
        for action in a.actions:
            records=[]
            for seed in seeds:
                case=out/category/action/str(seed);case.mkdir(parents=True)
                source=weights['relation_experiment_config'];cfg={'env':dict(source['env']),'sim':source['sim']}
                if a.original30d:cfg=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/tokenhsi_climb_30d_full_ft.yaml').read_text())
                cfg['env']['chairFT']=dict(cfg['env']['chairFT']);cfg['env']['chairFT']['point_evaluation']=dict(category=category,action=action,catalog=entries)
                cfg['env']['numAgents']=1;cfg['env']['numObjects']=1;cfg['env']['episodeLength']=900;cfg['env']['debug']=dict(reward=False)
                (case/'env.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
                cmd=['bash','-c','. tokenhsi/scripts/multi_agent/runtime_env.sh; exec python "$@"','evalcase','evaluation/finetuned/launch_case.py','--task','EvaluationTask','--cfg_train','tokenhsi/data/cfg/train/rlg/chair_table_eval_full_ft.yaml','--cfg_env',str(case/'env.yaml'),'--motion_file','tokenhsi/data/dataset_loco_sit_carry_climb.yaml','--checkpoint',str(ck),'--num_agents','1','--num_objects','1','--num_envs',str(count),'--task_graph',action.lower(),'--eval_skills','loco','--eval_skill_probs','1','--eval_repeats','1','--episode_length','900','--output_path',str(case),'--seed',str(seed),'--headless','--no_video','--test','--eval']
                if a.original30d:
                    cmd[cmd.index('tokenhsi/data/cfg/train/rlg/chair_table_eval_full_ft.yaml')]='tokenhsi/data/cfg/train/rlg/tokenhsi_climb_30d_full_ft.yaml'
                env=os.environ.copy();env['OURS_ORIGINAL30D']='1' if a.original30d else '0';env.update(TOKENHSI_GPU=a.gpu,TOKENHSI_CONDA_ENV='tokenhsi118',PYTORCH_JIT='0',PYTHONDONTWRITEBYTECODE='1')
                print('START',category,action,seed,flush=True)
                with (case/'run.log').open('wb') as log:rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
                if rc:raise RuntimeError(f'case failed rc={rc}: {case}')
                record=json.loads((case/'results.json').read_text());records.append(record);print('DONE',category,action,seed,record['point_ever'],record['region_ever'],flush=True)
            results[category+'_'+action]=dict(records=records,point_mean=100*np.mean([x['point_ever'] for x in records]),region_mean=100*np.mean([x['region_ever'] for x in records]),point_std=100*np.std([x['point_ever'] for x in records],ddof=1) if len(seeds)>1 else None,region_std=100*np.std([x['region_ever'] for x in records],ddof=1) if len(seeds)>1 else None)
            (out/'summary.json').write_text(json.dumps(results,indent=2))
    lines=['# Finetuned chair/table evaluation','',f'Checkpoint epoch: {weights["epoch"]}. {count} environments × seeds {seeds}.','', '| Object/task | Point ever % | Region ever % |','|---|---:|---:|']
    for name,d in results.items():
        def cell(k):return f'{d[k+"_mean"]:.2f} ± {d[k+"_std"]:.2f}' if len(seeds)>1 else f'{d[k+"_mean"]:.2f} (smoke)'
        lines.append(f'| {name} | {cell("point")} | {cell("region")} |')
    lines+=['','Single person/one fixed native object, loco start, root point distance≤20cm; region inner80%XY and target Z±20cm. First episode includes terminal frame. No feet/contact/hold condition. Chair seat+.10/.94, table bbox top+.10/metadata+.94. Test assets follow selected catalog; Mixed uses Box18/Cabinet9/Table8 with equal category probability. Three-seed sample standard deviation. No paired public-policy comparison; identical initial scenes with historical benchmark have not been established.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');(out/'completed.json').write_text(json.dumps(dict(rc=0,all_traces_rescored=True)));print('RESULT',out,flush=True)
if __name__=='__main__':main()
