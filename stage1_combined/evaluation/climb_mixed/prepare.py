"""Materialize an isolated eval backend using the validated metadata-goal harness."""
from pathlib import Path
import json,os,shutil
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
DEFAULT_WORKSPACE=ROOT/'runs/climb_mixed_eval_20261022'
DEFAULT_TEMPLATE=ROOT/'runs/table_metadata_climb_20261022/source'
DEFAULT_ASSETS=Path('/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test')

def prepare(workspace=DEFAULT_WORKSPACE,template=DEFAULT_TEMPLATE,asset_root=DEFAULT_ASSETS):
    workspace,template,asset_root=map(lambda x:Path(x).resolve(),(workspace,template,asset_root))
    source=workspace/'source'
    if source.exists():
        marker=source/'mixed_backend.json'
        if not marker.exists():raise ValueError('Refuse to overwrite an unknown backend: '+str(source))
        saved=json.loads(marker.read_text())
        if saved['source_template']!=str(template) or saved['asset_root']!=str(asset_root):
            raise ValueError('Existing backend has different inputs; select a fresh --workspace')
        for src,dest in [('probe.py','table_climb_probe.py'),('sampling.py','climb_mixed_sampling.py')]:
            if (HERE/src).read_bytes()!=(source/'tokenhsi/utils'/dest).read_bytes():
                raise ValueError('Backend code changed; select a fresh --workspace')
        return source
    if not template.is_dir():raise FileNotFoundError('Validated metadata evaluation template missing: '+str(template))
    shutil.copytree(template,source,symlinks=True,ignore=shutil.ignore_patterns('output','output_etc','__pycache__','.git','table_climb_assets'))
    # Preserve original category order and os.listdir model enumeration; persist it in catalog.
    original=ROOT/'TokenHSI/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml'
    import yaml
    cats=yaml.safe_load(original.read_text())['env']['climb']['objCategories']
    if cats!=['Box','Cabinet','Table_Square']:raise ValueError('Original category list changed')
    catalog=[];asset_dest=source/'climb_mixed_assets'
    for category in cats:
        names=os.listdir(asset_root/category)
        for name in names:
            src=asset_root/category/name
            if not src.is_dir():raise ValueError('Unexpected non-model entry: '+str(src))
            cfg=json.loads((src/'config.json').read_text())
            if cfg['center']!=[0,0,0] or cfg['tarSitPos'][:2]!=[0,0]:raise ValueError('Unsupported original object coordinates')
            dest=asset_dest/category/name
            shutil.copytree(src,dest,symlinks=False)
            catalog.append(dict(id=category+'/'+name,category=category,model_id=name,path=str(dest),source=str(src),bbox=cfg['bbox'],tarSitPos=cfg['tarSitPos']))
    (asset_dest/'catalog.json').write_text(json.dumps(catalog,indent=2))
    shutil.copy2(HERE/'probe.py',source/'tokenhsi/utils/table_climb_probe.py')
    shutil.copy2(HERE/'sampling.py',source/'tokenhsi/utils/climb_mixed_sampling.py')
    # Legacy internal table aliases are private to the copied backend; artifacts identify mixed objects.
    p=source/'tokenhsi/learning/multi_agent/single_match_eval.py';s=p.read_text()
    s=s.replace("'native_table_catalog':getattr(task,'_native_table_catalog',None),","'native_climb_catalog':task._native_table_catalog, 'sampling':task._climb_sampling,")
    s=s.replace("initial['table_model_ids']=cpu(task._native_table_ids)","initial['object_model_ids']=cpu(task._native_table_ids)")
    s=s.replace("results['per_table']=[{'id':entry['id']", "results['per_object']=[{'category':entry['category'],'id':entry['id']")
    needle="    (out/'results.json').write_text"
    index=s.index(needle)
    s=s[:index]+'''    results['category_counts']=task._climb_sampling['category_counts']
    results['per_category']=[]
    for category in task._climb_sampling['categories']:
        selected=np.array([task._native_table_catalog[int(i)]['category']==category for i in ids])
        denom=int(selected.sum())
        results['per_category'].append({'category':category,'num_envs':denom,'point_ever_count':int(ever[0,selected].sum()),'region_ever_count':int(ever[1,selected].sum())})
'''+s[index:]
    p.write_text(s)
    # Save the independent offline validator alongside the backend.
    old=template.parent/'rescore.py'
    shutil.copy2(old,workspace/'rescore.py')
    (source/'mixed_backend.json').write_text(json.dumps(dict(protocol='single_match_mixed_climb_metadata_v1',original_category_prior={c:1/3 for c in cats},source_template=str(template),asset_root=str(asset_root),model_count=len(catalog)),indent=2))
    return source

if __name__=='__main__':print(prepare())
