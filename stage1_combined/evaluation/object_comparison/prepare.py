from pathlib import Path
import json,os,shutil,hashlib
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
DEFAULT_WORKSPACE=ROOT/'runs/object_comparison_20261022'
DEFAULT_TEMPLATE=ROOT/'runs/tokenhsi_native_objects_20261022/chair_source'

def prepare(workspace=DEFAULT_WORKSPACE):
    workspace=Path(workspace).resolve();source=workspace/'source'
    generator_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if source.exists():
        marker=source/'comparison_backend.json'
        if not marker.exists() or json.loads(marker.read_text()).get('generator_sha256')!=generator_hash or (source/'tokenhsi/utils/chair_native_probe.py').read_bytes()!=(HERE/'probe.py').read_bytes():raise ValueError('Select a fresh --workspace for changed or unknown backend')
        return source
    shutil.copytree(DEFAULT_TEMPLATE,source,symlinks=True,ignore=shutil.ignore_patterns('output','output_etc','__pycache__','.git','table_climb_assets'))
    dest=source/'comparison_assets';catalog={}
    chair_meta=json.loads((ROOT/'stage1_combined/chair_assets/catalog.json').read_text())['test']
    originals=Path('/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test')
    for category in ['Box','Chair','Cabinet','Table']:
        entries=[]
        paths=[Path(e['path']) for e in chair_meta] if category=='Chair' else [originals/({'Table':'Table_Square'}.get(category,category))/name for name in os.listdir(originals/({'Table':'Table_Square'}.get(category,category)))]
        for path in paths:
            cfg=json.loads((path/'config.json').read_text());folder=dest/category/path.name;shutil.copytree(path,folder,symlinks=False)
            if category=='Chair':
                old=next(e for e in chair_meta if e['id']==path.name)
                surface=old['seat'];sit_surface=surface;climb_surface=surface;extent=old['seat_extent']
            else:
                assert cfg['tarSitPos'][:2]==[0,0] and cfg['center']==[0,0,0]
                sit_surface=[0.,0.,cfg['bbox'][2]/2];climb_surface=cfg['tarSitPos'];extent=cfg['bbox'][:2]
            entries.append(dict(id=path.name,category=category,path=str(folder),source=str(path),bbox=cfg['bbox'],sit_surface=sit_surface,climb_surface=climb_surface,support_extent=extent,tarSitPos=cfg['tarSitPos']))
        catalog[category]=entries
    (dest/'catalog.json').write_text(json.dumps(catalog,indent=2))
    shutil.copy2(HERE/'probe.py',source/'tokenhsi/utils/chair_native_probe.py')
    p=source/'tokenhsi/learning/multi_agent/single_match_eval.py';s=p.read_text();start=s.index('    from native_reference.bridge import NativePolicy');end=s.index('    obs = player.env_reset()',start)
    s=s[:start]+'''    policy=None
    if os.environ.get('PUBLIC_TOKENHSI_POLICY')=='1':
        from native_reference.bridge import NativePolicy
        policy=NativePolicy(task,player)
        manifest.update(policy='public_tokenhsi_stage1_40000',policy_checkpoint=str(policy.checkpoint),policy_checkpoint_sha256=sha256(policy.checkpoint),native_observation_size=354)
    else:manifest['policy']='our_stage1_73000'
    manifest.update(category=task._native_category,sampling=task._object_sampling)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
'''+s[end:]
    s=s.replace('policy_action = policy.get_action()','policy_action = policy.get_action() if policy is not None else player.get_action(obs,True)')
    s=s.replace("'native_chair_catalog':task._native_chair_catalog","'native_object_catalog':task._native_chair_catalog")
    s=s.replace("results['per_chair']", "results['per_object']")
    s=s.replace("'target_surface':'seat'","'target_surface':'category_specific'")
    needle="    initial.update(chair_model_ids=cpu(task._native_chair_ids),chair_seats=seats,chair_extents=extents)"
    s=s.replace(needle,needle+'''
    from isaacgym.torch_utils import quat_rotate
    local=task._native_chair_seat[:,0]
    obj=task._box_states[:,0]
    expected=obj[:,:3]+quat_rotate(obj[:,3:7],local)
    expected[:,2]+=.1 if action=='SIT' else .94
    _,diag=task._evaluate_ontop_context()
    np.testing.assert_allclose(cpu(diag['target'])[:,0],cpu(expected),rtol=0,atol=2e-6)
    (out/'target_validation.json').write_text(json.dumps({'category':task._native_category,'action':action,'target_matches_surface':True}))
''')
    p.write_text(s)
    p=source/'tokenhsi/native_reference/bridge.py';s=p.read_text().replace("if hasattr(task,'_native_chair_catalog'):","if task._native_category=='Chair':")
    p.write_text(s)
    rescore=(ROOT/'runs/tokenhsi_native_objects_20261022/rescore.py').read_text().replace("root/'chair_source/tokenhsi'","root/'source/tokenhsi'")
    (workspace/'rescore.py').write_text(rescore)
    (source/'comparison_backend.json').write_text(json.dumps(dict(protocol='separate_object_sit_climb_v1',generator_sha256=generator_hash,template=str(DEFAULT_TEMPLATE),counts={k:len(v) for k,v in catalog.items()}),indent=2))
    return source

if __name__=='__main__':print(prepare())
