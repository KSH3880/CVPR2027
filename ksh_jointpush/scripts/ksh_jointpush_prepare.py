"""Prepare an isolated KSH jointpush workspace; external sources stay read-only."""
import argparse,hashlib,json,os,sys,shutil
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'ksh_jointpush'
WORK=ROOT/'runs/jointpush/ksh_jointpush_rsi/workspace_push_solo45_coupled0'

def restore_inputs(manifest, data_root):
    """Import missing immutable metadata and link payloads from the local download.

    The original manifest remains a provenance record, never a destination or
    a machine-specific runtime path. Check every input before making changes.
    """
    pending=[]
    missing=[]
    for name,digest in manifest['files'].items():
        local=SOURCE/name
        candidate=local
        if not local.is_file():
            relative=Path(name)
            if relative.parts[:2] != ('tokenhsi','data'):
                raise FileNotFoundError('Immutable source missing: '+str(local))
            candidate=data_root/relative.relative_to('tokenhsi/data')
            if not candidate.is_file():
                missing.append(str(candidate))
                continue
            pending.append((candidate,local))
        if hashlib.sha256(candidate.read_bytes()).hexdigest()!=digest:
            raise ValueError('Snapshot hash differs: '+str(candidate))
    links=[]
    for name in manifest['data_links']:
        local=SOURCE/name
        if local.exists():
            continue
        target=data_root/Path(name).relative_to('tokenhsi/data')
        if not target.is_dir():
            missing.append(str(target))
        else:
            links.append((local,target))
    if missing:
        raise FileNotFoundError('Missing local KSH data ('+str(len(missing))+'):\n'
            +'\n'.join(missing)+'\nDownload the original tokenhsi/data/. to '+str(data_root)
            +' (see ksh_jointpush/EXECUTION.md).')
    for candidate,local in pending:
        local.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(candidate,local)
    for local,target in links:
        local.parent.mkdir(parents=True,exist_ok=True)
        if local.is_symlink():
            local.unlink()  # Replace only a broken dataset link in this checkout.
        local.symlink_to(target,target_is_directory=True)


def resolve_motion_path(value, data_root):
    path=Path(value)
    # The supplied manifest also contains paths relative to the previous
    # checkout, e.g. ../../../../CVPR2027/TokenHSI/tokenhsi/data/....
    marker='tokenhsi/data/'
    if marker in str(path):
        path=Path(str(path).rsplit(marker,1)[1])
    elif path.is_absolute():
        if path.is_file():
            return str(path.resolve())
        raise FileNotFoundError('Cannot relocate motion path: '+str(path))
    for candidate in (SOURCE/'tokenhsi/data'/path,data_root/path):
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError('Motion payload missing: '+str(data_root/path))


def prepare(rsi_mode='balanced', gate_mode='soft', checkpoint_path=None, init_safety='legacy', distractors='legacy', data_root=None, contact_mode='native', reference_configs=None):
    data_root=Path(data_root or os.environ.get('KSH_DATA_ROOT',ROOT/'runs/assets/ksh_jointpush/data')).expanduser().resolve()
    if distractors not in ('legacy','clear_paths'):
        raise ValueError('distractors must be legacy or clear_paths')
    if init_safety not in ('legacy','all_pairs'):
        raise ValueError('init_safety must be legacy or all_pairs')
    if gate_mode not in ('soft','hard'):
        raise ValueError('gate_mode must be soft or hard')
    if rsi_mode not in ('group','agent','balanced'):
        raise ValueError('rsi_mode must be group, agent or balanced')
    suffix={'group':'','agent':'_agent_rsi','balanced':'_balanced_rsi'}[rsi_mode]
    work = WORK.with_name(WORK.name + suffix + ('_hard_gate' if gate_mode == 'hard' else ''))
    custom_checkpoint=Path(checkpoint_path or SOURCE/'pth/Stage1PushFinetune_00019500.pth').expanduser().resolve(strict=True)
    if custom_checkpoint:
        digest=hashlib.sha256(custom_checkpoint.read_bytes()).hexdigest()
        work=work.with_name(work.name+'_base_'+custom_checkpoint.stem+'_'+digest[:8])
    if init_safety == 'all_pairs':
        work=work.with_name(work.name+'_safe_start')
    if distractors == 'clear_paths':
        work=work.with_name(work.name+'_clear_paths')
    if contact_mode not in ('native', 'legacy_jhh'):
        raise ValueError('Unknown CONTACT mode')
    if contact_mode == 'legacy_jhh':
        work=work.with_name(work.name+'_coupled_jhh_solo_ksh_contact')
    if contact_mode == 'legacy_jhh':
        if reference_configs is None:
            raise ValueError('Legacy CONTACT requires --reference-configs from the comparison run')
        reference=Path(reference_configs).resolve(strict=True)
        env=yaml.safe_load((reference/'env.yaml').read_text())
        actual=Path(env['experiment']['transfer']['checkpoint']).resolve(strict=True)
        if hashlib.sha256(actual.read_bytes()).hexdigest()!=digest:
            raise ValueError('Reference Stage1 checkpoint differs')
        stage2=env['env']['stage2JointPush']
        if 'contact_reward' in stage2:
            raise ValueError('Reference already overrides CONTACT')
        if (stage2['reward_gate']['mode']!='all_hands_face_tolerance_v1'
                or rsi_mode!='balanced' or gate_mode!='hard'
                or init_safety!='all_pairs' or distractors!='clear_paths'):
            raise ValueError('Legacy comparison requires current hard/balanced/safe/clear reference')
        stage2['contact_reward']=dict(mode='coupled_jhh_solo_ksh_contact_v1',scope='coupled_only',contact_height=1.04,
            midpoint_only=True,approach_stand_off=1.,clearance=.45,
            implementation_sha256=hashlib.sha256((SOURCE/'ksh_legacy_contact.py').read_bytes()).hexdigest())
        configs=work/'configs';configs.mkdir(parents=True,exist_ok=True)
        # Keep the reference motion_file path, preserving every other config value.
        contents={'env.yaml':yaml.safe_dump(env,sort_keys=False),
                  'train.yaml':(reference/'train.yaml').read_text(),
                  'motions.yaml':(reference/'motions.yaml').read_text()}
        for name,content in contents.items():
            target=configs/name
            if target.exists() and target.read_text()!=content:
                raise ValueError('Workspace config differs: '+str(target))
            target.write_text(content)
        report=dict(workspace=str(work),reference_configs=str(reference),
                    changed_fields=['env.stage2JointPush.contact_reward'],
                    training_started=False,gpu_used=False)
        (work/'preparation.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        return
    manifest=json.loads((ROOT/'ksh_jointpush/source_manifest.json').read_text())
    restore_inputs(manifest,data_root)
    # Output storage is local to this project; generated configs remain under runs/.
    for name in ('output','output_etc'):
        (SOURCE/name).mkdir(parents=True,exist_ok=True)
    broken=[name for name in manifest['data_links'] if not (SOURCE/name).exists()]
    if broken:raise ValueError('Missing dataset links: '+str(broken))
    checkpoint=custom_checkpoint
    if checkpoint.name in manifest['checkpoints'] and hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=manifest['checkpoints'][checkpoint.name]:raise ValueError('Checkpoint changed')
    work.mkdir(parents=True,exist_ok=True);(work/'configs').mkdir(exist_ok=True)
    env=yaml.safe_load((SOURCE/'tokenhsi/data/cfg/multi_agent/stage2_jointpush_rsi.yaml').read_text())
    env['experiment']['transfer']['checkpoint']=str(checkpoint)
    cone=dict(goal_half_angle_degrees=0.,solo_goal_half_angle_degrees=45.,goal_cone_implementation_sha256=hashlib.sha256((ROOT/'ksh_jointpush/ksh_goal_cone.py').read_bytes()).hexdigest())
    env['env']['jointPush']['scene'].update(cone)
    env['env']['stage2JointPush']['goal_half_angle_degrees']=0.
    env['env']['stage2JointPush']['reward_gate']=dict(mode='mean_hand_face_quality_v1',sigma=.18,success='box_goal_position',implementation_sha256=hashlib.sha256((ROOT/'ksh_jointpush/ksh_soft_gate.py').read_bytes()).hexdigest())
    if gate_mode == 'hard':
        implementation=(ROOT/'ksh_jointpush/ksh_hard_gate.py').read_bytes()+(ROOT/'ksh_jointpush/ksh_soft_gate.py').read_bytes()
        env['env']['stage2JointPush']['reward_gate']=dict(mode='all_hands_face_tolerance_v1',face_tolerance=.18,success='box_goal_position',implementation_sha256=hashlib.sha256(implementation).hexdigest())
    env['env']['stage2JointPush']['task_selection']=dict(mode='push_only_random_groups_v1',implementation_sha256=hashlib.sha256((ROOT/'ksh_jointpush/ksh_push_only.py').read_bytes()).hexdigest())
    if rsi_mode == 'agent':
        env['env']['stage2JointPush']['coupled_rsi_selection']='independent_agents_v1'
        env['env']['stage2JointPush']['rsi_selection_implementation_sha256']=hashlib.sha256((ROOT/'ksh_jointpush/ksh_agent_rsi.py').read_bytes()).hexdigest()
    if rsi_mode == 'balanced':
        env['env']['stage2JointPush'].update(coupled_rsi_selection='balanced_group_ternary_v1',
            rsi_group_state_probabilities=dict(all_push=1/3,mixed=1/3,all_loco=1/3),
            rsi_solo_push_probability=.5,
            rsi_selection_implementation_sha256=hashlib.sha256((ROOT/'ksh_jointpush/ksh_balanced_rsi.py').read_bytes()).hexdigest())
    if init_safety == 'all_pairs':
        env['env']['stage2JointPush']['initialization_safety']=dict(mode='all_agent_pairs_v1',minimum_root_distance=.7,body_margin=.005,implementation_sha256=hashlib.sha256((ROOT/'ksh_jointpush/ksh_initialization_safety.py').read_bytes()).hexdigest())
    if distractors == 'clear_paths':
        env['env']['stage2JointPush']['distractor_placement']=dict(mode='far_clear_paths_v1',radius=[8.,12.],agent_path_clearance=1.5,push_path_clearance=1.5,implementation_sha256=hashlib.sha256((ROOT/'ksh_jointpush/ksh_distractor_clearance.py').read_bytes()).hexdigest())
    raw=yaml.safe_load((SOURCE/'tokenhsi/data/dataset_stage1_jointpush.yaml').read_text())
    for rows in raw['motions'].values():
        for row in rows:
            if not isinstance(row,dict):continue
            for key in ('file','obj_file'):
                if key not in row:continue
                row[key]=resolve_motion_path(row[key],data_root)
    motion_path=work/'configs/motions.yaml'
    expected_motion=yaml.safe_dump(raw,sort_keys=False)
    if motion_path.exists() and motion_path.read_text()!=expected_motion:raise ValueError('Workspace motion manifest differs')
    if not motion_path.exists():motion_path.write_text(expected_motion)
    env['env']['motion_file']=str(motion_path)
    stage1=yaml.safe_load((SOURCE/'tokenhsi/data/cfg/multi_agent/stage1_jointpush.yaml').read_text())
    stage1['env']['jointPush']['scene'].update(cone)
    stage1['env']['motion_file']=str(motion_path)
    (work/'configs/stage1_env.yaml').write_text(yaml.safe_dump(stage1,sort_keys=False))
    train=yaml.safe_load((SOURCE/'tokenhsi/data/cfg/train/rlg/stage2_jointpush_rsi.yaml').read_text())
    for name,value in [('env.yaml',env),('train.yaml',train)]:
        target=work/'configs'/name;expected=yaml.safe_dump(value,sort_keys=False)
        if target.exists() and target.read_text()!=expected:raise ValueError('Workspace config already differs: '+str(target))
        if not target.exists():target.write_text(expected)
    motions=yaml.safe_load((SOURCE/env['env']['motion_file']).read_text())['motions'];missing=[];count=0
    for rows in motions.values():
        for row in rows:
            if isinstance(row,dict) and 'file' in row:
                count+=1
                if not (SOURCE/'tokenhsi/data'/row['file']).exists():missing.append(row['file'])
    if missing:raise ValueError('Motion payloads missing: '+str(missing))
    report=dict(source=str(SOURCE),source_commit=manifest['commit'],workspace=str(work),data_root=str(data_root),checkpoint=str(checkpoint),source_epoch=int(checkpoint.stem.rsplit('_',1)[-1]),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),agents=4,objects=8,environments=1024,motion_entries=count,training_started=False,source_files_verified=len(manifest['files']),gpu_used=False)
    (work/'preparation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--rsi-mode',choices=['group','agent','balanced'],default='balanced')
    parser.add_argument('--gate-mode',choices=['soft','hard'],default='soft')
    parser.add_argument('--checkpoint')
    parser.add_argument('--init-safety',choices=['legacy','all_pairs'],default='legacy')
    parser.add_argument('--distractors',choices=['legacy','clear_paths'],default='legacy')
    parser.add_argument('--data-root',help='Local copy of the original tokenhsi/data directory (or KSH_DATA_ROOT)')
    parser.add_argument('--reference-configs')
    parser.add_argument('--contact-mode',choices=['native','legacy_jhh'],default='native')
    args=parser.parse_args()
    prepare(args.rsi_mode,args.gate_mode,args.checkpoint,args.init_safety,args.distractors,args.data_root,args.contact_mode,args.reference_configs)
