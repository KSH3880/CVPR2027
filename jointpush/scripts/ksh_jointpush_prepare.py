"""Prepare an isolated KSH jointpush workspace; external sources stay read-only."""
import argparse,hashlib,json,sys,shutil
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'jointpush/ports/ksh_jointpush'
WORK=ROOT/'runs/jointpush/ksh_jointpush_rsi/workspace_push_only_aligned'

def prepare():
    manifest=json.loads((ROOT/'jointpush/ksh_jointpush_source_manifest.json').read_text())
    for name,digest in manifest['files'].items():
        local=SOURCE/name
        if not local.exists():
            original=Path(manifest['source'])/name
            if hashlib.sha256(original.read_bytes()).hexdigest()!=digest:raise ValueError('Cannot reconstruct changed source: '+name)
            local.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,local)
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Snapshot changed: '+name)
    for name,target in manifest['data_links'].items():
        link=SOURCE/name
        if not link.exists() and not link.is_symlink():
            link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(target,target_is_directory=Path(target).is_dir())
    run_root=ROOT/'runs/jointpush/ksh_jointpush_rsi'
    for name in ('output','output_etc'):
        (run_root/name).mkdir(parents=True,exist_ok=True)
        link=SOURCE/name
        if not link.exists() and not link.is_symlink():link.symlink_to(run_root/name,target_is_directory=True)
        if link.resolve()!=(run_root/name).resolve():raise ValueError('Runtime output location changed')
    checkpoint_dir=ROOT/'jointpush/checkpoints/ksh_jointpush';checkpoint_dir.mkdir(parents=True,exist_ok=True)
    if not (SOURCE/'pth').exists():(SOURCE/'pth').symlink_to(checkpoint_dir,target_is_directory=True)
    for name,digest in manifest['checkpoints'].items():
        target=checkpoint_dir/name
        if not target.exists():
            original=Path(manifest['source'])/'pth'/name
            if hashlib.sha256(original.read_bytes()).hexdigest()!=digest:raise ValueError('Checkpoint source changed: '+name)
            shutil.copy2(original,target)
    broken=[name for name in manifest['data_links'] if not (SOURCE/name).exists()]
    if broken:raise ValueError('Missing dataset links: '+str(broken))
    checkpoint=ROOT/'jointpush/checkpoints/ksh_jointpush/Stage1PushFinetune_00014000.pth'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=manifest['checkpoints'][checkpoint.name]:raise ValueError('Checkpoint changed')
    WORK.mkdir(parents=True,exist_ok=True);(WORK/'configs').mkdir(exist_ok=True)
    env=yaml.safe_load((SOURCE/'tokenhsi/data/cfg/multi_agent/stage2_jointpush_rsi.yaml').read_text())
    env['experiment']['transfer']['checkpoint']=str(checkpoint)
    cone=dict(goal_half_angle_degrees=0.,goal_cone_implementation_sha256=hashlib.sha256((ROOT/'jointpush/ksh_goal_cone.py').read_bytes()).hexdigest())
    env['env']['jointPush']['scene'].update(cone)
    env['env']['stage2JointPush']['goal_half_angle_degrees']=0.
    env['env']['stage2JointPush']['reward_gate']=dict(mode='mean_hand_face_quality_v1',sigma=.18,success='box_goal_position',implementation_sha256=hashlib.sha256((ROOT/'jointpush/ksh_soft_gate.py').read_bytes()).hexdigest())
    env['env']['stage2JointPush']['task_selection']=dict(mode='push_only_random_groups_v1',implementation_sha256=hashlib.sha256((ROOT/'jointpush/ksh_push_only.py').read_bytes()).hexdigest())
    raw=yaml.safe_load((SOURCE/'tokenhsi/data/dataset_stage1_jointpush.yaml').read_text())
    for rows in raw['motions'].values():
        for row in rows:
            if not isinstance(row,dict):continue
            for key in ('file','obj_file'):
                if key not in row:continue
                local=SOURCE/'tokenhsi/data'/row[key]
                original=Path(manifest['source'])/'tokenhsi/data'/row[key]
                row[key]=str((local if local.exists() else original).resolve(strict=True))
    motion_path=WORK/'configs/motions.yaml'
    expected_motion=yaml.safe_dump(raw,sort_keys=False)
    if motion_path.exists() and motion_path.read_text()!=expected_motion:raise ValueError('Workspace motion manifest differs')
    if not motion_path.exists():motion_path.write_text(expected_motion)
    env['env']['motion_file']=str(motion_path)
    stage1=yaml.safe_load((SOURCE/'tokenhsi/data/cfg/multi_agent/stage1_jointpush.yaml').read_text())
    stage1['env']['jointPush']['scene'].update(cone)
    stage1['env']['motion_file']=str(motion_path)
    (WORK/'configs/stage1_env.yaml').write_text(yaml.safe_dump(stage1,sort_keys=False))
    train=yaml.safe_load((SOURCE/'tokenhsi/data/cfg/train/rlg/stage2_jointpush_rsi.yaml').read_text())
    for name,value in [('env.yaml',env),('train.yaml',train)]:
        target=WORK/'configs'/name;expected=yaml.safe_dump(value,sort_keys=False)
        if target.exists() and target.read_text()!=expected:raise ValueError('Workspace config already differs: '+str(target))
        if not target.exists():target.write_text(expected)
    motions=yaml.safe_load((SOURCE/env['env']['motion_file']).read_text())['motions'];missing=[];count=0
    for rows in motions.values():
        for row in rows:
            if isinstance(row,dict) and 'file' in row:
                count+=1
                if not (SOURCE/'tokenhsi/data'/row['file']).exists():missing.append(row['file'])
    if missing:raise ValueError('Motion payloads missing: '+str(missing))
    report=dict(source=str(SOURCE),source_commit=manifest['commit'],workspace=str(WORK),checkpoint=str(checkpoint),source_epoch=14000,agents=4,objects=8,environments=1024,motion_entries=count,training_started=False,source_files_verified=len(manifest['files']),gpu_used=False)
    (WORK/'preparation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':prepare()
