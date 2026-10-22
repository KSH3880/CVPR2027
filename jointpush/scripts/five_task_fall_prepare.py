"""Copy saved run settings; originals and checkpoints are read-only."""
import json,hashlib
from pathlib import Path
import yaml
SOURCE=Path('/home/hwanhee/koo_cvpr/stage1_fix/ports/ksh_stage1_stage2_base/push')
ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'runs/jointpush/five_task_500_fall_20261021/workspace'
WORK.mkdir(parents=True,exist_ok=True)
configs=WORK/'configs';configs.mkdir(exist_ok=True)
models=[('depth180','stage1_v2_fixed_five_task_depth180_20kg','Stage1FiveTaskDepth18020kg_21-16-18-36','Stage1FiveTaskDepth18020kg_00000500.pth'),('depth200_friction04','stage1_v2_fixed_five_task_depth200_20kg_friction04','Stage1FiveTaskDepth20020kgFriction04_21-16-18-50','Stage1FiveTaskDepth20020kgFriction04_00000500.pth')]
report={}
for tag,experiment,run,filename in models:
 path=SOURCE/'output'/experiment/run
 saved=yaml.safe_load((path/'relation_config.yaml').read_text())
 (configs/f'{tag}_env.yaml').write_text(yaml.safe_dump({k:saved[k] for k in ('env','sim','experiment')},sort_keys=False))
 (configs/f'{tag}_train.yaml').write_text(yaml.safe_dump(saved['train'],sort_keys=False))
 checkpoint=path/'nn'/filename
 report[tag]=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),saved_config=str(path/'relation_config.yaml'),saved_config_sha256=hashlib.sha256((path/'relation_config.yaml').read_bytes()).hexdigest(),pushBox=saved['env']['pushBox'],training_agents=saved['num_agents'],training_objects=saved['num_objects'],train_observation_width=saved['observation_size'],network=saved['train']['params']['network']['transformer'])
for name in ('tokenhsi','body_models','lpanlib'):
 target=WORK/name
 if not target.exists():target.symlink_to(SOURCE/name,target_is_directory=True)
manifest={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE/'tokenhsi').rglob('*.py') if '__pycache__' not in str(p)}
(WORK/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
(WORK/'preparation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
