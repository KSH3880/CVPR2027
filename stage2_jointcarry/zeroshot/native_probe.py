"""Small serial zero-shot probe; no training and no viewer."""
import json
import os
import subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root.parent/'runs/jointcarry_zeroshot/native_probe_20261015'
base.mkdir(exist_ok=True)
records=[]
for name in ['tokenhsi_table_native','tokenhsi_armchair_native','teamhoi_large_rectangle']:
    output=base/name
    if output.exists():raise RuntimeError('Output already exists: '+str(output))
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',ZERO_SHOT_POLICY_DEVICE='cuda:0',NUM_ENVS='8',EVAL_REPEATS='1',EPISODE_LENGTH='600',JOINT_ZERO_SHOT_CAPTURE='1',HEADLESS='1',OUTPUT_PATH=str(output))
    with (base/(name+'_launch.log')).open('w') as log:
        result=subprocess.run(['bash',str(root/'zeroshot/native_cpu_test.sh'),name],env=env,stdout=log,stderr=subprocess.STDOUT)
    records.append(dict(asset=name,rc=result.returncode,output=str(output)))
    print(json.dumps(records[-1]),flush=True)
    (base/'status.json').write_text(json.dumps(records,indent=2)+'\n')
    if result.returncode:raise SystemExit(result.returncode)
