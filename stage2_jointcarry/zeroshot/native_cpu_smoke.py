import json
import os
import subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root.parent/'runs/jointcarry_zeroshot/native_cpu_smoke_v2'
base.mkdir(exist_ok=True)
records=[]
for name in ['tokenhsi_armchair_native','teamhoi_large_rectangle']:
    output=base/name
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',ZERO_SHOT_POLICY_DEVICE='cpu',NUM_ENVS='1',EVAL_REPEATS='1',EPISODE_LENGTH='32',OUTPUT_PATH=str(output))
    with (base/(name+'_launch.log')).open('w') as log:
        result=subprocess.run(['bash',str(root/'zeroshot/native_cpu_test.sh'),name],env=env,stdout=log,stderr=subprocess.STDOUT)
    records.append(dict(asset=name,rc=result.returncode,output=str(output)))
    print(json.dumps(records[-1]),flush=True)
    (base/'status.json').write_text(json.dumps(records,indent=2)+'\n')
    if result.returncode:raise SystemExit(result.returncode)
