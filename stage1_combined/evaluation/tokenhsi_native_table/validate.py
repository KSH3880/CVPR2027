"""Independently re-score native success, precision and first-episode collection."""
from pathlib import Path
import json
import numpy as np


def score(output,metrics):
    output=Path(output);completion=json.loads((output/'completed.json').read_text());n=completion['num_envs']
    assert completion['rc']==0 and completion['original_player_loop'] and completion['native_environment'] and completion['model_and_rms_frozen']
    rows=[]
    for repeat in range(3):
        with np.load(output/f'trace_{repeat}.npz') as f:data={k:f[k] for k in f.files}
        with np.load(output/f'initial_{repeat}.npz') as f:initial={k:f[k] for k in f.files}
        assert np.isfinite(data['objects']).all() and np.isfinite(data['goals']).all()
        q=data['objects'][...,3:7].astype(np.float64);v=initial['up'][None].astype(np.float64)
        uv=np.cross(q[...,:3],v);rotated=v+2*(q[...,3:4]*uv+np.cross(q[...,:3],uv))
        dot=(rotated*v).sum(-1)
        distance=np.linalg.norm(data['objects'][...,:3].astype(np.float64)-data['goals'].astype(np.float64),axis=-1)
        hit=(distance<=.2)&(dot>=.86615)
        active=np.ones(n,bool);ever=np.zeros(n,bool);precision=np.full(n,np.inf);lengths=np.zeros(n,int);terminal=np.zeros(n,bool)
        for i in range(len(hit)):
            ever|=hit[i]&active;precision=np.minimum(precision,np.where(hit[i]&active,distance[i],np.inf))
            np.testing.assert_array_equal(data['success'][i,active]>0,ever[active])
            np.testing.assert_allclose(data['precision'][i,active],precision[active],rtol=1e-5,atol=1e-6)
            lengths+=active
            done=data['done'][i].astype(bool)&active;terminal[done]=data['terminate'][i,done].astype(bool)
            active&=~done
        assert not active.any() and (lengths[~terminal]==599).all()
        record=metrics[f'ObjectSet_test_{repeat}']
        assert record['num_trials']==n and record['success_trials']==int(ever.sum())
        assert record['fail_trials_because_terminate']==int((terminal&~ever).sum())
        assert record['fail_trials_total']==n-int(ever.sum())
        error=float(precision[ever].mean()) if ever.any() else 0.
        np.testing.assert_allclose(record['success_precision'],error,rtol=1e-5,atol=1e-6)
        rows.append(dict(repeat=repeat,success_count=int(ever.sum()),trials=n,success_percent=100*ever.mean(),success_precision_cm=error*100,
                         trace_rescored=True,time_limit_steps=599))
    values=np.array([r['success_percent'] for r in rows])
    return dict(validated=True,num_envs=n,seed=0,repeats=rows,
        mean_percent=float(values.mean()),repeat_sample_std_pp=float(values.std(ddof=1)),
        excluding_first_mean_percent=float(values[1:].mean()),excluding_first_repeat_std_pp=float(values[1:].std(ddof=1)),
        repeat_std_is_not_independent_seed_std=True)
