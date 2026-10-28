from pathlib import Path
import importlib.util
import json
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('native_table_validation',Path(__file__).with_name('validate.py'))
validation=importlib.util.module_from_spec(spec);spec.loader.exec_module(validation)


@pytest.fixture
def fixture(tmp_path):
    n=4;t=599
    objects=np.zeros((t,n,13),np.float32);objects[...,6]=1;objects[...,0]=1
    objects[:,0,0]=.1
    objects[:,1,0]=.1;objects[:,1,3]=.5;objects[:,1,6]=np.sqrt(.75)  # Near goal, tilted60degrees.
    objects[1:,2,0]=.1  # Success only after first episode already terminated.
    objects[-1,3,0]=.15  # Terminal-frame success must count.
    goals=np.zeros((t,n,3),np.float32)
    success=np.zeros((t,n),np.int64);success[:,0]=1;success[1:,2]=1;success[-1,3]=1
    precision=np.full((t,n),np.inf,np.float32);precision[:,0]=.1;precision[1:,2]=.1;precision[-1,3]=.15
    done=np.zeros((t,n),bool);done[0,2]=True;done[-1]=True
    terminate=np.zeros((t,n),bool);terminate[0,2]=True
    up=np.zeros((n,3),np.float32);up[:,2]=1
    metrics={}
    for repeat in range(3):
        np.savez(tmp_path/f'trace_{repeat}.npz',objects=objects,goals=goals,success=success,precision=precision,done=done,terminate=terminate)
        np.savez(tmp_path/f'initial_{repeat}.npz',up=up)
        metrics[f'ObjectSet_test_{repeat}']=dict(num_trials=n,success_trials=2,fail_trials_because_terminate=1,fail_trials_total=2,success_precision=.125)
    (tmp_path/'completed.json').write_text(json.dumps(dict(rc=0,num_envs=n,original_player_loop=True,native_environment=True,model_and_rms_frozen=True)))
    return tmp_path,metrics


def test_upright_terminal_frame_and_first_episode_only(fixture):
    path,metrics=fixture;result=validation.score(path,metrics)
    assert result['mean_percent']==50 and result['repeat_sample_std_pp']==0
    assert all(r['success_count']==2 and r['success_precision_cm']==pytest.approx(12.5) for r in result['repeats'])


def test_corrupt_native_counts_are_rejected(fixture):
    path,metrics=fixture;metrics['ObjectSet_test_0']['success_trials']=3
    with pytest.raises(AssertionError):validation.score(path,metrics)


def test_mutable_policy_is_rejected(fixture):
    path,metrics=fixture;p=path/'completed.json';value=json.loads(p.read_text());value['model_and_rms_frozen']=False;p.write_text(json.dumps(value))
    with pytest.raises(AssertionError):validation.score(path,metrics)
