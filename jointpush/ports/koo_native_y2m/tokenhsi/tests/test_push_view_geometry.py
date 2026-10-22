"""Viewer geometry must leave source contracts intact and protect training/RSI."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from isaacgym import gymapi  # before torch
import pytest
import torch
import yaml
from utils.push_view_geometry import viewer_geometry
from utils.push_scene import validate_scene, sample_random_at_scene, random_at_goal_valid


def setup():
    env=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/push_native_phase_xy.yaml').read_text())['env']
    env['numEnvs']=1
    args=SimpleNamespace(test=True,eval=True,headless=False,eval_skills='loco')
    variables={'TOKENHSI_VIEWER_PUSH_LENGTH_Y':'2','TOKENHSI_GUI_GPU_MODE':'native'}
    return env,args,variables


def test_actual_assets_use_two_metre_y_without_changing_source():
    env,args,variables=setup();before=deepcopy(env)
    result=viewer_geometry(env,args,variables)
    assert env==before
    assert result['pushBox']['mass']==20 and result['pushBox']['friction']==.6
    assert result['pushBox']['size']==[1.5,2.,1.3]
    assert {tuple(s) for s in result['pushBox']['size_profiles']}=={
        (round(1+i*.1,1),2.,1.3) for i in range(6)}
    validate_scene(result)
    # Full training-scale layout sample on CPU: larger footprint and joint paths.
    xy,q,goals=sample_random_at_scene(2048,result,'cpu',torch.Generator().manual_seed(42))
    assert random_at_goal_valid(xy,q,goals,result).all()


@pytest.mark.parametrize('field,value',[('test',False),('eval',False),('headless',True),('eval_skills','push')])
def test_reject_non_viewer_modes(field,value):
    env,args,variables=setup();setattr(args,field,value)
    with pytest.raises(ValueError):viewer_geometry(env,args,variables)


@pytest.mark.parametrize('value',['nan','inf','0','1.5','4'])
def test_reject_invalid_lengths(value):
    env,args,variables=setup();variables['TOKENHSI_VIEWER_PUSH_LENGTH_Y']=value
    with pytest.raises(ValueError):viewer_geometry(env,args,variables)


def test_reject_training_rsi_and_mps_viewer():
    env,args,variables=setup();variables['TOKENHSI_EVAL_TRAIN_RSI']='1'
    with pytest.raises(ValueError):viewer_geometry(env,args,variables)
    variables.pop('TOKENHSI_EVAL_TRAIN_RSI');variables['TOKENHSI_GUI_GPU_MODE']='mesa'
    with pytest.raises(ValueError):viewer_geometry(env,args,variables)


def test_no_override_preserves_existing_calls():
    env,args,_=setup()
    assert viewer_geometry(env,args,{}) is env
