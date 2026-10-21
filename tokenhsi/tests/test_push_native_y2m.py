"""Y2m resume must retain the approved pool and reject unrelated changes."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from isaacgym import gymapi  # before torch
import pytest
import torch
import yaml
from utils.open_push_spec import validate_env
from utils.push_scene import sample_random_at_scene,random_at_goal_valid


def env(name):
    return yaml.safe_load(Path(f'tokenhsi/data/cfg/multi_agent/{name}.yaml').read_text())['env']


def test_y2m_training_scene_and_pool_preserved():
    old,new=env('push_native_phase_xy'),env('push_native_phase_y2m')
    validate_env(old);validate_env(new)
    assert new['pushBox']['size_profiles']==[[round(1+i*.1,1),2.,1.3] for i in range(6)]
    assert new['relationReward']==old['relationReward']
    assert new['ampExpertMapping']==old['ampExpertMapping']
    before=json.loads(Path(old['openPushRSI']['bank_file']).read_text())
    after=json.loads(Path(new['openPushRSI']['bank_file']).read_text())
    assert before['candidates']==after['candidates']
    for seed in (42,20261021,351):
        xy,q,goals=sample_random_at_scene(2048,new,'cpu',torch.Generator().manual_seed(seed))
        assert random_at_goal_valid(xy,q,goals,new).all()


@pytest.mark.parametrize('size',[[1.5,2.1,1.3],[1.5,2.,1.4],[1.5,1.5,1.3]])
def test_y2m_profiles_reject_inconsistent_base(size):
    cfg=env('push_native_phase_y2m');cfg['pushBox']['size']=size
    with pytest.raises(ValueError):validate_env(cfg)


def test_resume_refuses_a_source_without_matching_contract(tmp_path):
    path=Path('tokenhsi/scripts/multi_agent/transfer_push_native_y2m_checkpoint.py')
    spec=importlib.util.spec_from_file_location('y2m_transfer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # A real matching reward graph cannot bypass a changed source geometry.
    from utils.relation_task_spec import checkpoint_metadata
    audit=json.loads(Path('output/push_native_phase_y2m_check/screen.json').read_text())
    source=tmp_path/'bad.pth';output=tmp_path/'target.pth'
    torch.save(dict(relation_metadata=checkpoint_metadata(env('push_native_phase_xy')['relationReward']),
        open_push_contract=deepcopy(audit['contract'])),source)
    with pytest.raises(ValueError,match='Source checkpoint'):
        module.transfer(source,output)
    assert not output.exists()
