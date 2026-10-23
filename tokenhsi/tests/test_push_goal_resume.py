"""Goal curriculum retains screened poses and rejects unrelated changes."""
from isaacgym import gymapi
from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from utils.push_goal_resume import environments, project_bank, migrate, TRAIN
from utils.push_only import training_contract
from utils.relation_task_spec import checkpoint_metadata
import yaml


def test_bank_preserves_candidates_and_goal_scene_bounds():
    source,target=environments()
    bank=project_bank(source,target)
    old=json.loads(Path(source['openPushRSI']['bank_file']).read_text())
    assert bank['candidates']==old['candidates']
    assert target['relationReward']==source['relationReward']
    from utils.open_push_scene import sample_boxes,sample_goal_offsets
    torch.manual_seed(42)
    boxes,_=sample_boxes(2048,'cpu',target['pushSceneRandomization'],target['pushBox']['size'])
    offsets=sample_goal_offsets(boxes,target['pushGoalRandomization'])
    assert ((boxes[:,0,:2]-boxes[:,1,:2]).norm(dim=-1)>=12).all()
    distance=offsets[...,:2].norm(dim=-1)
    assert distance.min()>=3 and distance.max()<=5
    assert abs(distance.mean().item()-4)<.05
    bad=deepcopy(target);bad['pushBox']['mass']+=1
    with pytest.raises(ValueError,match='screen settings'):
        project_bank(source,bad)


def test_full_state_preserved_and_wrong_source_rejected():
    source,target=environments()
    snapshot=dict(env=deepcopy(source),train=yaml.safe_load(TRAIN.read_text()),
        num_envs=2048,num_agents=2,num_objects=4,observation_size=570)
    weights=dict(model={'w':torch.randn(3)},optimizer={'state':{'step':123}},epoch=8500,frame=123456,
        running_mean_std={'mean':torch.randn(3)},reward_mean_std={},amp_input_mean_std={},
        amp_enabled=True,relation_experiment_config=snapshot,
        open_push_contract=training_contract(source,TRAIN),
        relation_metadata=checkpoint_metadata(source['relationReward']))
    result=migrate(weights,source,target)
    for key,value in weights.items():
        if key not in ('open_push_contract','relation_experiment_config'):
            assert result[key] is value
    assert weights['relation_experiment_config']['env']==source
    assert result['relation_experiment_config']['env']['pushGoalRandomization']==target['pushGoalRandomization']
    bad=deepcopy(weights);bad['open_push_contract']['reward']['push']['contact_mode']='goal_face'
    with pytest.raises(ValueError,match='original fixed-face'):
        migrate(bad,source,target)
