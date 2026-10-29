from pathlib import Path
from types import SimpleNamespace
import json
import csv
import numpy as np
import pytest
from evaluation.hold_at.metrics import PROTOCOL, aggregate, classify, first_episode, validate


def test_at_uses_object_center_full_3d_and_inclusive_boundary():
    objects = np.zeros((4, 13)); goals = np.zeros((4, 3))
    objects[:, :3] = [[.2, 0, 0], [.15, .15, 0], [0, 0, .201], [0, 0, .19]]
    np.testing.assert_array_equal(classify(objects, goals), [True, False, False, True])


def test_terminal_success_counts_but_later_episode_success_does_not():
    point = [[False, False], [True, True], [False, True]]
    done = [[False, True], [True, False], [False, False]]
    active, ever, lengths = first_episode(point, done)
    np.testing.assert_array_equal(active, [[True, True], [True, False], [False, False]])
    np.testing.assert_array_equal(ever, [True, False])
    np.testing.assert_array_equal(lengths, [2, 1])


def test_incomplete_episode_is_rejected():
    with pytest.raises(ValueError, match='Incomplete'):
        first_episode([[True]], [[False]])


def test_seed_statistics_are_sample_std_and_require_independent_seeds():
    rows = [dict(protocol=PROTOCOL, seed=i, num_envs=512, point_ever_count=k, region_ever_count=k) for i,k in enumerate([0, 256, 512])]
    stats = aggregate(rows)
    assert stats['mean_percent'] == stats['sample_std_percentage_points'] == 50
    with pytest.raises(ValueError):
        aggregate(rows[:2])
    rows[1]['seed'] = 0
    with pytest.raises(ValueError):
        aggregate(rows)


def test_actual_rollout_adapter_never_resets_or_counts_initial_state(tmp_path, monkeypatch):
    import torch
    from evaluation.hold_at.backend import run_eval
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path/'checkpoint.pth'; checkpoint.write_bytes(b'test-only checkpoint hash')
    (tmp_path/'resolved_env.yaml').write_text('env: {}\n')
    task = SimpleNamespace(_is_eval=True, num_agents=1, num_objects=1, max_episode_length=900,
        dt=1/30, num_envs=2, _sampling_failures=0,
        cfg={'args':SimpleNamespace(output_path=str(tmp_path), checkpoint=str(checkpoint), seed=0), 'env':{'holdAtFT':{'asset_split':'test'}}},
        _humanoid_root_states=torch.zeros(2,1,13), _box_states=torch.zeros(2,1,13),
        _tar_pos=torch.tensor([[[4.,0,.5]],[[4.,0,.5]]]), _dof_pos=torch.zeros(2,1,32),
        _dof_vel=torch.zeros(2,1,32), _box_size=torch.ones(2,1,3),
        _native_up=torch.tensor([[0.,0.,1.],[0.,0.,1.]]),
        _object_model_ids=torch.tensor([0,1]), _env_origins=torch.zeros(2,3),
        _object_ground_z=torch.full((2,),.5), _object_front=torch.tensor([[0.,1,0]]).repeat(2,1))
    task._box_states[:,0,0]=1.5; task._box_states[:,0,2]=.55
    task._box_states[:,0,6]=1
    class Player:
        def __init__(self): self.env=SimpleNamespace(task=task);self.resets=0;self.steps=0
        def env_reset(self): self.resets+=1;return {'obs':torch.ones(2,280)}
        def get_batch_size(self,*args): pass
        def get_action(self,obs,deterministic): assert deterministic;return torch.zeros(2,32)
        def env_step(self,env,action):
            self.steps+=1
            done = torch.tensor([self.steps==2,self.steps==1])
            if self.steps==2:task._box_states[:,0,:3]=task._tar_pos[:,0]
            return {'obs':torch.ones(2,280)},torch.zeros(2),done,{'terminate':done}
    player=Player();result=run_eval(player)
    assert player.resets==1 and player.steps==2
    assert result['point_ever_count']==result['region_ever_count']==1
    assert result['tokenhsi_criteria']['success_count']==1
    rows=list(csv.DictReader((tmp_path/'episodes.csv').open()))
    assert [r['ours_30s_success'] for r in rows]==['1','0']
    assert [r['tokenhsi_20s_success'] for r in rows]==['1','0']
    assert validate(tmp_path)['ok']
    with (tmp_path/'results.json').open() as stream: record=json.load(stream)
    record['point_ever_count']=2
    (tmp_path/'results.json').write_text(json.dumps(record))
    with pytest.raises(AssertionError):validate(tmp_path)
