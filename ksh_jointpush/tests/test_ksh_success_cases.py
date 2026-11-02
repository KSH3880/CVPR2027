import json
import numpy as np
import pytest

from ksh_jointpush.ksh_success_cases import first_hits
from ksh_jointpush.scripts.ksh_success_replay import load_cases


def test_first_hit_uses_official_arrival_even_if_lost_before_episode_end():
    flags = np.array([[False, False, True, False],
                      [False, True, False, False],
                      [True, False, True, False],
                      [False, False, False, False]])
    assert first_hits(flags).tolist() == [2, 1, -1, -1]


def test_one_agent_success_does_not_require_every_agent():
    flags = np.zeros((5, 4), bool)
    flags[3, 2] = True
    assert first_hits(flags).tolist() == [-1, -1, 3, -1]
    with pytest.raises(ValueError):
        first_hits(np.zeros((4,), bool))


def test_loader_rejects_mislabeled_and_nonfinite_recordings(tmp_path):
    flags = np.zeros((3, 4), bool)
    flags[1, 0] = True
    file = tmp_path / 'case.npz'
    arrays = dict(success=flags, roots=np.zeros((3, 28, 13)),
                  dofs=np.zeros((3, 4, 32, 2)), bodies=np.zeros((3, 4, 15, 3)),
                  targets=np.zeros((4, 3)), box_sizes=np.ones((24, 3)))
    np.savez(file, **arrays)
    meta = dict(file='old/cwd/case.npz', first_hit_frames=[1, -1, -1, -1])
    (tmp_path / 'cases.json').write_text(json.dumps(dict(cases=[meta])))
    assert len(load_cases(tmp_path)) == 1
    arrays['roots'][0, 0, 0] = np.nan
    np.savez(file, **arrays)
    with pytest.raises(ValueError, match='Nonfinite'):
        load_cases(tmp_path)
    arrays['roots'][0, 0, 0] = 0
    arrays['success'][:] = False
    np.savez(file, **arrays)
    with pytest.raises(ValueError, match='no verified later arrival'):
        load_cases(tmp_path)
