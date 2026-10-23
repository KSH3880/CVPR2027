import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils.impact_candidates import shortlist, ProbeCandidateReplay


@pytest.fixture
def files(tmp_path):
    source = tmp_path/'agents.jsonl'
    (tmp_path/'summary.json').write_text(json.dumps({'mode':'impact_sweep'}))
    bank = tmp_path/'bank.npz'
    bank.with_suffix('.json').write_text('{}')
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]/'runs') as out:
        yield source,Path(out)/'candidates.json',bank


def row(dv, phase=.9):
    return dict(skill='carryWith',motion_id=0,phase=phase,size_cm=[50,50,40],
                scene_finite=True,first_two_steps_delta_v=dv,round=0,env=0,agent=0)


def test_shortlist_tiers_and_exact_original_times(files):
    source,out,bank=files
    source.write_text('\n'.join(json.dumps(r) for r in [row(.9),row(1.5,.91),row(2.1,.92)]))
    p=shortlist(source,out,bank)
    assert [(r['phase'],r['tier']) for r in p['candidates']]==[(.9,'preferred'),(.91,'conditional')]
    assert p['stage']=='preliminary_not_training_safe'
    with pytest.raises(FileExistsError):
        shortlist(source,out,bank)


def test_no_nonfinite_or_outside_late_candidates(files):
    source,out,bank=files
    bad=row(.5,.91);bad['scene_finite']=False
    source.write_text('\n'.join(json.dumps(r) for r in [row(.5),bad,row(.5,.2),row(.5)]))
    assert len(shortlist(source,out,bank)['candidates'])==1


def test_missing_motion_ids_fail(files):
    source,out,bank=files
    r=row(.5);del r['motion_id'];source.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='motion IDs'):
        shortlist(source,out,bank)


def test_replay_preserves_size_skill_and_original_phase():
    class Lib:
        _motion_weights=torch.tensor([1.,1.])
        def sample_motions(self,n):return torch.zeros(n,dtype=torch.long)
        def sample_time_rsi(self,ids):return torch.full(ids.shape,.2)
        def get_motion_length(self,ids):return torch.full(ids.shape,4.)
    replay=ProbeCandidateReplay.__new__(ProbeCandidateReplay)
    replay.torch=torch
    replay.task=SimpleNamespace(_motion_lib={'carryWith':Lib()})
    candidate=dict(id=5,size_cm=[50,50,40],skill='carryWith',motion_id=1,phase=.9,tier='preferred')
    replay.groups={((50,50,40),'carryWith'):[candidate]}
    replay.stats={'candidate_draws':0,'missing_size_skill_draws':0}
    sizes=torch.tensor([[.5,.5,.4],[.65,.65,.55]])
    before=sizes.clone()
    ids,times,empty=replay.sample('carryWith',sizes)
    torch.testing.assert_close(sizes,before)
    assert ids.tolist()==[1,0]
    torch.testing.assert_close(times,torch.tensor([3.6,.2]))
    assert not empty.any()
    assert replay.stats=={'candidate_draws':1,'missing_size_skill_draws':1}
    assert replay.match([50,50,40],'carryWith',1,.9000001)==5
    assert replay.match([65,65,55],'carryWith',1,.9) is None
