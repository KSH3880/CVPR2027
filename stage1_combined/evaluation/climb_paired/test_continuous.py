import ast
from pathlib import Path
import textwrap
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from evaluation.climb_paired.native_loop import build_loop
from evaluation.climb_paired.repeat_metrics import bodies_after_reset,repeat_statistics,seed_statistics

BASELINE=Path(__file__).resolve().parents[3]/'TokenHSI/tokenhsi/learning/transformer/trans_players.py'


def test_repeat0_preserved_but_excluded_from_primary_statistics():
    records=[dict(repeat=i,point_ever=value,region_ever=value) for i,value in enumerate([0,.5,1])]
    stats=repeat_statistics(records)['region']
    assert stats['repeat_percent']==[0,50,100]
    assert stats['mean_percent']==75
    assert stats['within_seed_sample_std_percentage_points']==pytest.approx(35.3553390593)
    assert stats['all_three_mean_percent']==50
    assert stats['all_three_sample_std_percentage_points']==50
    with pytest.raises(AssertionError):repeat_statistics(records[::-1])


def test_repeat_scatter_is_not_between_seed_scatter():
    def result(values):return dict(repeat_statistics=repeat_statistics([dict(repeat=i,point_ever=value,region_ever=value) for i,value in enumerate(values)]))
    rows=[dict(seed=0,tokenhsi=result([0,1,1]),ours=result([0,.5,1])),
          dict(seed=1,tokenhsi=result([0,.5,.5]),ours=result([0,.5,.5]))]
    stats=seed_statistics(rows)['region']
    assert stats['ours']['mean_percent']==62.5
    assert stats['ours']['between_seed_sample_std_percentage_points']==pytest.approx(17.6776695297)
    assert stats['ours']['within_seed_sample_std_percentage_points']==pytest.approx([35.3553390593,0])
    assert stats['winning_seeds']==[] and stats['tied_seeds']==[1] and stats['losing_seeds']==[0]


def test_reset_body_selection_preserves_physics_views_and_other_rows():
    physical=[torch.ones((3,2,width))*i for i,width in enumerate([3,4,3,3])]
    before=[v.clone() for v in physical]
    kinematic=torch.arange(3*2*13).reshape(3,2,13).float()
    selected=bodies_after_reset(physical,kinematic,torch.tensor([1]))
    for new,old,section in zip(selected,before,[slice(0,3),slice(3,7),slice(7,10),slice(10,13)]):
        assert torch.equal(new[1],kinematic[1,...,section])
        assert torch.equal(new[[0,2]],old[[0,2]])
    for actual,old in zip(physical,before):assert torch.equal(actual,old)


def test_instrumented_native_loop_has_identical_resets_rng_actions_and_collection():
    text=BASELINE.read_text();cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef))
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run_eval')
    original=textwrap.dedent(''.join(text.splitlines(keepends=True)[fn.lineno-1:fn.end_lineno]))
    original=original[:original.index('    # save metrics')]+'    self.saved=eval_res\n    return\n'
    namespace=dict(torch=torch);exec(compile(original,str(BASELINE),'exec'),namespace)
    class Player:
        def __init__(self):self.env=SimpleNamespace(num_envs=2);self.num_agents=1;self.device='cpu';self.is_determenistic=True;self.progress=torch.zeros(2,dtype=torch.long);self.events=[]
        def env_reset(self,ids=None):
            ids=torch.arange(2) if ids is None else torch.as_tensor(ids,dtype=torch.long)
            self.events.append(('reset',ids.tolist(),torch.rand(len(ids)).tolist()))
            self.progress[ids]=0
            return {'obs':self.progress.clone()}
        def get_batch_size(self,*args):return 2
        def get_action(self,obs,deterministic):self.events.append(('action',obs['obs'].tolist()));return torch.zeros(2,1)
        def env_step(self,env,action):
            self.progress+=1;done=self.progress>=torch.tensor([2,4])
            self.events.append(('step',done.tolist()))
            return {'obs':self.progress.clone()},None,done,{'success':torch.ones(2,dtype=torch.bool),'precision':torch.full((2,),.1),'terminate':torch.zeros(2,dtype=torch.bool)}
    class Tracker:
        def __init__(self,player):self.player=player;self.begin_ids=[];self.end_ids=[];player.tracker=self
        def begin(self,i):self.begin_ids.append(i)
        def after_reset(self,ids):pass
        def step(self,done,info):return info
        def finish(self,i):self.end_ids.append(i)
        def complete(self,results):self.player.saved=results
    ref=Player();got=Player()
    torch.manual_seed(123);namespace['run_eval'](ref)
    torch.manual_seed(123);build_loop(BASELINE,Tracker)(got)
    assert got.events==ref.events
    assert got.saved==ref.saved
    assert got.tracker.begin_ids==got.tracker.end_ids==[0,1,2]
    assert len([event for event in got.events if event[0]=='action'])==12
    assert all(v['success_trials']==2 for v in got.saved.values())
