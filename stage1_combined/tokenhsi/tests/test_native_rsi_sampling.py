import numpy as np
import torch
from utils.native_rsi_sampling import audited_candidates,pack_pools,sample_pools


def test_full_motion_keeps_transition_but_rejects_collision_and_post_phase():
    d=dict(clip=np.zeros(7,dtype=int),frame=np.arange(7),phase=np.array([0,0,0,1,1,0,0],dtype=bool),static_pass=np.array([1,1,0,1,1,1,1],dtype=bool),physics_short_pass=np.ones(7,dtype=bool),physics_hold_pass=np.ones(7,dtype=bool))
    np.testing.assert_array_equal(np.flatnonzero(audited_candidates(d,True)),[0,1,3,4])
    np.testing.assert_array_equal(np.flatnonzero(audited_candidates(d,False)),[3,4])


def test_late_mixture_preserves_clip_weights_and_reaches_approach():
    clips=np.repeat([0,1],20);frames=np.tile(np.arange(20),2);phase=frames>=15
    arrays=pack_pools(clips,frames,phase,[.2,.8]);args=tuple(torch.tensor(x) for x in arrays)
    torch.manual_seed(42);ids,late=sample_pools(*args,100000,.7)
    assert abs(float(late.float().mean())-.7)<.01
    assert abs(float(torch.tensor(clips)[ids].float().mean())-.8)<.01
    assert (torch.tensor(frames)[ids[late]]>=18).all()
    assert (torch.tensor(frames)[ids[~late]]<15).any()
    assert not (torch.tensor(frames)[ids[late]]<15).any()


def test_missing_phase_falls_back_to_safe_full_pool_without_empty_draw():
    arrays=pack_pools(np.array([1,1]),np.array([2,5]),np.zeros(2,dtype=bool),[.5,.5])
    ids,late=sample_pools(*(torch.tensor(x) for x in arrays),100,1.)
    assert set(ids.tolist())=={0,1}
    assert late.all()
