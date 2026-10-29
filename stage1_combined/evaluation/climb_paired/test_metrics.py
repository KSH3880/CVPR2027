import numpy as np
import pytest
from evaluation.climb_paired.metrics import score_trace, summarize


def test_success_after_thirty_seconds_and_terminal_frame_count(tmp_path):
    frames=1200;count=2
    root=np.zeros((frames,count,3));root[...,0]=2
    objects=np.zeros((frames,count,13));objects[...,6]=1
    root[950,0]=[0,0,.94]  # Success absent from the old 900-step window.
    root[-1,1]=[0,0,.94]  # Timeout frame must be counted.
    done=np.zeros((frames,count),bool);done[-1]=True
    path=tmp_path/'trace.npz'
    np.savez(path,root=root,objects=objects,alive=np.ones_like(done),done=done,
             surface=np.zeros((count,3)),extent=np.ones((count,2)),end_frame=np.full(count,frames-1))
    result=score_trace(path)
    assert result['point_count']==result['region_count']==2
    assert result['max_end_frame']==1199


def test_later_episode_success_excluded_and_incomplete_rejected(tmp_path):
    root=np.array([[[2,0,.94]],[[2,0,.94]],[[0,0,.94]]])
    objects=np.zeros((3,1,13));objects[...,6]=1
    path=tmp_path/'trace.npz'
    def save(done,alive):
        np.savez(path,root=root,objects=objects,alive=alive,done=done,
                 surface=np.zeros((1,3)),extent=np.ones((1,2)),end_frame=[1])
    save([[False],[True],[False]],[[True],[True],[False]])
    assert score_trace(path)['point_count']==0
    save([[False],[False],[False]],np.ones((3,1),bool))
    with pytest.raises(AssertionError,match='Incomplete'):score_trace(path)


def test_all_seeds_retained_even_when_one_seed_wins():
    rows=[dict(seed=seed,tokenhsi=dict(point_ever=public,region_ever=public),
               ours=dict(point_ever=ours,region_ever=ours))
          for seed,public,ours in [(0,1.,.5),(1,.5,.75)]]
    result=summarize(rows)['metrics']['region']
    assert result['winning_seeds']==[1]
    assert result['losing_seeds']==[0]
    assert result['paired_delta_mean_percentage_points']==-12.5
    assert result['ours']['mean_percent']==62.5
    assert result['ours']['sample_std_percentage_points']==pytest.approx(17.6776695297)
    with pytest.raises(AssertionError):summarize([rows[0],rows[0]])


def test_checkpoint_comparison_retains_2k_without_strict_improvement():
    from evaluation.climb_paired.compare_checkpoints import choose_checkpoint
    def pilot(value):return {'metrics':{'region':{'ours':{'mean_percent':value}}}}
    assert choose_checkpoint({2000:pilot(99.8),3000:pilot(99.09)})==2000
    assert choose_checkpoint({2000:pilot(99.8),3000:pilot(99.8)})==2000
    assert choose_checkpoint({2000:pilot(99.8),3000:pilot(100)})==3000


def test_chair_sit_uses_seat_plus10cm_and_rotated_seat_region(tmp_path):
    from scipy.spatial.transform import Rotation
    rotation=Rotation.from_euler('z',90,degrees=True)
    seat=np.array([[.3,0,.1],[.3,0,.1]])
    obj=np.zeros((2,13));obj[:,:3]=[1,2,.5];obj[:,3:7]=rotation.as_quat()
    target=obj[:,:3]+rotation.apply(seat);target[:,2]+=.1
    # A point outside20cm can still be inside the seat region, and vice versa.
    root=target+rotation.apply([[0,.25,0],[.18,0,0]])
    path=tmp_path/'sit.npz'
    np.savez(path,root=np.stack([root,root]),objects=np.stack([obj,obj]),
        alive=np.ones((2,2),bool),done=[[False,False],[True,True]],surface=seat,
        extent=[[.4,.8],[.4,.8]],end_frame=[1,1])
    stats=score_trace(path,'SIT')
    assert stats['point_count']==stats['region_count']==1
    assert score_trace(path,'CLIMB')['point_count']==0
