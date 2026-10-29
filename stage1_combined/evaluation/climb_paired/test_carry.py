"""AT40s must count late/terminal success, while excluding reset episodes."""
import numpy as np
import pytest
from evaluation.climb_paired.continuous_carry import score_trace,CarryTracker,PROTOCOL,TABLE_PROTOCOL
from types import SimpleNamespace


@pytest.mark.parametrize('category,protocol',[('ArmChair_Normal',PROTOCOL),('Table_Circle',TABLE_PROTOCOL)])
def test_carry_protocol_follows_actual_object_category(tmp_path,category,protocol):
    task=SimpleNamespace(num_envs=4,num_agents=1,num_objects=1,_is_eval=True,max_episode_length=1200,dt=1/30,
        cfg=dict(args=SimpleNamespace(output_path=str(tmp_path)),env=dict(holdAtFT=dict(category=category,asset_split='test'))))
    tracker=CarryTracker(SimpleNamespace(env=SimpleNamespace(task=task)))
    assert tracker.protocol==protocol


def write_trace(path,positions,done,terminate=None,max_steps=1200):
    objects=np.zeros((*positions.shape,13),np.float32);objects[...,0]=positions
    goals=np.zeros((*positions.shape,3),np.float32)
    active=np.ones(positions.shape[1],bool);masks=[];lengths=np.zeros(len(active),int)
    for d in done:
        masks.append(active.copy());lengths+=active;active&=~d
    point=np.abs(positions.astype(np.float64))<=.2
    np.savez(path,objects=objects,goals=goals,done=done,active=np.stack(masks),point=point,region=point,
        terminate=np.zeros_like(done) if terminate is None else terminate,episode_steps=lengths,max_steps=max_steps)


def test_at40s_late_terminal_success_and_no_post_reset_leak(tmp_path):
    positions=np.ones((1199,3),np.float32);done=np.zeros_like(positions,dtype=bool)
    positions[1100,0]=.1;positions[1198,1]=.1
    done[0,2]=True;positions[1:,2]=0;done[-1,:2]=True
    terminate=np.zeros_like(done);terminate[0,2]=True
    path=tmp_path/'trace.npz';write_trace(path,positions,done,terminate)
    result=score_trace(path)
    assert result['point_count']==result['region_count']==2
    assert result['max_end_frame']==1199 and result['initial_frame_scored'] is False
    assert result['float32_float64_ever_equal']


def test_at40s_incomplete_trace_rejected(tmp_path):
    path=tmp_path/'trace.npz';positions=np.zeros((2,1),np.float32)
    write_trace(path,positions,np.zeros((2,1),bool))
    with pytest.raises(ValueError,match='Incomplete'):score_trace(path)
