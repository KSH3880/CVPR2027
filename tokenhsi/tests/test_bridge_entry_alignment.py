import pytest
import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from utils.experiments import native_config


@pytest.mark.parametrize('pushers', [1,2,3])
@pytest.mark.parametrize('depth', [.01,.015,.02])
def test_aligned_extent_and_initial_insertion(pushers, depth):
    cfg = native_config('stage2_sensor_900')['env']['sensorScene']['bridge']
    half_a = torch.tensor([[4.,6.],[5.,4.],[4.,4.]])
    half_b = torch.tensor([[6.,4.],[4.,6.],[6.,4.]])
    bridge = torch.tensor([[1.6*pushers,1.6,1.45]]).repeat(3,1)
    far,gap,_ = scene.bridge_stages(half_a,half_b,bridge,cfg,align_entry=True)
    torch.testing.assert_close(far[:,1]+half_b[:,1],half_a[:,1])
    torch.testing.assert_close(gap,bridge[:,0]+.2)
    near = torch.tensor([False,True,False])
    layout = scene.bridge_layout(half_a,half_b,far,gap,bridge,cfg,near,
                                 torch.ones(3,dtype=torch.bool),insert_depth=depth)
    torch.testing.assert_close(half_a[:,1]-(layout['start'][:,1]-bridge[:,1]/2),
                               torch.full((3,),depth),atol=1e-6,rtol=0)
    torch.testing.assert_close(layout['start'][:,0],half_a[:,0]+gap/2)
    torch.testing.assert_close(layout['pusher'][:,1]-layout['start'][:,1],
                               torch.full((3,),cfg['pusher_stand_off']))
    assert ((half_a[:,1]-layout['landing'][:,1]) >= bridge[:,1]/2+cfg['landing_margin']-1e-6).all()


def test_original_layout_default_is_preserved():
    cfg = native_config('stage2_sensor_900')['env']['sensorScene']['bridge']
    a,b,bridge = torch.tensor([[4.,5.]]),torch.tensor([[5.,4.]]),torch.tensor([[3.2,1.6,1.45]])
    torch.manual_seed(3)
    x = scene.bridge_stages(a,b,bridge,cfg)
    torch.manual_seed(3)
    y = scene.bridge_stages(a,b,bridge,cfg,align_entry=False)
    for original,explicit in zip(x,y):
        assert torch.equal(original,explicit)
    torch.manual_seed(4)
    x = scene.bridge_layout(a,b,y[0],y[1],bridge,cfg,torch.tensor([False]),torch.tensor([True]))
    torch.manual_seed(4)
    y = scene.bridge_layout(a,b,y[0],y[1],bridge,cfg,torch.tensor([False]),torch.tensor([True]),insert_depth=0.)
    for key in x:
        assert torch.equal(x[key],y[key])
