import numpy as np
import torch

from ind_mechanisms_common import OUT
from ind_mechanisms_scenes import solo, crossing, validate_pair, rotate_xy
from ind_mechanisms_metrics import temporal
from run_ind_mechanisms import fits, jobs
from summarize_ind_mechanisms import paired_ci


def test_success_fall_and_sustained_window():
    current = np.array([[[False,False]],[[True,False]],[[True,False]],[[False,False]]])
    valid = np.ones_like(current)
    result = temporal(current,valid,np.array([[4,3]]),.5)
    assert result['first'].tolist()==[[2,-1]]
    assert result['longest'].tolist()==[[2,0]]
    assert result['prefall'].tolist()==[[False,True]]
    assert result['hold1'].tolist()==[[True,False]]
    assert result['window'][:,0,0].tolist()==[True,True,False,False]


def test_memory_reservations_and_job_counts():
    assert fits(58000,2000,6000,70000)
    assert not fits(65000,1000,6000,70000)
    tasks=jobs('all')
    assert len(tasks)==24 and len({x['key'] for x in tasks})==24
    for phase in ('a','b'):
        for gpu in (6,7):
            subset=[x for x in tasks if x['experiment']==phase and x['preferred_gpu']==gpu]
            assert len(subset)==6
            assert sum(x['model']=='sct' for x in subset)==3


def test_scene_preservation_and_crossing_geometry():
    source=torch.load(OUT/'capture/seed1/extended.pt',weights_only=False)
    for slot in (0,1):
        scene=solo(source,slot)
        assert torch.equal(scene['roots'][:,0],source['roots'][:,slot])
        assert torch.equal(scene['kinematic'][:,0],source['kinematic'][:,slot])
        assert torch.equal(scene['boxes'],source['boxes'][:,[slot,2+slot]])
        assert scene['expected_observation'].shape==(512,318)
    cross,separate=crossing(source,101,False),crossing(source,101,True)
    validate_pair(cross,separate)
    centers=(cross['boxes'][:,:2,:2]+cross['goals'][...,:2])/2
    assert torch.allclose(centers[:,0],centers[:,1],atol=1e-4)
    centers=(separate['boxes'][:,:2,:2]+separate['goals'][...,:2])/2
    assert torch.allclose((centers[:,0]-centers[:,1]).norm(dim=-1),torch.full((512,),8.),atol=2e-4)
    offsets=[]
    for scene in (cross,separate):
        offsets.append(rotate_xy(scene['boxes'][:,2:,:3]-scene['boxes'][:,:2,:3],-scene['headings']))
    assert torch.allclose(*offsets,atol=3e-4)


def test_cluster_bootstrap_keeps_people_together():
    values=np.tile(np.array([1.,-1.]),(3,20,1))
    result=paired_ci(values)
    assert result['mean']==0 and result['ci95']==[0.,0.]
