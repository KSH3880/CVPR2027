from types import SimpleNamespace
import torch
from jointpush.ksh_push_only import push_templates,fetch_push_demo
from utils.stage2_jointpush import sample_groups,group_sizes,physical_assignment


def test_push_only_retains_solo_pairs_and_large_groups_without_aliasing():
    torch.manual_seed(42)
    groups=sample_groups(10000,4,.8,'cpu')
    sizes=group_sizes(groups)
    templates=push_templates(groups)
    assert (templates==5).all() and set(sizes.unique().tolist())=={1,2,3,4}
    assert ((sizes==2).all(-1)).any() and ((sizes==3).sum(-1)==3).any()
    assigned=physical_assignment(templates,groups)
    assert torch.equal(assigned[:,:,None]==assigned[:,None,:],groups[:,:,None]==groups[:,None,:])


def test_demo_uses_only_push_family_and_loco_push_references():
    counts={}
    class Library:
        def __init__(self,name):self.name=name
        def sample_motions(self,count):counts[self.name]=count;return torch.zeros(count,dtype=torch.long)
        def sample_time(self,ids,truncate_time):return torch.zeros(len(ids))
    task=SimpleNamespace(cfg={'env':{'skill':['loco','sit','push'],'ampExpertMapping':{'push':[.5,0.,.5]}}},device='cpu',dt=1/30,_num_amp_obs_steps=10,_motion_lib={name:Library(name) for name in ('push_loco','push')},get_num_amp_obs=lambda:1330,build_amp_obs_demo=lambda ids,times,lib:torch.zeros(len(ids),1290))
    torch.manual_seed(42)
    result=fetch_push_demo(task,4096).reshape(4096,10,133)
    assert torch.isfinite(result).all()
    assert (result[...,-4:].argmax(-1)==3).all()
    assert set(counts)=={'push_loco','push'} and sum(counts.values())==4096
    assert .45<counts['push_loco']/4096<.55
