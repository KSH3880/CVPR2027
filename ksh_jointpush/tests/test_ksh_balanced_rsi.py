import pytest
import torch
from ksh_jointpush.ksh_balanced_rsi import sample_balanced_initialization


@pytest.mark.parametrize('people',[2,3,4])
def test_equal_group_categories_and_nontrivial_mixture(people):
    torch.manual_seed(42)
    groups=torch.zeros(60000,people,dtype=torch.long)
    result=sample_balanced_initialization(groups,torch.full_like(groups,5))
    count=result.sum(-1)
    for category in [count==0,(count>0)&(count<people),count==people]:
        assert .32 < category.float().mean() < .35
    if people==3:
        mixed=(count>0)&(count<people)
        # All six valid mixed subsets should occur with roughly equal probability.
        for pattern in range(1,7):
            encoded=(result*(2**torch.arange(people))).sum(-1)
            assert .14 < (encoded[mixed]==pattern).float().mean() < .19


def test_solo_is_half_and_inactive_templates_are_masked():
    torch.manual_seed(43)
    groups=torch.arange(4).expand(20000,-1)
    templates=torch.full_like(groups,5);templates[:,1]=1
    result=sample_balanced_initialization(groups,templates)
    assert not result[:,1].any()
    for column in [0,2,3]:assert .48 < result[:,column].float().mean() < .52


def test_interleaved_groups_and_partial_batch():
    torch.manual_seed(44)
    groups=torch.tensor([[0,1,0,1]]).expand(20000,-1)
    result=sample_balanced_initialization(groups,torch.full_like(groups,5))
    for columns in [[0,2],[1,3]]:
        count=result[:,columns].sum(-1)
        for n in [0,1,2]:assert .31 < (count==n).float().mean() < .36
    small=groups[:3]
    assert sample_balanced_initialization(small,torch.full_like(small,5)).shape==(3,4)
