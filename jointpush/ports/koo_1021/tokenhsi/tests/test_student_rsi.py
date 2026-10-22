import json
from types import SimpleNamespace

import pytest
import torch

from utils.reference_rsi import digest, validate_reference_rsi
from utils.student_rsi import StudentRSIBank


class Lib:
    def __init__(self,path):
        self._motion_files=[str(path)]*2
        self._motion_weights=torch.tensor([1.,3.])
    def num_motions(self):return 2
    def sample_motions(self,n):return torch.ones(n,dtype=torch.long)
    def sample_time_rsi(self,ids):return torch.full_like(ids,.15,dtype=torch.float)
    def get_motion_length(self,ids):return torch.ones_like(ids,dtype=torch.float)


def make_bank(tmp_path,records,**overrides):
    dataset=tmp_path/'dataset.yaml';dataset.write_text('dataset')
    asset=tmp_path/'asset.xml';asset.write_text('<mujoco/>')
    motion=tmp_path/'motion.npy';motion.write_text('motion')
    obj=tmp_path/'obj.npy';obj.write_text('object')
    names=('sit','climb','pickUp','carryWith','putDown')
    libs={name:Lib(motion) for name in names}
    meta={'dataset_sha256':digest(dataset),'asset_sha256':digest(asset)}
    for key,values in [('files',['motion.npy']*2),('objects',['obj.npy']*2),
                       ('motion_sha256',[digest(motion)]*2),('object_sha256',[digest(obj)]*2)]:
        meta[key]={name:values for name in names}
    payload={'schema':1,'stage':'student_repeated_low_impact_not_safety_certified',
             'source_student_summary':{'measurement_version':2,'policy_role':'student',
                 'nonfinite_scenes':0,'partial_reset_checked':True},
             'reference_metadata':meta,'candidates':records}
    bank=tmp_path/'bank.json';bank.write_text(json.dumps(payload))
    spec={'mode':'student_candidates','bank':str(bank),'min_observations':3,
          'max_delta_v':2.,'carry_max_hand_increase':.2,'late_fraction':1.,
          'phase_range':[.9,.98],'biased_skills':['pickUp','carryWith','putDown'],
          'min_late_candidates':3,'empty_skill':'original','restore_box_velocity':True}
    spec.update(overrides)
    return StudentRSIBank(spec,libs,dataset,asset,'cpu')


def row(mid=0,phase=.95,size=(65,65,65),skill='carryWith',dv=.5,hand=.1,n=3):
    return {'motion_id':mid,'phase':phase,'size_cm':list(size),'skill':skill,
            'student_n':n,'student_first2_delta_v_max':dv,
            'student_hand_box_distance_increase_max_m':hand}


def test_fixed_size_and_original_skill_fallback(tmp_path):
    bank=make_bank(tmp_path,[row()])
    sizes=torch.tensor([[.65,.65,.65],[.2,.2,.2]])
    before=sizes.clone();ids,times,empty=bank.sample('carryWith',sizes)
    torch.testing.assert_close(sizes,before)
    assert ids.tolist()==[0,1]
    torch.testing.assert_close(times,torch.tensor([.95,.15]))
    assert not empty.any() # caller must not convert missing skill to loco
    assert bank.stats['missing_size_skill_original']==1
    assert bank.stats['same_size_general_fallback']==1


def test_late_pool_then_same_size_less_late_fallback(tmp_path):
    records=[row(phase=.7),row(phase=.91),row(phase=.93),row(phase=.95),
             row(phase=.72,size=(20,20,20)),row(phase=.94,size=(20,20,20))]
    bank=make_bank(tmp_path,records)
    sizes=torch.tensor([[.65,.65,.65]]*100+[[.2,.2,.2]]*100)
    _,times,_=bank.sample('carryWith',sizes)
    assert (times[:100]>=.9).all()
    assert (times[100:]<.9).all()
    assert bank.stats['late_selected']==100
    assert bank.stats['same_size_general_fallback']==100


def test_repeat_delta_and_carry_grasp_filters(tmp_path):
    bank=make_bank(tmp_path,[row(n=2),row(dv=2.1),row(hand=.21),
                            row(skill='putDown',hand=1.)])
    sizes=torch.tensor([[.65,.65,.65]])
    _,carry,_=bank.sample('carryWith',sizes)
    _,release,_=bank.sample('putDown',sizes)
    assert carry.item()==pytest.approx(.15)
    assert release.item()==pytest.approx(.95)
    assert bank.stats['excluded_candidates']==3


def test_clip_weights_do_not_depend_on_candidate_count(tmp_path):
    torch.manual_seed(38)
    bank=make_bank(tmp_path,[row(phase=p) for p in (.7,.75,.8,.85)]+[row(mid=1,phase=.85)],late_fraction=0.)
    ids,_,_=bank.sample('carryWith',torch.tensor([[.65,.65,.65]]*4000))
    assert .72<float((ids==1).float().mean())<.78


def test_student_quality_weights_and_climb_no_end_bias(tmp_path):
    torch.manual_seed(44)
    bank=make_bank(tmp_path,[row(phase=.7,dv=.5,skill='climb'),row(phase=.95,dv=1.5,skill='climb')])
    _,times,_=bank.sample('climb',torch.tensor([[.65,.65,.65]]*4000))
    assert .63<float((times<.9).float().mean())<.70
    assert bank.stats['late_requests']==0


@pytest.mark.parametrize('changes',[{'empty_skill':'loco'},{'restore_box_velocity':False},
    {'min_observations':2},{'max_delta_v':3.},{'carry_max_hand_increase':.3}])
def test_reject_invalid_contract(tmp_path,changes):
    with pytest.raises(ValueError):make_bank(tmp_path,[row()],**changes)
