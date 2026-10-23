"""CPU checks of task-matched late sampling and rigid pose alignment."""
from isaacgym import gymapi
from types import SimpleNamespace
import pytest
import torch
from utils.open_push_rsi import OpenPushRSIBank, validate_rsi, align_pose


def spec():
    return dict(bank_file='unused.json',interaction_probability=.5,late_probability=.7,
                late_fraction=.3,stand_off=.06)


def libraries():
    def lib(n):
        return SimpleNamespace(gts=torch.zeros(1),num_motions=lambda:n,
            _motion_weights=torch.ones(n)/n,_motion_lengths=torch.ones(n)*10)
    return {'doorOpen':lib(1),'push':lib(2)}


def proposals():
    return {skill:[dict(motion_id=m,time=float(i),phase=i/10,angle=.1*i if skill=='doorOpen' else 0.,
        box_progress=.08*i if skill=='push' else 0.) for m in range(n) for i in range(10)]
        for skill,n in [('doorOpen',1),('push',2)]}


def test_late_sampling_preserves_equal_push_clip_weights():
    torch.manual_seed(5);bank=OpenPushRSIBank(spec(),libraries(),{},proposals())
    s=bank.sample('push',10000)
    assert .47<s['motion_id'].float().mean()<.53
    assert .67<s['late_requested'].float().mean()<.73
    assert (s['phase'][s['late_requested']]>=.7).all()
    assert (s['phase'][~s['late_requested']]<.7).any()


def test_bank_requires_every_clip_and_valid_parameters():
    p=proposals();p['push']=[r for r in p['push'] if r['motion_id']==0]
    with pytest.raises(ValueError,match='clip 1'):OpenPushRSIBank(spec(),libraries(),{},p)
    for key in ('interaction_probability','late_probability','late_fraction','stand_off'):
        s=spec();s[key]=True
        with pytest.raises(ValueError):validate_rsi(s)


def test_alignment_rotates_reference_and_keeps_floor_height():
    body=torch.tensor([[[0.,0.,.9],[.5,0.,1.],[.4,0.,.8],[0.,0.,.05]]])
    rot=torch.tensor([0.,0.,0.,1.]).expand(1,4,4).clone()
    velocity=torch.tensor([1.,0.,0.]).expand_as(body).clone()
    target=torch.tensor([[2.,3.,1.]])
    angle=torch.tensor([1.2])
    p,r,v,a,q=align_pose(body,rot,velocity,velocity,rot[:,0],target,angle,[1,2],.06)
    normal=torch.tensor([[-torch.sin(angle[0]),torch.cos(angle[0]),0.]])
    torch.testing.assert_close(p[:,1],target-.06*normal)
    torch.testing.assert_close(p[...,2],body[...,2])
    torch.testing.assert_close((p[:,1]-p[:,0]).norm(dim=-1),(body[:,1]-body[:,0]).norm(dim=-1))
    torch.testing.assert_close(v.norm(dim=-1),velocity.norm(dim=-1))


def test_push_alignment_fits_both_points_without_lifting_feet():
    from utils.open_push_rsi import align_push_pose
    body=torch.tensor([[[0.,0.,.9],[.5,-.2,1.05],[.5,.2,1.05],[0.,0.,.05]]])
    rot=torch.tensor([0.,0.,0.,1.]).expand(1,4,4).clone()
    targets=torch.tensor([[[2.2,3.,1.05],[1.8,3.,1.05]]])
    vel=torch.zeros_like(body)
    p,r,v,a,q=align_push_pose(body,rot,vel,vel,targets,[1,2],.06)
    expected=targets.clone();expected[...,1]-=.06
    torch.testing.assert_close(p[:,[1,2]],expected)
    torch.testing.assert_close(p[...,2],body[...,2])


def test_lower_push_arms_preserves_feet_and_segment_lengths():
    from utils.open_push_rsi import lower_push_arms
    body=torch.tensor([[[0.,0.,.9],[0.,-.2,1.1],[.25,-.2,1.075],[.5,-.2,1.05],
                        [0.,.2,1.1],[.25,.2,1.075],[.5,.2,1.05],[0.,0.,.05]]])
    rot=torch.tensor([0.,0.,0.,1.]).expand(1,8,4).clone()
    vel=torch.zeros_like(body);dof=torch.zeros(1,18)
    parents=[-1,0,1,2,0,4,5,0]
    p,r,v,a,d,dv=lower_push_arms(body,rot,vel,vel,dof,dof,parents,[3,6],
                               [1,2,3,4,5,6],[0,3,6,9,12,15,18],.8)
    torch.testing.assert_close(p[:,[3,6],2],torch.full((1,2),.8))
    torch.testing.assert_close(p[:,[0,1,4,7]],body[:,[0,1,4,7]])
    for child,parent in enumerate(parents):
        if parent>=0:
            torch.testing.assert_close((p[:,child]-p[:,parent]).norm(dim=-1),
                                      (body[:,child]-body[:,parent]).norm(dim=-1))
    assert torch.isfinite(d).all() and d.abs().sum()>0
    torch.testing.assert_close(d[:,3:9],dof[:,3:9])
    torch.testing.assert_close(body[:,[3,6],2],torch.full((1,2),1.05))


def test_retarget_matches_shoulder_forward_kinematics():
    from utils.open_push_rsi import lower_push_arms
    from utils.door_reference import rotate
    from utils.torch_utils import exp_map_to_quat
    body=torch.tensor([[[0.,0.,.9],[0.,0.,1.1],[.25,0.,1.05],[.5,0.,1.]]])
    rot=torch.tensor([0.,0.,0.,1.]).expand(1,4,4).clone();vel=torch.zeros_like(body);dof=torch.zeros(1,9)
    angular=torch.zeros_like(body);angular[:,1:,1]=.3
    vel[:,1:]=torch.cross(angular[:,1:],body[:,1:]-body[:,1,None],dim=-1)
    dof_vel=dof.clone();dof_vel[:,1]=.3
    p,r,v,a,d,dv=lower_push_arms(body,rot,vel,angular,dof,dof_vel,[-1,0,1,2],[3],
                              [1,2,3],[0,3,6,9],.8)
    actual=body[:,1]+rotate(exp_map_to_quat(d[:,:3]),body[:,3]-body[:,1])
    torch.testing.assert_close(actual,p[:,3])
    torch.testing.assert_close(v,torch.zeros_like(v),atol=1e-6,rtol=0)
    torch.testing.assert_close(a,torch.zeros_like(a),atol=1e-6,rtol=0)
    torch.testing.assert_close(dv[:,:3],torch.zeros(1,3))


def test_bank_load_checks_pd_screen_and_physics_contract(tmp_path):
    import json
    s=spec();path=tmp_path/'bank.json';s['bank_file']=str(path)
    data=dict(contract={'physics':1},screen='PD_12_steps',candidates=proposals())
    path.write_text(json.dumps(data))
    OpenPushRSIBank(s,libraries(),{'physics':1})
    with pytest.raises(ValueError,match='contract mismatch'):
        OpenPushRSIBank(s,libraries(),{'physics':2})
    data['screen']='geometry_only';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='screen/contract mismatch'):
        OpenPushRSIBank(s,libraries(),{'physics':1})


def test_new_config_keeps_original_loco_only_experiment():
    from pathlib import Path
    import yaml
    from utils.open_push_spec import validate_env
    root=Path(__file__).resolve().parents[1]
    base=yaml.safe_load((root/'data/cfg/multi_agent/open_push_unified_amp.yaml').read_text())['env']
    new=yaml.safe_load((root/'data/cfg/multi_agent/open_push_late_rsi.yaml').read_text())['env']
    validate_env(new)
    assert 'openPushRSI' not in base
    assert new['openPushRSI']['interaction_probability']==.5
    assert {k:v for k,v in new.items() if k!='openPushRSI'}==base


def test_open_rsi_excludes_already_open_candidates_and_keeps_late_sampling():
    rows=proposals();rows['doorOpen'][-1]['angle']=1.6
    bank=OpenPushRSIBank(spec(),libraries(),{},rows,max_open_angle=1.48)
    samples=bank.sample('doorOpen',1000)
    assert (samples['angle']<=1.48).all()
    assert samples['late_requested'].any()
    assert all(r['angle']<=1.48 for r in bank.rows['doorOpen'])
    with pytest.raises(ValueError,match='No screened OPEN'):
        OpenPushRSIBank(spec(),libraries(),{},rows,max_open_angle=-1)


def test_geometry_final_frame_time_stays_inside_float32_motion_duration():
    from utils.open_push_rsi import geometry_candidates
    n = 361
    pos = torch.zeros(n, 4, 3)
    pos[:, 0, 2] = 1.
    pos[:, 1:3, 0] = .5
    pos[:, 1:3, 2] = 1.
    pos[:, 3, 2] = .1
    rot = torch.zeros(n, 4, 4)
    rot[..., 3] = 1.
    lib = SimpleNamespace(gts=pos, grs=rot, length_starts=torch.tensor([0]),
        num_motions=lambda:1, _motion_num_frames=torch.tensor([n]),
        _motion_dt=torch.tensor([1 / 30]), _motion_lengths=torch.tensor([12.]),
        _motion_weights=torch.ones(1))
    libs = {'doorOpen': lib, 'push': lib}
    assert (n - 1) * float(lib._motion_dt[0]) > float(lib._motion_lengths[0])
    rows = geometry_candidates(libs, [1, 2], [3], history=0.)
    assert rows['push'][-1]['phase'] == 1.
    assert rows['push'][-1]['time'] == 12.
    OpenPushRSIBank(spec(), libs, {}, rows)


def test_push_only_bank_accepts_no_open_candidates_and_keeps_push_clips():
    p=proposals();p['doorOpen']=[]
    bank=OpenPushRSIBank(spec(),libraries(),{},p,active_skills=('push',))
    assert bank.rows['doorOpen']==[] and set(bank.pools)=={'push'}
    sample=bank.sample('push',100)
    assert set(sample['motion_id'].tolist())=={0,1}
    p['push']=[r for r in p['push'] if r['motion_id']==0]
    with pytest.raises(ValueError,match='clip 1'):
        OpenPushRSIBank(spec(),libraries(),{},p,active_skills=('push',))
