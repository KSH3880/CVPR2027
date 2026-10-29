"""CONTACT ablation equivalence and isolation from AT/gate/initialization."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
import yaml
from ksh_jointpush import ksh_legacy_contact as legacy

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'runs/jointpush/ksh_jointpush_rsi/launch_base19500_hard_safe_clear_gpu5_20261028_214039/configs'
WORK=ROOT/'runs/jointpush/ksh_jointpush_rsi/workspace_push_solo45_coupled0_balanced_rsi_hard_gate_base_Stage1PushFinetune_00019500_4eb295a8_safe_start_clear_paths_coupled_jhh_solo_ksh_contact/configs'

def original(path,name):
    source=(ROOT/path).read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    scope=dict(torch=torch,quat_rotate=legacy.quat_rotate,quat_conjugate=legacy.quat_conjugate)
    exec(ast.get_source_segment(source,node),scope)
    return scope[name]

def test_matches_original_geometry_and_route():
    torch.manual_seed(81)
    boxes=torch.randn(3,4,13);boxes[...,3:7]=torch.nn.functional.normalize(boxes[...,3:7],dim=-1)
    sizes=torch.rand(3,4,3)+1
    hands=torch.randn(3,4,2,3);hf=torch.randn_like(hands);bf=torch.randn(3,4,3)
    expected=original(Path('jointpush/ports/jhh_raw_cuboid/tokenhsi/env/tasks/multi_agent/push_reward.py'),'contact_geometry')
    for a,b in zip(expected(hands,boxes,sizes,hf,bf,True,1.04),legacy.contact_geometry(hands,boxes,sizes,hf,bf,True,1.04)):
        assert torch.equal(a,b)
    route=original(Path('jointpush/partial/reward.py'),'approach_distance')
    roots=torch.randn(3,4,3);normals=legacy.contact_geometry(hands,boxes,sizes,hf,bf)[3]
    args=(roots,boxes[...,:3],normals,sizes/2,torch.randn(3,4))
    assert torch.equal(route(*args),legacy.approach_distance(*args))

def test_height_and_contact_success():
    box=torch.zeros(1,1,13);box[...,2]=.75;box[...,6]=1
    size=torch.full((1,1,3),1.5)
    hands=torch.tensor([[[[-.1,-.75,1.04],[.1,-.75,1.04]]]])
    hf=torch.tensor([[[[0.,-1.,0.],[0.,-1.,0.]]]]);bf=torch.tensor([[[0.,1.,0.]]])
    phi,hit,*_=legacy.contact_geometry(hands,box,size,hf,bf,True,1.04)
    assert phi.item()==1 and hit.item()
    hands[...,2]=.8
    phi,hit,*_=legacy.contact_geometry(hands,box,size,hf,bf,True,1.04)
    assert phi.item()<1e-10 and not hit.item()

def test_only_contact_edges_change(monkeypatch):
    gate=torch.tensor([[.3,.8]])
    class Task:
        def _evaluate_ontop_context(self,ids=None):
            self.relation_runtime.gate.copy_(gate)
            return torch.tensor([[.11,.22,.33,.44]]), {k:torch.tensor([[1.,2.,3.,4.]]) for k in ('progress','distance','distance_xy','z_error')}
    task=Task();task.num_envs=1;task.device='cpu';task._rigid_body_pos=torch.zeros(1,2,1,3)
    task._members=torch.tensor([[1,2]])
    task.relation_runtime=SimpleNamespace(gate=torch.zeros_like(gate),graph=SimpleNamespace(edge_owner=torch.tensor([[0,0,1,1]]),edge_relation=torch.tensor([[12,7,12,7]])))
    monkeypatch.setattr(legacy,'values',lambda *args:(torch.tensor([[.7,.8]]),None,*[torch.tensor([[.5,.6]])]*4))
    legacy.install({'stage2JointPush':{'contact_reward':legacy.contract()}},Task)
    phi,diag=task._evaluate_ontop_context()
    assert torch.equal(phi,torch.tensor([[.11,.22,.8,.44]]))
    for value in diag.values():assert torch.equal(value[:,[0,1,3]],torch.tensor([[1.,2.,4.]]))
    assert torch.equal(task.relation_runtime.gate,gate)

def test_reference_config_only_contact_diff():
    before=yaml.safe_load((REFERENCE/'env.yaml').read_text())
    after=yaml.safe_load((WORK/'env.yaml').read_text())
    assert after['env']['stage2JointPush'].pop('contact_reward')==legacy.contract()
    assert before==after
    assert (REFERENCE/'train.yaml').read_bytes()==(WORK/'train.yaml').read_bytes()
    assert (REFERENCE/'motions.yaml').read_bytes()==(WORK/'motions.yaml').read_bytes()

def test_native_install_noop():
    class Task:pass
    legacy.install({'stage2JointPush':{}},Task)
    assert not hasattr(Task,'_legacy_contact_installed')

def test_checkpoint_contract_rejects_native_model(monkeypatch):
    monkeypatch.chdir(ROOT/'ksh_jointpush')
    import pytest
    from utils.stage2_jointpush import checkpoint_contract, check_checkpoint
    before=yaml.safe_load((REFERENCE/'env.yaml').read_text())['env']
    after=yaml.safe_load((WORK/'env.yaml').read_text())['env']
    before.update(numAgents=4,numObjects=8);after.update(numAgents=4,numObjects=8)
    native={'stage2_jointpush_contract':checkpoint_contract(before)}
    with pytest.raises(ValueError,match='settings do not match'):
        check_checkpoint(native,after)
    check_checkpoint({'stage2_jointpush_contract':checkpoint_contract(after)},after)

def test_physics_success_uses_ksh_for_solo_and_jhh_for_shared(monkeypatch):
    class Task:
        def _evaluate_ontop_context(self, ids=None): pass
    task=Task();task.num_envs=1;task.device='cpu';task.control_freq_inv=1
    task.render=lambda:None;task._refresh_sim_tensors=lambda:None
    task.gym=SimpleNamespace(simulate=lambda sim:None,fetch_results=lambda *args:None);task.sim=None
    task._members=torch.tensor([[1,2]]);task._push_hits=torch.zeros(1,2,dtype=torch.bool)
    task._rigid_body_pos=torch.zeros(1,2,2,3);task._key_body_ids=torch.tensor([0,1])
    task._contact_forces=torch.tensor([[[[0.,-1.,0.],[0.,-1.,0.]],[[0.,-1.,0.],[0.,-1.,0.]]]])
    task._ontop_box_contact=torch.tensor([[[0.,1.,0.],[0.,1.,0.]]])
    task._assigned_box_values=lambda value:value
    task._contact_geometry=lambda *args:(torch.zeros(1,2),torch.tensor([[[0.,1.],[0.,1.]]]),None,None,None)
    task._joint={'contact':dict(near_distance_scale=10.,satisfaction_threshold=.9,min_hand_force=.5,min_box_force=.5)}
    monkeypatch.setattr(legacy,'values',lambda *args:(None,torch.zeros(1,2,dtype=torch.bool)))
    legacy.install({'stage2JointPush':{'contact_reward':legacy.contract()}},Task)
    task._physics_step()
    assert torch.equal(task._push_hits,torch.tensor([[True,False]]))
    # A later shared hit is accumulated, without replacing the solo hit.
    monkeypatch.setattr(legacy,'values',lambda *args:(None,torch.ones(1,2,dtype=torch.bool)))
    task._physics_step()
    assert task._push_hits.all()
