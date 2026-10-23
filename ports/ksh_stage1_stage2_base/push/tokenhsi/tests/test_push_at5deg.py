"""AT tilt success gate: boundaries, yaw, saturation, partial reset and contracts."""
from isaacgym import gymapi
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from utils.edge_context_spec import AT, HOLDING
from utils.edge_scenario_spec import compose_canonical_graph
from utils.stage1_push import validate_env, check_checkpoint
from env.tasks.multi_agent.humanoid_ma_stage1_push import PushRuntime


def config():
    return yaml.safe_load(Path("tokenhsi/data/cfg/multi_agent/stage1_push_midpoint_safe_at5deg.yaml").read_text())["env"]


def test_config_and_checkpoint_separation():
    env=config(); validate_env(env)
    old=yaml.safe_load(Path("tokenhsi/data/cfg/multi_agent/stage1_push_midpoint_safe.yaml").read_text())["env"]
    expected=deepcopy(old);expected["pushTask"]["at_success_max_tilt_deg"]=5.
    assert env==expected
    check_checkpoint({"relation_experiment_config":{"env":env}},env)
    with pytest.raises(ValueError):check_checkpoint({"relation_experiment_config":{"env":old}},env)


@pytest.mark.parametrize("bad", [-1,181,float("nan"),float("inf"),True,"5"])
def test_invalid_limits(bad):
    env=config();env["pushTask"]["at_success_max_tilt_deg"]=bad
    with pytest.raises(ValueError,match="tilt"):validate_env(env)


@pytest.mark.parametrize("angle,passed", [(0,True),(4.999,True),(5,True),(5.001,False),(45,False),(180,False)])
def test_gate_and_partial_reset(angle,passed):
    env=config();graph=compose_canonical_graph(torch.full((3,2),3))
    boxes=torch.zeros(3,4,13);boxes[...,6]=1
    # yaw 90deg followed by pitch: yaw must not contribute to tilt.
    a=math.radians(angle)/2;s=math.sqrt(.5)
    boxes[1,0,3:7]=torch.tensor([-s*math.sin(a),s*math.sin(a),s*math.cos(a),s*math.cos(a)])
    task=SimpleNamespace(cfg={"env":env},_box_states=boxes,_push_hits=torch.zeros(3,2,dtype=torch.bool))
    runtime=PushRuntime(task,3,graph,env["relationReward"],"cpu")
    phi=torch.full((3,4),.1);at=graph.edge_relation==AT;hold=graph.edge_relation==HOLDING
    phi[at]=math.exp(-10*.19**2);z=torch.zeros_like(phi);prog=torch.full_like(phi,.3)
    result=runtime.step(phi,prog,z,z,z)
    mask=at[1] & (graph.edge_src[1]==2)
    assert result["own_success"][1,mask].item()==passed
    assert result["own_success"][at].sum().item()==5+int(passed)
    assert not result["own_success"][hold].any()
    assert result["reward_saturated"][1].sum().item()==2+2*int(passed)
    before=runtime.own_success.clone()
    runtime.reset(torch.tensor([2,1]),phi[[2,1]],z[[2,1]],z[[2,1]],z[[2,1]])
    assert torch.equal(runtime.own_success,before)
    # Native distance and height requirements remain necessary below 5deg.
    phi[at]=math.exp(-10*.21**2)
    assert not runtime.step(phi,prog,z,z,z)["own_success"].any()
    phi[at]=1.;z[at]=.002
    assert not runtime.step(phi,prog,z,z,z)["own_success"].any()
