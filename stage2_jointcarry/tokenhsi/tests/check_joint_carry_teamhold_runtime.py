import json
import runpy
import sys
from copy import deepcopy
from pathlib import Path

import isaacgym
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from learning.multi_agent.ma_players import MAPlayerContinuous
from env.tasks.multi_agent.joint_carry_reward import JointCarryRuntime, anchor_scores
from env.tasks.multi_agent.edge_context_reward import owner_sum
from utils.edge_context_spec import HOLDING
from utils.edge_ontop_spec import ON_TOP


def expected_score(task,ids=None):
    if ids is None:
        body=task._rigid_body_pos
        objects=task._logical_box_values(task._box_states)
        sizes=task._logical_box_values(task._box_size)
        joint=task._joint_env_mask
    else:
        body=task._kinematic_humanoid_rigid_body_states[ids,...,:3]
        objects=task._logical_box_values(task._box_states,ids)
        sizes=task._logical_box_values(task._box_size,ids)
        joint=task._joint_env_mask[ids]
    hands=body[...,task._key_body_ids[[0,1]],:]
    scores,_=anchor_scores(hands,objects[:,0],sizes[:,0],.15)
    c=torch.stack((scores[:,0,0].minimum(scores[:,1,1]),
                   scores[:,0,1].minimum(scores[:,1,0])),-1).amax(-1)
    return c*joint


@torch.no_grad()
def check(player):
    task=player.env.task
    config=deepcopy(task._relation_cfg)
    assert config['joint_carry'].pop('holding_state')=='opposite_anchor_min'
    assert 'alignment' not in config['joint_carry']
    obs=player.env_reset();player.get_batch_size(obs['obs'],1)
    graph=task.relation_runtime.graph
    old=JointCarryRuntime(task.num_envs,graph,config,task.device)
    changed=0
    dense=0
    for _ in range(32):
        obs,_,_,info=player.env_step(player.env,player.get_action(obs,True))
        obs=player.obs_to_torch(obs)
        result=task.relation_runtime.last_result
        diag=task._ontop_last_diag
        c=expected_score(task)
        torch.testing.assert_close(task.relation_runtime.joint_holding_state,c)
        old.coupled_holding.copy_(task.relation_runtime.coupled_holding)
        before=old.step(result['phi_raw'],result['progress_raw'],diag['z_error'],
                        diag['feet_height_error'],diag['region_error'])
        shared=(graph.edge_relation==HOLDING)&graph.edge_valid&task._joint_env_mask[:,None]
        expected=torch.where(shared,.2*c[:,None],before['state_component'])
        torch.testing.assert_close(result['state_component'],expected)
        for key in ['progress_component','success_component','own_success']:
            torch.testing.assert_close(result[key],before[key],rtol=0,atol=0)
        torch.testing.assert_close(task.relation_runtime.done,old.done)
        torch.testing.assert_close(result['total'],before['total']+expected-before['state_component'])
        torch.testing.assert_close(result['agent_task_reward'],owner_sum(result['total'],graph))
        torch.testing.assert_close(result['agent_task_reward'][~task._joint_env_mask],
                                  before['agent_task_reward'][~task._joint_env_mask],rtol=0,atol=0)
        held_state=owner_sum(result['state_component']*shared,graph)[task._joint_env_mask]
        torch.testing.assert_close(held_state[:,0],held_state[:,1])
        terms=info['reward_terms'].reshape(task.num_envs,2,-1)
        torch.testing.assert_close(terms[...,0],owner_sum(expected,graph))
        torch.testing.assert_close(terms[...,:-1].sum(-1),terms[...,-1])
        assert 'joint_alignment' not in task.REWARD_TERM_NAMES
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
        assert torch.isfinite(task.rew_buf).all()
        changed+=int(((result['state_component']-before['state_component']).abs()>1e-6).sum())
        dense+=int(((c>0)&~task.relation_runtime.coupled_holding&task._joint_env_mask).sum())
    assert changed>0 and dense>0
    ids=torch.arange(0,task.num_envs,2,device=task.device)
    keep=torch.ones(task.num_envs,device=task.device,dtype=torch.bool);keep[ids]=False
    previous=task.relation_runtime.joint_holding_state[keep].clone()
    task._reset_envs(ids)
    torch.testing.assert_close(task.relation_runtime.joint_holding_state[keep],previous,rtol=0,atol=0)
    torch.testing.assert_close(task.relation_runtime.joint_holding_state[ids],expected_score(task,ids))
    ontop=((graph.edge_relation==ON_TOP)&graph.edge_valid).any(-1)
    report=dict(steps=32,joint_at=int((task._joint_env_mask&~ontop).sum()),
        joint_ontop=int((task._joint_env_mask&ontop).sum()),independent=int((~task._joint_env_mask).sum()),
        modified_holding_edges=changed,dense_gate_off_scene_steps=dense,
        individual_progress_success_unchanged=True,placement_gate_unchanged=True,
        independent_rewards_unchanged=True,state_replaced_not_added=True,
        both_agents_receive_same_holding_state=True,partial_reset_cache_correct=True,
        observation_amp_reward_finite=True)
    out=Path(task.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
    (out/'teamhold_runtime_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print('TEAMHOLD_RUNTIME',json.dumps(report),flush=True)


MAPlayerContinuous.run_eval=check
runpy.run_path(str(ROOT/'run.py'),run_name='__main__')
