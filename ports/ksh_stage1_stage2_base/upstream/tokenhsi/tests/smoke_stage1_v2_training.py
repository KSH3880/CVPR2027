"""Run with standard run.py train/resume arguments; assert live RSI/AMP alignment."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
import run as entry
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry, build_amp_observations
from learning.multi_agent.ma_agent import MAAgent
from utils.unified_training import append_family, amp_family_ids
from utils.stage1_v2 import pose_geometry_valid

REPORT = dict(reset_batches=0,partial_resets=0,reference_slots=0,history_frames=0,
              minibatches=0,amp_matched_rows=0,high_climb_reference_slots=0,skills={})
original_init = HumanoidMACarry._init_amp_obs
original_gradients = MAAgent.calc_gradients
original_preproc = MAAgent._preproc_amp_obs
output = None


@torch.no_grad()
def checked_init(task, ids):
    global output
    if task._relation_cfg.get('stage1_variant') != 'stage1_v2':
        return original_init(task, ids)
    output = Path(task.cfg['args'].output_path)
    before = task._amp_obs_buf.clone()
    original_init(task, ids)
    labels = task._amp_family(ids).reshape(len(ids),2)
    actual = task._amp_obs_buf[ids,...,-3:]
    expected = torch.nn.functional.one_hot(labels,3).float()[:,:,None].expand_as(actual)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    unreset=torch.ones(task.num_envs,device=task.device,dtype=torch.bool);unreset[ids]=False
    torch.testing.assert_close(task._amp_obs_buf[unreset],before[unreset],atol=0,rtol=0)
    REPORT['partial_resets'] += int(bool(unreset.any()))
    seen=[]
    if task._reset_default_slots is not None:
        e,a=task._reset_default_slots;seen.append(e*2+a)
    for skill,(e,a) in task._reset_ref_slots.items():
        seen.append(e*2+a)
        motions=task._reset_ref_motion_ids[skill]
        times=task._reset_ref_motion_times[skill]
        lib=task._motion_lib[skill]
        root,rot,dof,vel,ang,dofvel,keys=lib.get_motion_state(motions,times)
        offset=task._agent_spawn_offsets[a]+task._env_origins[e]
        torch.testing.assert_close(task._humanoid_root_states[e,a,:3],root+offset)
        torch.testing.assert_close(task._dof_pos[e,a],dof)
        steps=task._num_amp_obs_steps-1
        past=(times[:,None]-task.dt*torch.arange(1,steps+1,device=task.device)).flatten()
        past_ids=motions[:,None].expand(-1,steps).flatten()
        r,q,d,v,w,dv,k=lib.get_motion_state(past_ids,past)
        history=build_amp_observations(r,q,v,w,d,dv,k,task._local_root_obs,task._root_height_obs,
                                      task._dof_obs_size,task._dof_offsets)
        history=append_family(history,task._amp_family(e,a).repeat_interleave(steps))
        torch.testing.assert_close(task._hist_amp_obs_buf[e,a],history.reshape(len(e),steps,-1),atol=1e-6,rtol=1e-5)
        if skill in ('sit','climb'):
            physical=task._scenario_physical_object(e,a)
            body=task._kinematic_humanoid_rigid_body_states[e,a,:,:3]
            assert pose_geometry_valid(body,body[:,task._key_body_ids[-2:]],
                task._box_states[e,physical,None],task._box_size[e,physical,None]).all()
            if skill=='climb':
                REPORT['high_climb_reference_slots']+=int((task._box_size[e,physical,2]>.5).sum())
        REPORT['reference_slots']+=len(e)
        REPORT['history_frames']+=len(e)*steps
        REPORT['skills'][skill]=REPORT['skills'].get(skill,0)+len(e)
    slots=torch.cat(seen).sort().values
    expected_slots=(ids[:,None]*2+torch.arange(2,device=task.device)).flatten().sort().values
    assert torch.equal(slots,expected_slots)
    REPORT['reset_batches']+=1


def checked_gradients(agent,batch):
    task=agent.vec_env.env.task
    steps=task._num_amp_obs_steps
    obs=batch['obs']
    packet=obs[:,-10:].reshape(-1,2,5).long()
    labels=torch.where(packet[:,:,1]==0,1,torch.where(packet[:,:,1]==1,2,0))
    ordered=torch.empty_like(labels).scatter_(1,packet[:,:,2],labels).flatten()
    actual=amp_family_ids(batch['amp_obs'].flatten(0,1),steps)
    assert torch.equal(actual,ordered)
    for name in ('amp_obs_demo','amp_obs_replay'):
        assert torch.equal(actual,amp_family_ids(batch[name].flatten(0,1),steps))
    REPORT['minibatches']+=1
    REPORT['amp_matched_rows']+=len(actual)
    return original_gradients(agent,batch)


def checked_preproc(agent,raw):
    normalized=original_preproc(agent,raw)
    steps=agent.vec_env.env.task._num_amp_obs_steps
    assert torch.equal(raw.reshape(-1,steps,raw.shape[-1]//steps)[...,-3:],
                       normalized.reshape(-1,steps,raw.shape[-1]//steps)[...,-3:])
    return normalized


if __name__=='__main__':
    HumanoidMACarry._init_amp_obs=checked_init
    MAAgent.calc_gradients=checked_gradients
    MAAgent._preproc_amp_obs=checked_preproc
    entry.main()
    assert REPORT['minibatches']>0 and REPORT['high_climb_reference_slots']>0
    output.mkdir(parents=True,exist_ok=True)
    (output/'rsi_amp_verification.json').write_text(json.dumps(REPORT,indent=2))
    print('PASS live RSI/AMP:',json.dumps(REPORT),flush=True)
