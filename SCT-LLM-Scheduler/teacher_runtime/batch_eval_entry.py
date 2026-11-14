"""Local Stage1+LLM adapter around the original 16-person formal evaluator."""
import os,sys,json,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parent
# Same read-only audit guard as the local capture entry, without executing it.
source=(ROOT/'capture_initial_entry.py').read_text()
exec(source.split('import isaacgym')[0],globals())
import isaacgym
import torch
import numpy as np
from learning.multi_agent import formal_eval
from utils.task_role_spec import task_graph_packet
from utils.evaluation_protocol import predecessors
import stage2_protocol_eval
sys.path.insert(0,str(ROOT.parent))
from sct_scheduler.stage1 import policy_manifest, require_stage1_request
from sct_scheduler.isaac_adapter import InitialReleaseFreeze
if os.environ.get('SCT_FREEZE_IMPL', 'legacy') == 'snapshot_resume':
    from sct_scheduler.snapshot_freeze import SnapshotFreeze as InitialReleaseFreeze
elif os.environ.get('SCT_FREEZE_IMPL', 'legacy') != 'legacy':
    raise ValueError('Unknown SCT_FREEZE_IMPL')
from sct_scheduler.core import earliest_schedule,validate_response
from sct_scheduler.executable import validate_executable
from sct_scheduler.fewshot import matching_initial_request
from export_initial_scene import export

base_capture=stage2_protocol_eval.capture_initial

def capture(player):
    result=base_capture(player)
    task=player.env.task; n,m=task.num_envs,task.num_agents;out=Path(task.cfg['args'].output_path)
    raw=dict(packet=task_graph_packet(task.relation_runtime.graph,n).reshape(n,m,5).long(),
        predecessor=predecessors(task.relation_runtime.graph,n,task.device),roots=task._humanoid_root_states,
        physical_boxes=task._box_states,physical_sizes=task._box_size,
        logical_boxes=task._evaluation_box_values(task._box_states),logical_sizes=task._evaluation_box_values(task._box_size),
        logical_box_order=task._logical_box_order,assignment=task._agent_box_assignment,goals=task._tar_pos,
        center_goals=task._center_tar_pos,groups=task._groups,templates=task._templates,
        task_slots=task._stage2_task_slots,
        hands=task._kinematic_humanoid_rigid_body_states[:,:,task._key_body_ids[:2],:3])
    raw={k:v.cpu().clone() for k,v in raw.items()};torch.save(raw,out/'initial_scene.pt')
    (out/'capture.json').write_text(json.dumps(dict(completed=True,environments=n,agents=m,seed=task.cfg['args'].seed,
        policy_controller_id=policy_manifest()['controller_id'],control_dt_s=task.dt,simulated_steps=0),indent=2))
    for i in range(n):export(out,out/'requests'/('env_%04d'%i),i,raw)
    print('[REQUESTS_EXPORTED]',n,flush=True)
    torch.cuda.synchronize(task.device)
    return result
stage2_protocol_eval.capture_initial=capture
base_eval=formal_eval.run_formal_eval

@torch.no_grad()
def evaluate(player):
    task=player.env.task;out=Path(task.cfg['args'].output_path);seed=task.cfg['args'].seed
    reference=Path(os.environ['STAGE2_REFERENCE_ROOT'])/('general_seed'+str(seed))
    expected=torch.load(reference/'initial_scene.pt',map_location='cpu',weights_only=False)
    requests=[json.loads((reference/'requests'/('env_%04d'%i)/'request.json').read_text()) for i in range(task.num_envs)]
    preflight=os.environ.get('SCT_PREFLIGHT')=='1';answers=[]
    execution_only=os.environ.get('SCT_EXECUTION_ONLY')=='1'
    excluded=[];validation_errors=[]
    for i,req in enumerate(requests):
        require_stage1_request(req)
        if preflight:
            scene=req['current_scene'];owner={}
            for t in scene['tasks']:
                for k,v in t.items():
                    if k.endswith('_event_id'):owner[v]=t['agent_id']
            preds=[dict(agent_id=a,event_id=e,estimated_elapsed_time_s=1.+(int(a[1:])%4),confidence='low',basis='Preflight only; not an LLM prediction.') for e,a in owner.items()]
            releases=earliest_schedule(scene,preds)
            answer=dict(predictions=preds,schedule=[dict(agent_id=a,release_time_s=t) for a,t in releases.items()],
                predicted_episode_events=[dict(agent_id=p['agent_id'],event_id=p['event_id'],episode_time_s=round(releases[p['agent_id']]+p['estimated_elapsed_time_s'],2)) for p in preds])
        else:
            folder=Path(os.environ['SCT_SCHEDULE_ROOT'])/('general_seed'+str(seed))/('env_%04d'%i)
            assert matching_initial_request(json.loads((folder/'request.json').read_text()),req),'API request does not match captured scene'
            try:
                answer=json.loads((folder/('response.raw.txt' if execution_only else 'response.validated.json')).read_text())
                if not isinstance(answer,dict):raise ValueError('Response is not an object')
            except (OSError,ValueError):
                if not execution_only:raise
                answer={}
        # Historical output fields are dropped only from the in-memory copy.
        if answer.get('status') == 'ok':
            answer = {k:v for k,v in answer.items() if k != 'status'}
        answer = {k:v for k,v in answer.items() if k != 'limitations'}
        if execution_only and not preflight:
            try:
                validate_response(req,answer)
            except Exception as exc:
                validation_errors.append(dict(environment=i,error=str(exc)))
            try:
                validate_executable(req,answer)
            except Exception as exc:
                excluded.append(dict(environment=i,error=str(exc)))
                # Keep the original 512-env sampling and slot mapping. This slot
                # stays frozen throughout and is removed before any scoring.
                answer=dict(schedule=[dict(agent_id=a['agent_id'],release_time_s=1000000.)
                    for a in req['current_scene']['initial_observation']['agents']])
        else:
            validate_response(req,answer)
        answers.append(answer)
    if preflight and os.environ.get('SCT_PREFLIGHT_EXCLUDE_LAST')=='1':
        i=len(requests)-1
        excluded.append(dict(environment=i,error='Synthetic exclusion test; not an API failure'))
        answers[i]=dict(schedule=[dict(agent_id=a['agent_id'],release_time_s=1000000.)
            for a in requests[i]['current_scene']['initial_observation']['agents']])
    task._sct_excluded_envs=[r['environment'] for r in excluded]
    (out/'schedule_execution_audit.json').write_text(json.dumps(dict(
        execution_only=execution_only,excluded=excluded,validation_errors=validation_errors,
        schedules_modified=False,evaluated_scenes=task.num_envs-len(excluded)),indent=2))
    reset,action,step=player.env_reset,player.get_action,player.env_step
    adapter=None;frame=0;first_step={};frozen_error=0.
    def reset_once(*args,**kwargs):
        nonlocal adapter
        assert adapter is None,'Unexpected reset during evaluation'
        obs=reset(*args,**kwargs)
        comparisons={k:bool(torch.equal(getattr(task,attr).cpu(),expected[k])) for k,attr in
            [('roots','_humanoid_root_states'),('goals','_tar_pos'),('groups','_groups'),('logical_box_order','_logical_box_order')]}
        (out/'llm_initial_match.json').write_text(json.dumps(comparisons,indent=2));assert all(comparisons.values())
        adapter=InitialReleaseFreeze(task,requests,answers,[[a['agent_id'] for a in r['current_scene']['initial_observation']['agents']] for r in requests],physical_gpu=int(os.environ['SCT_PHYSICAL_GPU']),execution_only=execution_only)
        task._compute_observations()
        return player.obs_to_torch(player.env._policy_observation())
    def scheduled_action(obs,is_determenistic=False):
        adapter.before_action(frame)
        task._compute_observations()
        obs=player.obs_to_torch(player.env._policy_observation())
        if frame==0 and os.environ.get('SCT_PROTOCOL_VERSION'):
            actual=obs['obs'].clone()
            kin=task._kinematic_humanoid_rigid_body_states
            velocities=kin[...,7:13].clone()
            buffer=task.obs_buf.clone()
            try:
                kin[...,7:13][adapter.frozen]=0
                ids=torch.arange(task.num_envs,device=task.device)
                task.obs_buf.copy_(task._compute_clean_scene_obs(ids))
                expected_input=player.obs_to_torch(player.env._policy_observation())['obs'].clone()
            finally:
                kin[...,7:13].copy_(velocities)
                task.obs_buf.copy_(buffer)
            torch.save(dict(actual=actual.cpu(),expected_reset_with_freeze=expected_input.cpu(),
                            frozen=adapter.frozen.cpu()),out/'first_policy_input.pt')
            exact=torch.equal(actual,expected_input)
            (out/'first_policy_input.json').write_text(json.dumps(dict(
                passed=exact,comparison='indexed reset kinematics with frozen velocities zero',
                max_abs_error=float((actual-expected_input).abs().max()),
                protocol=os.environ['SCT_PROTOCOL_VERSION'])))
            assert exact,'First policy input differs from reset state with initial freeze'
        return adapter.mask_actions(action(obs,is_determenistic))
    def checked_step(*args,**kwargs):
        nonlocal frame,frozen_error
        value=step(*args,**kwargs)
        if frame==0:
            displacement=(task._humanoid_root_states[...,:3].cpu()-expected['roots'][...,:3]).norm(dim=-1)
            gap=(task._humanoid_root_states[...,:3]-task._rigid_body_pos[:,:,0]).norm(dim=-1)
            first_step.update(max_displacement_m=float(displacement.max()),max_root_body_gap_m=float(gap.max()))
            (out/'first_step_check.json').write_text(json.dumps(first_step,indent=2))
            assert float(displacement.max())<.5,'Initial teleport detected; evaluation aborted'
            assert float(gap.max())<.01,'Inconsistent root/body after first step'
        mask=adapter.frozen
        if mask.any():frozen_error=max(frozen_error,float((task._humanoid_root_states[mask]-adapter.saved_root[mask]).abs().max()))
        frame+=1
        return value
    player.env_reset,player.get_action,player.env_step=reset_once,scheduled_action,checked_step
    if preflight:task.cfg['env']['episodeLength']=30
    try:
        result=base_eval(player)
        result['llm_scheduler']=dict(preflight=preflight,first_step=first_step,max_frozen_root_error=frozen_error,api_during_rollout=0,scene_count=len(requests)-len(excluded),excluded=excluded,execution_only=execution_only,validation_errors=validation_errors)
        return result
    finally:
        player.env_reset,player.get_action,player.env_step=reset,action,step
        torch.cuda.synchronize(task.device)
        if adapter is not None:(out/'release_audit.json').write_text(json.dumps(adapter.close(),indent=2))
formal_eval.run_formal_eval=evaluate
entry='table2_eval.py' if os.environ.get('TABLE2_SETTING') else 'stage1_protocol_eval.py'
runpy.run_path(str(ROOT/'eval'/entry),run_name='__main__')
