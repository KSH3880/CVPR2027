"""Native Isaac Gym GPU viewer for a saved initial-release schedule."""
import os
from pathlib import Path
import sys
import json
import runpy

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True

def read_only_ksh(event, args):
    paths = []
    if event == 'open':
        path, mode, flags = args
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
            paths = [path]
    elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.truncate'):
        paths = [args[0]]
    elif event in ('os.rename', 'os.link', 'os.symlink'):
        paths = args[:2]
    for path in paths:
        if isinstance(path, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(path)).resolve()
            if str(resolved).startswith('/home/hwanhee/ksh/'):
                raise PermissionError('Recording process forbids writes under ksh')

sys.addaudithook(read_only_ksh)
sys.path[:0] = [str(ROOT/'tokenhsi'), str(ROOT/'eval')]
import isaacgym
import torch
from learning.multi_agent import formal_eval
from utils.task_role_spec import task_graph_packet
from utils.evaluation_protocol import predecessors
import stage2_protocol_sampling
import stage2_scalability_layout
stage2_scalability_layout.install()

# Reuse identical formal scene initialization, but display it interactively.
# This viewer does not produce a formal evaluation score.
from utils import evaluation_protocol
_original_configure = evaluation_protocol.configure
def viewer_configure(args, cfg, name):
    headless = args.headless
    try:
        args.headless = True
        return _original_configure(args, cfg, name)
    finally:
        args.headless = headless
evaluation_protocol.configure = viewer_configure

import numpy as np
import time
from isaacgym import gymapi

class ViewerClosed(Exception):
    pass
sys.path.insert(0, str(ROOT.parent))
from sct_scheduler.isaac_adapter import InitialReleaseFreeze
from sct_scheduler.core import validate_response
from sct_scheduler.stage1 import require_stage1_request, load_stage1

@torch.no_grad()
def replay(player):
    task = player.env.task
    dest = Path(task.cfg['args'].output_path)
    source = Path(os.environ['SCT_REPLAY_CAPTURE'])
    audit = Path(os.environ['SCT_REPLAY_SCHEDULE'])
    request = json.loads((audit/'request.json').read_text())
    answer = json.loads((audit/'response.validated.json').read_text())
    validate_response(request, answer)
    require_stage1_request(request)
    expected = torch.load(source/'initial_scene.pt',map_location='cpu',weights_only=False)
    controller = None
    policy = None
    try:
        obs = player.env_reset()
        n,m=task.num_envs,task.num_agents
        observed=dict(roots=task._humanoid_root_states,physical_boxes=task._box_states,
            physical_sizes=task._box_size,packet=task_graph_packet(task.relation_runtime.graph,n).reshape(n,m,5).long(),
            predecessor=predecessors(task.relation_runtime.graph,n,task.device),
            logical_box_order=task._logical_box_order,assignment=task._agent_box_assignment,
            goals=task._tar_pos,center_goals=task._center_tar_pos,groups=task._groups,templates=task._templates)
        checks={}
        for key,value in observed.items():
            actual=value.cpu(); old=expected[key]
            matches=torch.equal(actual,old) if not actual.is_floating_point() else torch.allclose(actual,old,atol=1e-5,rtol=0)
            checks[key]=dict(matches=bool(matches),max_abs_difference=float((actual-old).abs().max()))
        (dest/'initial_match.json').write_text(json.dumps(checks,indent=2))
        if not all(x['matches'] for x in checks.values()):
            raise ValueError('Reset state differs from scheduled initial scene; no rollout performed')
        order=[a['agent_id'] for a in request['current_scene']['initial_observation']['agents']]
        controller=InitialReleaseFreeze(task,[request],[answer],[order],physical_gpu=7)
        task._compute_observations()
        obs = player.obs_to_torch(player.env._policy_observation())
        policy = load_stage1(player, obs['obs'], request)
        player.get_batch_size(obs['obs'],1)
        if task.viewer is None:
            raise ValueError('Native viewer required; do not pass --headless')
        groups = list(dict.fromkeys(t['group_id'] for t in request['current_scene']['tasks']))
        ui = dict(paused=True, finished=False, camera=0)
        task.gym.subscribe_viewer_keyboard_event(task.viewer,gymapi.KEY_SPACE,'sct_pause')
        task.gym.subscribe_viewer_keyboard_event(task.viewer,gymapi.KEY_Q,'sct_quit')
        for k in range(len(groups)+1):
            task.gym.subscribe_viewer_keyboard_event(task.viewer,getattr(gymapi,'KEY_'+str(k)),'sct_camera_'+str(k))
        def set_camera(k):
            members = list(range(m)) if k == 0 else [i for i,t in enumerate(request['current_scene']['tasks']) if t['group_id']==groups[k-1]]
            pts=task._humanoid_root_states[0,members,:3].clone()
            extras=[]
            for i in members:
                t=request['current_scene']['tasks'][i]
                for key in ('object_goal_position_m','support_intended_final_center_m'):
                    if key in t: extras.append(t[key])
            if extras: pts=torch.cat((pts,pts.new_tensor(extras)),0)
            center=pts.mean(0).cpu().tolist(); radius=max(3.,float((pts[:,:2]-pts[:,:2].mean(0)).norm(dim=-1).max())+1.)
            task.gym.viewer_camera_look_at(task.viewer,None,
                gymapi.Vec3(center[0]-radius*1.3,center[1]-radius*1.3,radius*1.1+2.),
                gymapi.Vec3(center[0],center[1],.7))
            ui['camera']=k
        def render(sync_frame_time=False):
            if task.gym.query_viewer_has_closed(task.viewer): raise ViewerClosed()
            for evt in task.gym.query_viewer_action_events(task.viewer):
                if evt.value<=0: continue
                if evt.action in ('QUIT','sct_quit'): raise ViewerClosed()
                if evt.action=='sct_pause' and not ui['finished']: ui['paused']=not ui['paused']
                if evt.action.startswith('sct_camera_'):set_camera(int(evt.action.rsplit('_',1)[1]))
            task.gym.step_graphics(task.sim)
            task.gym.draw_viewer(task.viewer,task.sim,True)
        task.render=render
        set_camera(1)
        (dest/'viewer_ready.json').write_text(json.dumps(dict(ready=True,paused=True,physical_gpu=7,
            groups={str(k+1):[t['agent_id'] for t in request['current_scene']['tasks'] if t['group_id']==g] for k,g in enumerate(groups)},
            controls={'Space':'start/pause','0':'all agents','1-7':'group camera','Q':'GPU synchronize and normal exit'}),indent=2))
        print('[NATIVE_VIEWER_READY] Space=start/pause, 0=all, 1-7=groups, Q=normal exit',flush=True)
        active=sorted(set(task._logical_box_order[0].cpu().tolist())|set(task._agent_box_assignment[0].cpu().tolist()))
        trace={k:[] for k in ('time_s','roots','bodies','boxes','frozen','terminated')}
        zero_drift_root=zero_drift_dof=0.
        def capture(t):
            trace['time_s'].append(t)
            for key,value in dict(roots=task._humanoid_root_states[0],bodies=task._rigid_body_pos[0],
                boxes=task._box_states[0,active],frozen=controller.frozen[0],terminated=task._terminate_buf.reshape(n,m)[0]).items():
                trace[key].append(value.cpu().numpy().copy())
        capture(0.)
        for frame in range(600):
            while ui['paused']:
                render()
                time.sleep(1/30)
            started=time.monotonic()
            controller.before_action(frame)
            task._compute_observations()
            obs=player.obs_to_torch(player.env._policy_observation())
            actions=controller.mask_actions(player.get_action(obs,True))
            player.env_step(player.env,actions)
            time.sleep(max(0.,task.dt-(time.monotonic()-started)))
            mask=controller.frozen
            if mask.any():
                zero_drift_root=max(zero_drift_root,float((task._humanoid_root_states[mask]-controller.saved_root[mask]).abs().max()))
                zero_drift_dof=max(zero_drift_dof,float((task._dof_pos[mask]-controller.saved_dof[mask]).abs().max()))
            if (frame+1)%2==0:
                capture((frame+1)*task.dt)
            if (frame+1)%150==0:
                print('[REPLAY_STEP]',frame+1,flush=True)
        torch.cuda.synchronize(task.device)
        np.savez_compressed(dest/'rollout.npz',**{k:np.asarray(v) for k,v in trace.items()},
                            box_ids=np.asarray(active),box_sizes=task._box_size[0,active].cpu().numpy())
        (dest/'request.json').write_text(json.dumps(request,indent=2))
        (dest/'schedule.json').write_text(json.dumps(answer,indent=2))
        (dest/'replay.json').write_text(json.dumps(dict(completed=True,steps=600,dt_s=task.dt,
            snapshot_stride=2,physical_gpu=7,initial_state_matches=True,api_calls=0,
            max_frozen_root_error_after_projection=zero_drift_root,
            max_frozen_dof_error_after_projection=zero_drift_dof,
            body_names=task.gym.get_actor_rigid_body_names(task.envs[0],task.humanoid_handles_all[0][0]),
            policy='Stage1 SCT 26k with initial LLM release only; Stage2 coordination disabled.',
            source_capture=str(source),source_api=str(audit)),indent=2))
        print('[REPLAY_COMPLETE] Viewer stays open; Q exits normally.',flush=True)
        ui['finished']=True
        while True:
            render()
            time.sleep(1/30)
    except ViewerClosed:
        print('[VIEWER_NORMAL_EXIT]',flush=True)
    finally:
        torch.cuda.synchronize(task.device)
        if controller is not None:
            (dest/'release_audit.json').write_text(json.dumps(controller.close(),indent=2))
        if policy is not None:
            proof = policy.finish()
            proof.update(external_wait_gate=True, controller='LLM initial release only',
                         scene_carrier_checkpoint=task.cfg['args'].checkpoint)
            (dest/'stage1_policy.json').write_text(json.dumps(proof,indent=2))

formal_eval.run_formal_eval=replay
runpy.run_path(str(ROOT/'tokenhsi/run.py'),run_name='__main__')
