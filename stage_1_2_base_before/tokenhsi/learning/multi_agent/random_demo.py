"""Save high-success multi-person random scenes with every BEFORE pair successful."""
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
from utils.edge_ontop_spec import batched
from utils.demo_reset import DemoResetRejected, task_only_demo_boxes
from utils.task_role_spec import task_graph_packet
from utils.evaluation_protocol import spatial_success


def candidates(spatial, native, valid, pair_success, initial_gate, pairs, dt, minimum_success=None, minimum_pairs=1):
    """Rank simultaneous sustained success; no vacuous/early BEFORE successes."""
    span = math.ceil(.5 / dt - 1e-6)
    n, m = spatial.shape[1:]
    minimum_success = math.ceil(.75*m) if minimum_success is None else minimum_success
    streak = np.zeros((n,m),int)
    pair_streak = np.zeros(len(pairs),int)
    best = {}
    first_gate = [None]*len(pairs)
    early = np.asarray(initial_gate,dtype=bool).copy()
    for t in range(len(spatial)):
        good = spatial[t] & native[t] & valid[t,:,None]
        streak = np.where(good,streak+1,0)
        pair_streak = np.where(pair_success[t] & valid[t,pairs[:,0]], pair_streak+1, 0)
        for p,(e,f,l) in enumerate(pairs):
            # AT is independently rechecked from geometry, then gated native pair.
            if first_gate[p] is None:
                if spatial[t,e,l] and native[t,e,l]: first_gate[p] = t
                elif spatial[t,e,f]: early[p] = True
        for e in range(n):
            ps = np.flatnonzero(pairs[:,0]==e)
            if len(ps)<minimum_pairs or early[ps].any() or not (pair_streak[ps]>=span).all(): continue
            members = np.unique(pairs[ps,1:].reshape(-1))
            if not (streak[e,members]>=span).all(): continue
            count = int((streak[e]>=span).sum())
            if count < minimum_success: continue
            if e not in best or count > best[e]['count']:
                best[e] = dict(env=e,frame=t,count=count,successful=np.flatnonzero(streak[e]>=span).tolist(),
                               pairs=pairs[ps,1:].tolist())
    return sorted(best.values(),key=lambda x:(-x['count'],x['frame']))


def near_waits(roots, native, valid, packet, goals, pairs, dt, radius, seconds=1., extent=0., require_walk=False, initial_roots=None):
    """Every follower waits near the final goal before the first native AT success."""
    from learning.multi_agent.ontop_demo import qualifying_wait
    m=len(packet)
    # AT target tokens follow all agent and logical object tokens.
    reports=[]
    for follower,leader in pairs:
        # Each AT actor has a scenario goal indexed by target token; infer token
        # offset from the standard task graph goal numbering (M+O, O=2*M).
        goal_id=int(packet[leader,4])-3*m
        if not 0 <= goal_id < len(goals):raise ValueError('Unexpected goal token for random8 scene')
        distance=np.linalg.norm(roots[:,follower,:2]-goals[goal_id,:2],axis=-1)
        speed=np.linalg.norm(roots[:,follower,7:9],axis=-1)
        gate=native[:,leader]
        eligible=np.ones(len(roots),bool)
        if require_walk:
            initial=(roots[0] if initial_roots is None else initial_roots)[follower,:2]
            displacement=np.linalg.norm(roots[:,follower,:2]-initial,axis=-1)
            approach=np.linalg.norm(initial-goals[goal_id,:2])-distance
            eligible=(displacement>=1.) & (approach>=.75)
        wait=None
        for r in sorted(set([min(2.,radius),radius])):
            wait=qualifying_wait(distance,speed,eligible,gate,valid,dt,radius=r,seconds=seconds,positions=roots[:,follower,:2],max_extent=extent or None)
            if wait is not None and distance[wait[0]:wait[2]+1].max()<=radius:break
            wait=None
        if wait is None:return None
        start,end,first=wait
        reports.append(dict(follower=int(follower),leader=int(leader),wait_start_step=start+1,wait_end_step=end+1,
            first_gate_step=first+1,max_wait_distance=float(distance[start:end+1].max()),
            max_distance_until_gate=float(distance[start:first+1].max()),max_wait_speed=float(speed[start:end+1].max())))
    return reports


@torch.no_grad()
def search(player):
    task=player.env.task
    assert task.headless and task.num_agents in (8,16) and task._task_graph_preset=='random_scenario'
    n,m=task.num_envs,task.num_agents
    if task_only_demo_boxes(task):
        for module in player.model.modules():
            if hasattr(module,'_humans'): module.demo_task_boxes_only=True
    spawn_radius=float(os.environ.get('RANDOM_DEMO_SPAWN_RADIUS','0'))
    if spawn_radius:
        if getattr(task.cfg['args'],'eval_protocol',''): raise ValueError('Demo spawn override forbidden for formal evaluation')
        angles=torch.arange(m,device=task.device)*(2*math.pi/m)
        task._agent_spawn_offsets[:,:2]=spawn_radius*torch.stack((angles.cos(),angles.sin()),-1)
        print(f'[random demo] initial spawn radius={spawn_radius}m',flush=True)
    out=Path(task.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
    player._reward_debug=False
    summaries=[]
    minimum_pairs=int(os.environ.get('RANDOM_DEMO_MIN_PAIRS','1'))
    maximum_pairs=int(os.environ.get('RANDOM_DEMO_MAX_PAIRS','0'))
    if maximum_pairs:
        if n!=1: raise ValueError('Pair-count proposal filtering requires one scene per batch')
        original_draw=task._draw_episode_graph
        def selected_graph(ids):
            for _ in range(1024):
                proposal=original_draw(ids)
                count=int((proposal[1]>=0).sum())
                if minimum_pairs<=count<=maximum_pairs: return proposal
            raise RuntimeError('No graph fits the requested pair-count proposal range')
        task._draw_episode_graph=selected_graph
    minimum_success=int(os.environ.get('RANDOM_DEMO_MIN_SUCCESS',str(math.ceil(.75*m))))
    high_success=int(os.environ.get('RANDOM_DEMO_HIGH_SUCCESS',str(math.ceil(.875*m))))
    target_cases=int(os.environ.get('RANDOM_DEMO_TARGET_CASES','1'))
    saved_high=0
    continue_failed=os.environ.get('RANDOM_DEMO_CONTINUE_FAILED','0')=='1'
    wait_radius=float(os.environ.get('RANDOM_DEMO_WAIT_RADIUS','0'))
    if not 0 <= wait_radius <= 4:raise ValueError('Wait radius must be 0 (off) to 4m')
    wait_seconds=float(os.environ.get('RANDOM_DEMO_WAIT_SECONDS','1'))
    wait_extent=float(os.environ.get('RANDOM_DEMO_WAIT_EXTENT','0'))
    wait_require_walk=os.environ.get('RANDOM_DEMO_WAIT_REQUIRE_WALK','0')=='1'
    if wait_seconds <= 0 or wait_extent < 0: raise ValueError('Invalid wait window')
    for batch in range(int(os.environ.get('RANDOM_DEMO_BATCHES','6'))):
        try:
            obs=player.env_reset()
        except DemoResetRejected as error:
            summaries.append(dict(batch=batch,scenes=n,reset_rejected=True,reason=str(error)))
            (out/'search_summary.json').write_text(json.dumps(summaries,indent=2))
            print(f'[random demo] skipped infeasible batch={batch}',flush=True)
            continue
        player.get_batch_size(obs['obs'],1)
        graph=task.relation_runtime.graph
        packet=task_graph_packet(graph,n).reshape(n,m,5).long().clone()
        owner,edge_valid,required=[batched(getattr(graph,k),n) for k in ('edge_owner','edge_valid','required_goal')]
        mask=graph.prereq_mask
        if mask.dim()==2: mask=mask[None].expand(n,-1,-1)
        rows,edges=(edge_valid & required & mask.any(-1)).nonzero(as_tuple=True)
        parent=mask[rows,edges].long().argmax(-1)
        followers,leaders=owner[rows,edges],owner[rows,parent]
        pairs=torch.stack((rows,followers,leaders),-1).cpu().numpy()
        initial_gate=task.relation_runtime.own_success[rows,parent].cpu().numpy()
        owned=(owner[:,None]==torch.arange(m,device=task.device)[None,:,None]) & (edge_valid & required)[:,None]
        assert owned.any(-1).all()
        initial=dict(packet=packet.cpu().numpy(), logical_box_order=task._logical_box_order.cpu().numpy().copy(),
            sizes_physical=task._box_size.cpu().numpy().copy(),goals=task._tar_pos.cpu().numpy().copy(),origin=task._env_origins.cpu().numpy().copy(),
            initial_roots=task._root_states.reshape(n,-1,13).cpu().numpy().copy(),initial_dofs=task._dof_state.reshape(n,m,-1,2).cpu().numpy().copy())
        sizes=task._logical_box_values(task._box_size).clone()
        traces={k:[] for k in ('roots','bodies','dofs','spatial','native','valid','pair_success','agent_valid')}
        alive=torch.ones(n,device=task.device,dtype=torch.bool)
        agent_alive=torch.ones(n,m,device=task.device,dtype=torch.bool)
        for step in range(task.max_episode_length):
            obs,_,done,_=player.env_step(player.env,player.get_action(obs,True))
            if not isinstance(obs,dict):obs={'obs':obs}
            raw=task.relation_runtime.last_result['raw_own_success']
            agent_alive &= ~task._terminate_buf.reshape(n,m).bool()
            valid=torch.ones_like(alive) if continue_failed else alive & ~task._terminate_buf.reshape(n,m).bool().any(-1)
            native=(task.relation_runtime.own_success[:,None] | ~owned).all(-1)
            if continue_failed: native &= agent_alive
            spatial=spatial_success(packet,task._humanoid_root_states[...,:3],task._logical_box_values(task._box_states),sizes,task._tar_pos)[...,1]
            values=dict(roots=task._root_states.reshape(n,-1,13),dofs=task._dof_state.reshape(n,m,-1,2),
                bodies=task._rigid_body_state.reshape(n,-1,13)[:,:m*task.num_bodies].reshape(n,m,task.num_bodies,13),
                spatial=spatial,native=native,valid=valid,agent_valid=agent_alive.clone(),pair_success=task.relation_runtime.own_success[rows,edges] & raw[rows,parent] & agent_alive[rows,followers] & agent_alive[rows,leaders])
            for k,v in values.items():traces[k].append(v.cpu().numpy().copy())
            if continue_failed:
                alive=agent_alive.any(-1)
            else:
                alive &= ~done.reshape(n,m).bool().any(-1)
            if (step+1)%200==0:print(f'[random demo] batch={batch} step={step+1} alive={int(alive.sum())}',flush=True)
            if not alive.any():break
        traces={k:np.stack(v) for k,v in traces.items()}
        ranked=candidates(traces['spatial'],traces['native'],traces['valid'],traces['pair_success'],initial_gate,pairs,task.dt,minimum_success,minimum_pairs)
        all_ranked=ranked
        ranked=[c for c in ranked if len(c['pairs'])>=minimum_pairs]
        if wait_radius:
            kept=[]
            for c in ranked:
                e=c['env'];end=c['frame']+1
                waits=near_waits(traces['roots'][:end,e],traces['native'][:end,e],traces['valid'][:end,e],initial['packet'][e],initial['goals'][e],c['pairs'],task.dt,wait_radius,wait_seconds,wait_extent,wait_require_walk,initial['initial_roots'][e])
                if waits is not None:
                    c['near_waits']=waits;kept.append(c)
            ranked=sorted(kept,key=lambda c:(max((w['max_wait_distance'] for w in c['near_waits']),default=0)>2,-c['count'],c['frame']))
        summary=dict(batch=batch,scenes=n,candidates=ranked,unfiltered_candidates=all_ranked,recorded_steps=len(traces['valid']),peak_instant_success=int(((traces['spatial'] & traces['native']) & traces['valid'][...,None]).sum(-1).max()))
        print('[random demo] candidates before/after pair filter:',len(all_ranked),len(ranked),flush=True)
        summaries.append(summary);(out/'search_summary.json').write_text(json.dumps(summaries,indent=2))
        for candidate in ranked[:3]:
            e=candidate['env'];frame=candidate['frame'];end=frame+1
            # End on the verified high-success pose, without subsequent deterioration.
            directory=out/f'case_b{batch}_e{e}';directory.mkdir()
            ps=np.flatnonzero(pairs[:,0]==e)
            np.savez_compressed(directory/'trajectory.npz',**{k:v[:end,e] for k,v in traces.items() if k!='pair_success'},
                pair_success=traces['pair_success'][:end,ps],initial_gate=initial_gate[ps],pairs=pairs[ps,1:],**{k:v[e] for k,v in initial.items()})
            packet_e=initial['packet'][e];order=initial['logical_box_order'][e]
            boxes=[];box_colors=[]
            for a in range(m):
                slots=[packet_e[a,4]] if packet_e[a,1]<2 else [packet_e[a,3]]+([packet_e[a,4]] if packet_e[a,1]==3 else [])
                for slot in slots:
                    b=int(order[int(slot)-m])
                    if b not in boxes: boxes.append(b);box_colors.append(a)
            ckpt=Path(task.cfg['args'].checkpoint)
            meta=dict(protocol='random_all_before_v2',task_name=f'RANDOM{m}',num_agents=m,dt=task.dt,seed=task.cfg['args'].seed,batch=batch,env_id=e,
                checkpoint=str(ckpt.resolve()),checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                people=list(range(m)),replay_box_ids=boxes,box_color_agents=box_colors,
                tasks=packet_e[:,1].tolist(),before_pairs=candidate['pairs'],successful_agents=candidate['successful'],
                minimum_success=minimum_success,minimum_pairs=minimum_pairs,proposal_maximum_pairs=maximum_pairs or None,
                task_boxes_only=os.environ.get('RANDOM_DEMO_TASK_BOXES_ONLY')=='1',spawn_radius_override=spawn_radius or None,
                continue_after_individual_failure=continue_failed,
                wait_radius=wait_radius,wait_seconds=wait_seconds,wait_extent=wait_extent,wait_require_walk=wait_require_walk,near_waits=candidate.get('near_waits'),
                success_count=candidate['count'],success_step=end,success_seconds=end*task.dt,
                success_definition='simultaneous native and independent spatial success >=0.5s; every BEFORE pair succeeds; failed agents excluded; scene continuation follows metadata',
                preset='random_scenario')
            (directory/'metadata.json').write_text(json.dumps(meta,indent=2));print('[random demo] SAVED '+str(directory)+' '+str(candidate),flush=True)
        saved_high += sum(c['count']>=high_success for c in ranked[:3])
        if saved_high>=target_cases:break
    torch.cuda.synchronize();print('[random demo] complete',flush=True)
