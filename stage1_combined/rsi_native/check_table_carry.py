"""GPU audit of every mesh-adjusted Table RSI candidate; no optimizer or policy rollout."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tokenhsi'),str(ROOT)]
import isaacgym
from isaacgym import gymtorch
import torch,numpy as np
from learning.multi_agent.ma_players import MAPlayerContinuous


def audit(player):
    t=player.env.task;player.env_reset();pool=t._table_carry_rsi
    initial=t._box_states[:,0].clone();root=t._humanoid_root_states[:,0].clone();active=pool.active
    assert t.num_envs==2048
    assert active.any(), 'No RSI candidates'
    targets=t._dof_pos.contiguous().clone()
    t.gym.set_dof_position_target_tensor(t.sim,gymtorch.unwrap_tensor(targets))
    handids=t._key_body_ids[:2];initialbody=t._kinematic_humanoid_rigid_body_states[:,0].clone()
    maxroot=torch.zeros(t.num_envs,device=t.device);maxdrop=maxroot.clone();passed=torch.ones(t.num_envs,dtype=torch.bool,device=t.device)
    measurements={}
    for step in range(18):
        t.gym.simulate(t.sim);t.gym.fetch_results(t.sim,True);t._refresh_sim_tensors()
        obj=t._box_states[:,0];human=t._humanoid_root_states[:,0]
        maxroot=torch.maximum(maxroot,(human[:,:3]-root[:,:3]).norm(dim=-1))
        maxdrop=torch.maximum(maxdrop,initial[:,2]-obj[:,2])
        passed &= torch.isfinite(human).all(-1)&torch.isfinite(obj).all(-1)&(human[:,7:10].norm(dim=-1)<3)&(human[:,2]>.5)
        if step+1 in (1,6,18):
            hp=t._rigid_body_pos[:,0,handids]
            from isaacgym.torch_utils import quat_rotate
            inv=obj[:,3:7].clone();inv[:,:3]*=-1
            local=quat_rotate(inv[:,None].expand(-1,2,-1).reshape(-1,4),(hp-obj[:,None,:3]).reshape(-1,3)).reshape(t.num_envs,2,3)
            # Fixed saved mesh contact points are tracked in object frame.
            anchors=torch.empty_like(local)
            for model in t._object_model_ids.unique().tolist():
                rows=((t._object_model_ids==model)&active).nonzero().flatten();d=pool.load(pool.keys[model])
                if d is None:continue
                anchors[rows]=d['anchors'][pool.sample[rows]]
            error=(local-anchors).norm(dim=-1).amax(-1)
            if step+1==1:measurements['fk_error']=(t._rigid_body_pos[:,0]-initialbody[:,:,:3]).norm(dim=-1).amax(-1).cpu().numpy()
            measurements[f'hand_anchor_{step+1}']=error.cpu().numpy()
            passed &= error<(.09 if step+1<=6 else .10)
    passed &= (maxroot<.30)&(maxdrop<.15)
    out=Path(t.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
    records=[]
    for model in t._object_model_ids.unique().tolist():
        rows=((t._object_model_ids==model)&active).nonzero().flatten();key=pool.keys[model];d=pool.load(key)
        if d is None:continue
        candidate_pass=np.zeros(len(d['root_state']),bool)
        for ix in range(len(candidate_pass)):
            sel=rows[pool.sample[rows]==ix]
            if not len(sel):raise ValueError('Candidate not screened')
            candidate_pass[ix]=bool(passed[sel].all())
            records.append(dict(asset=key,index=ix,replicas=len(sel),passed=bool(candidate_pass[ix]),max_root=float(maxroot[sel].max()),max_drop=float(maxdrop[sel].max()),**{k:float(v[sel.cpu().numpy()].max()) for k,v in measurements.items()}))
        path=pool.path/key/'carryWith.npz';raw=dict(np.load(path));raw['physics_pass']=candidate_pass;np.savez_compressed(path,**raw)
    (out/'physics.json').write_text(json.dumps(records,indent=2))
    print('TABLE_RSI_PHYSICS',sum(v['passed'] for v in records),'/',len(records),flush=True)

MAPlayerContinuous.run_eval=audit
import run as app
try:app.main()
finally:
    if torch.cuda.is_initialized():torch.cuda.synchronize()
