"""Audit retargeted RSI in the native PhysX scene; never run an optimizer."""
from pathlib import Path
import json,math,hashlib
import numpy as np
import torch
from isaacgym import gymapi,gymtorch
from evaluation.finetuned.backend import EvaluationTask
from learning.multi_agent.chair_ft_agent import ChairFTPlayer
from rsi_native.geometry import NativeGeometry,collision_samples


class NativeRsiAuditTask(EvaluationTask):
    def _load_box_asset(self):
        self._audit_root=Path(self.cfg['env']['chairFT']['rsi_audit']['bank'])
        manifest=json.loads((self._audit_root/'manifest.json').read_text());entries=manifest['assets'];self._audit_assets=entries
        ids=torch.arange(self.num_envs,device=self.device)%len(entries);self._audit_ids=ids
        def values(key):return torch.tensor([e[key] for e in entries],device=self.device,dtype=torch.float32)[ids,None]
        self._box_size=values('bbox');self._box_scale=self._box_size/torch.tensor(self._build_base_size,device=self.device)
        self._box_density=torch.full((self.num_envs,),100.,device=self.device)
        self._chair_features=values('feature');self._chair_seats=values('seat');self._chair_seat_extents=values('extent');self._climb_metadata_heights=values('metadata_height')
        kinds=torch.tensor([2 if e['category']=='Chair' else 1 for e in entries],device=self.device);self._chair_kinds=kinds[ids,None]
        self._rsi_asset_sizes=torch.full((self.num_envs,1,3),.45,device=self.device)
        self._box_assets=[];cache={}
        for index in ids.cpu().tolist():
            if index not in cache:
                options=gymapi.AssetOptions();options.fix_base_link=True;options.default_dof_drive_mode=gymapi.DOF_MODE_NONE
                options.vhacd_enabled=True;options.vhacd_params.resolution=100000;options.vhacd_params.max_convex_hulls=128;options.vhacd_params.max_num_vertices_per_ch=64
                asset=self.gym.load_asset(self.sim,entries[index]['path'],'asset.urdf',options)
                if asset is None or self.gym.get_asset_rigid_body_count(asset)!=1 or self.gym.get_asset_dof_count(asset)!=0:raise ValueError('Invalid native object')
                cache[index]=asset
            self._box_assets.append(cache[index])


class NativeRsiAuditPlayer(ChairFTPlayer):
    @torch.no_grad()
    def run_eval(self):
        t=self.env.task;bank=t._audit_root;out=Path(t.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
        manifest=json.loads((bank/'manifest.json').read_text());names=manifest['body_names'];N=t.num_envs;entries=t._audit_assets;A=len(entries)
        assert N==2048 and t.num_agents==1 and t.num_objects==1
        shapes=collision_samples('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml',names)
        geometry=[NativeGeometry(e['path'],shapes) for e in entries]
        original_root=t._root_states.clone();original_dof=t._dof_state.clone();rows_by_asset=[torch.nonzero(t._audit_ids==i).flatten() for i in range(A)]
        steps=max(1,round(.5/t.sim_params.dt));short_steps=max(1,round(.1/t.sim_params.dt));report=[]
        head=names.index('head')
        def cpu(x):return x.detach().cpu().numpy().copy()
        print('RSI_PHYSICS_BEGIN',A,'assets',N,'envs',steps,'steps',flush=True)
        for skill in ['sit','climb']:
            data=[dict(np.load(bank/e['id']/(skill+'.npz'))) for e in entries]
            candidates=[np.flatnonzero(d['static_pass']) for d in data]
            metrics=[{key:np.full(len(d['clip']),np.nan) for key in ['root_displacement','root_speed','head_force','joint_excess','final_clearance','final_floor','goal_error']} for d in data]
            for m in metrics:m['short_pass']=np.zeros(len(m['root_displacement']),dtype=bool);m['hold_pass']=m['short_pass'].copy()
            rounds=max((math.ceil(len(c)/len(r)) for c,r in zip(candidates,rows_by_asset)),default=0)
            for block in range(rounds):
                pairs=[];rowlist=[];states=[];dofs=[];vels=[]
                for asset,(valid,rows) in enumerate(zip(candidates,rows_by_asset)):
                    selected=valid[block*len(rows):(block+1)*len(rows)];chosen=rows[:len(selected)]
                    if not len(selected):continue
                    pairs.extend((asset,int(index)) for index in selected);rowlist.append(chosen)
                    states.append(torch.tensor(data[asset]['root_state'][selected],device=t.device,dtype=torch.float32));dofs.append(torch.tensor(data[asset]['dof'][selected],device=t.device,dtype=torch.float32));vels.append(torch.tensor(data[asset]['dof_vel'][selected],device=t.device,dtype=torch.float32))
                if not pairs:continue
                rows=torch.cat(rowlist);state=torch.cat(states);joint=torch.cat(dofs);jvel=torch.cat(vels)
                t._root_states.copy_(original_root);t._dof_state.copy_(original_dof)
                t._humanoid_root_states[:,:,:2]=t._env_origins[:,None,:2]+15.;t._humanoid_root_states[:,:,7:13]=0
                t._box_states[:,:,:2]=t._env_origins[:,None,:2];t._box_states[:,:,2]=t._box_size[:,:,2]/2
                t._box_states[:,:,3:7]=torch.tensor([0.,0.,0.,1.],device=t.device);t._box_states[:,:,7:13]=0
                state[:,:3]+=t._env_origins[rows]
                t._humanoid_root_states[rows,0]=state;t._dof_pos[rows,0]=joint;t._dof_vel[rows,0]=jvel
                initial=state.clone();excess=torch.maximum(t.dof_limits_lower-joint,joint-t.dof_limits_upper).clamp_min(0).amax(-1)
                ok=torch.isfinite(state).all(-1)&torch.isfinite(joint).all(-1)&(excess<=.03)
                targets=t._dof_pos.contiguous().clone()
                t.gym.set_actor_root_state_tensor(t.sim,gymtorch.unwrap_tensor(t._root_states));t.gym.set_dof_state_tensor(t.sim,gymtorch.unwrap_tensor(t._dof_state));t.gym.set_dof_position_target_tensor(t.sim,gymtorch.unwrap_tensor(targets))
                max_head=torch.zeros(len(rows),device=t.device);max_disp=max_head.clone();short_ok=None
                for frame in range(steps):
                    t.gym.simulate(t.sim);t.gym.fetch_results(t.sim,True);t._refresh_sim_tensors()
                    current=t._humanoid_root_states[rows,0]
                    disp=(current[:,:3]-initial[:,:3]).norm(dim=-1);max_disp=torch.maximum(max_disp,disp)
                    max_head=torch.maximum(max_head,t._contact_forces[rows,0,head].norm(dim=-1))
                    finite=torch.isfinite(current).all(-1)&torch.isfinite(t._dof_state.reshape(N,1,-1)[rows,0]).all(-1)
                    ok&=finite&(max_head<=200)&((current[:,7:10]-initial[:,7:10]).norm(dim=-1)<=3.)
                    if frame+1==short_steps:short_ok=cpu(ok&(max_disp<=.15))
                final=t._humanoid_root_states[rows,0];passed=cpu(ok&(max_disp<=.30)&(final[:,7:10].norm(dim=-1)<=2.))
                current_body=cpu(t._rigid_body_pos[rows,0])-cpu(t._env_origins[rows])[:,None]
                current_rot=cpu(t._rigid_body_rot[rows,0]) if hasattr(t,'_rigid_body_rot') else cpu(t._rigid_body_state.view(N,1,t.num_bodies,13)[rows,0,:,3:7])
                finalroot=cpu(final[:,:3])-cpu(t._env_origins[rows]);disp=cpu(max_disp);speed=cpu(final[:,7:10].norm(dim=-1));headforce=cpu(max_head);joint_excess=cpu(excess)
                offset=0
                for asset,(valid,asset_rows) in enumerate(zip(candidates,rows_by_asset)):
                    selected=valid[block*len(asset_rows):(block+1)*len(asset_rows)];count=len(selected)
                    if not count:continue
                    sel=slice(offset,offset+count);e=entries[asset]
                    clearance,floor=geometry[asset].check(current_body[sel],current_rot[sel],e['bbox'][2]/2)
                    target=np.r_[e['seat'][:2],e['bbox'][2]/2+e['seat'][2]+(.1 if skill=='sit' else .94)]
                    error=np.linalg.norm(finalroot[sel]-target,axis=-1)
                    stable=passed[sel]&(clearance>=-.015)&(floor>=-.02)&(~data[asset]['phase'][selected]|(error<=.25))
                    values=dict(root_displacement=disp[sel],root_speed=speed[sel],head_force=headforce[sel],joint_excess=joint_excess[sel],final_clearance=clearance,final_floor=floor,goal_error=error,short_pass=short_ok[sel],hold_pass=stable)
                    for key,value in values.items():metrics[asset][key][selected]=value
                    offset+=count
                print('RSI_PHYSICS',skill,block+1,'/',rounds,'tested',len(rows),'short',int(short_ok.sum()),'hold',int(sum(m['hold_pass'].sum() for m in metrics)),flush=True)
            for asset,(e,d,m) in enumerate(zip(entries,data,metrics)):
                usable=d['static_pass']&m['short_pass']&m['hold_pass']&d['phase']
                folder=out/'audited_bank'/e['id'];folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/(skill+'.npz'),**d,**{'physics_'+k:v for k,v in m.items()},usable=usable)
                accepted={k:v[usable] for k,v in d.items() if len(v)==len(usable)}
                np.savez_compressed(folder/(skill+'_accepted.npz'),**accepted)
                report.append(dict(asset=e['id'],category=e['category'],skill=skill,total=len(usable),static_pass=int(d['static_pass'].sum()),short_pass=int(m['short_pass'].sum()),hold_pass=int(m['hold_pass'].sum()),usable_phase=int(usable.sum()),mesh_watertight=geometry[asset].watertight))
            (out/'physics_report.json').write_text(json.dumps(report,indent=2))
        (out/'manifest.json').write_text(json.dumps(dict(source_bank=str(bank),assets=entries,protocol='native_mesh_RSI_candidate_audit_v1',optimizer_steps=0,geometry_limitations='sampled capsules/feet with parity on potentially open render meshes; PhysX VHACD collision checked by rollout, not an exact volumetric collision proof',short_seconds=short_steps*t.sim_params.dt,hold_seconds=steps*t.sim_params.dt,thresholds=dict(head_force=200,root_displacement=.30,root_speed=2.,geometry_depth=.015,floor=.02)),indent=2))
        lines=['# Native chair/table RSI audit','', '| Category/task | Assets | Assets with usable phase | Accepted phase candidates |','|---|---:|---:|---:|']
        for cat in ['Chair','Table']:
            for skill in ['sit','climb']:
                r=[x for x in report if x['category']==cat and x['skill']==skill];lines.append(f'| {cat}/{skill} | {len(r)} | {sum(x["usable_phase"]>0 for x in r)} | {sum(x["usable_phase"] for x in r)} |')
        lines+=['','No training started. Accepted means sampled geometry and 0.1s/0.5s static-PD tests passed, not guaranteed policy stability. Empty assets require further retargeting/IK. Native meshes may be open; geometric containment is approximate. Only training assets are used; held-out assets have not been screened. AMP history and two-person scene reset integration are not enabled by this audit.']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n');(out/'completed.json').write_text(json.dumps(dict(rc=0,optimizer_steps=0)));print('RSI_AUDIT_DONE',out,flush=True)
