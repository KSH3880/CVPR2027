"""Bounded object-aware leg IK. Offline candidates only; never alters live RSI banks."""
from pathlib import Path
import argparse
import json
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from rsi_native.repair import NativeFK
from rsi_native.geometry import NativeGeometry, collision_samples, rotate
from rsi_native.support import supported
from rsi_native.front_filter import front_mask
from rsi_native.foot_surface import FootSurface


class ContactIK:
    def __init__(self, fk, shapes, geometry, entry):
        self.fk, self.shapes, self.geometry, self.entry = fk, shapes, geometry, entry
        self.object_z = entry['bbox'][2] / 2
        # Hip(3), knee(1), ankle(3), on each leg. Upper body stays unchanged.
        self.indices = np.concatenate([np.arange(o, o+s) for name, (o,s) in zip(fk.names,fk.joints)
                                       if any(x in name for x in ('thigh','shin','foot'))])
        ids, points, radii = shapes
        self.foot_ids, self.foot_points = [], []
        for foot in (11,14):
            p = points[ids == foot]
            lo, hi = p.min(0), p.max(0)
            self.foot_ids.extend([foot]*4)
            self.foot_points.extend([[x,y,lo[2]] for x in (lo[0],hi[0]) for y in (lo[1],hi[1])])
        self.foot_ids, self.foot_points = np.array(self.foot_ids), np.array(self.foot_points)
        # Limit numerical geometry cost; final acceptance uses ALL original samples + edges.
        self.sample = np.arange(0,len(ids),3)

    def sole(self, body):
        return body[self.foot_ids,:3] + rotate(body[self.foot_ids,3:7],self.foot_points)

    def solve(self, root, joint, skill, max_nfev=65):
        root, joint = root.astype(float).copy(), joint.astype(float).copy()
        original = self.fk.state(root[None],joint[None],np.zeros((1,32)))[0]
        feet = self.sole(original)
        top = self.object_z + self.entry['seat'][2]
        target = feet.copy()
        if skill == 'climb':
            # Stand on the actual seat/top. Keep sole inside usable seat footprint.
            extent = np.maximum(np.asarray(self.entry['extent'])/2-.045,.025)
            center = np.asarray(self.entry['seat'][:2])
            target[:,:2] = np.clip(feet[:,:2],center-extent,center+extent)
            target[:,2] = top + .005
            contact = np.ones(8,dtype=bool)
        else:
            # Hanging legs on a high table are allowed. Only source-near-ground feet
            # get floor constraints; pelvis/object support is checked independently.
            contact = np.repeat([feet[:4,2].min()<.18, feet[4:,2].min()<.18],4)
            target[contact,2] = .005
        ids, points, radii = self.shapes
        subset = self.sample
        def decode(x):
            r = root.copy();r[:3]+=x[:3]
            q = joint.copy();q[self.indices]+=x[3:]
            pos, quat = self.fk.forward(r[None],q[None])
            return r,q,np.concatenate([pos[0],quat[0],np.zeros((len(pos[0]),6))],-1)
        def residual_batch(x):
            count=len(x);r=np.tile(root,(count,1));q=np.tile(joint,(count,1))
            r[:,:3]+=x[:,:3];q[:,self.indices]+=x[:,3:]
            pos,quat=self.fk.forward(r,q)
            sole=pos[:,self.foot_ids]+rotate(quat[:,self.foot_ids],self.foot_points[None])
            world=pos[:,ids[subset]]+rotate(quat[:,ids[subset]],points[None,subset])
            local=world.copy();local[...,2]-=self.object_z
            query=o3d.core.Tensor(local.reshape(-1,3).astype(np.float32))
            distance=self.geometry.scene.compute_distance(query).numpy().reshape(count,-1)
            inside=self.geometry.scene.compute_occupancy(query,nsamples=3).numpy().reshape(count,-1)>.5
            clearance=np.where(inside,-distance,distance)-radii[None,subset]
            collision=np.minimum(clearance+.002,0)*8
            floor=np.minimum(world[...,2]-radii[None,subset]-.002,0)*8
            return np.concatenate([((sole[:,contact]-target[None,contact])*5).reshape(count,-1),
                                   ((sole[:,~contact]-feet[None,~contact])*.15).reshape(count,-1),
                                   x[:,:3]*5.,x[:,3:]*.35,collision,floor],axis=1)
        def residual(x):return residual_batch(x[None])[0]
        lower = np.r_[[-.02,-.02,-.02],np.maximum(self.fk.lower[self.indices]-joint[self.indices],-.1)]
        upper = np.r_[[.02,.02,.02],np.minimum(self.fk.upper[self.indices]-joint[self.indices],.1)]
        # Native joint clamps can sit exactly on a bound; least_squares nudges inward.
        def jacobian(x):
            # Batch absolute finite differences: float32 geometry needs >=0.1mm.
            h=2e-4;steps=np.eye(len(x))*h
            values=residual_batch(np.concatenate([x[None]+steps,x[None]-steps]))
            return ((values[:len(x)]-values[len(x):])/(2*h)).T
        initial=np.clip(np.zeros(len(lower)),lower+1e-3,upper-1e-3)
        result = least_squares(residual,initial,bounds=(lower,upper),
                               jac=jacobian,max_nfev=max_nfev,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        r,q,b = decode(result.x)
        return r,q,b,dict(cost=float(result.cost),nfev=result.nfev,
                          foot_target_error=float(np.linalg.norm(self.sole(b)[contact]-target[contact],axis=-1).max()) if contact.any() else 0.,
                          joint_delta=float(np.abs(q-joint).max()),root_delta=float(np.linalg.norm(r[:3]-root[:3])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--per-skill',type=int,default=4)
    p.add_argument('--asset-limit',type=int,default=0);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    m=json.loads((a.source/'manifest.json').read_text())
    if a.asset_limit:m['assets']=m['assets'][:a.asset_limit]
    fk=NativeFK('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml')
    shapes=collision_samples('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml',fk.names)
    feet=FootSurface('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml',fk.names);report=[]
    for e in m['assets']:
        geo=NativeGeometry(e['path'],shapes);solver=ContactIK(fk,shapes,geo,e)
        folder=a.output/e['id'];folder.mkdir()
        for skill in ('sit','climb'):
            d=dict(np.load(a.source/'audited_bank'/e['id']/(skill+'.npz')))
            # Preserve original arrays for before/after review. Only failed phase is changed.
            for key in ('root_state','dof','dof_vel','body_state'):d['ik_before_'+key]=d[key].copy()
            d['ik_attempted']=np.zeros(len(d['clip']),bool);d['ik_accepted']=np.zeros(len(d['clip']),bool)
            d['ik_joint_delta']=np.zeros_like(d['dof']);d['ik_root_delta']=np.zeros((len(d['clip']),3),np.float32)
            failed=d['phase']&d['anchor_valid']&~(d['static_pass']&d['physics_short_pass']&d['physics_hold_pass'])
            if e['category']=='Chair':failed &= front_mask(d['root_state'][:,:2],e['seat'][:2],e['facing'])
            # Prioritize nearly feasible poses and diversify clips.
            pool=np.flatnonzero(failed);rank=sorted(pool,key=lambda i:(max(0,-float(d['clearance'][i])),int(d['frame'][i])))
            selected=[];seen=set()
            for i in rank:
                if int(d['clip'][i]) not in seen:selected.append(i);seen.add(int(d['clip'][i]))
                if len(selected)>=a.per_skill:break
            for i in selected:
                d['ik_attempted'][i]=True
                r,q,b,stats=solver.solve(d['root_state'][i],d['dof'][i],skill)
                depth,floor=geo.check(b[None,:,:3],b[None,:,3:7],solver.object_z)
                support,gap=supported(b[None],geo,solver.object_z);cross=feet.crossing(b[None],geo,solver.object_z)
                target=np.r_[e['seat'][:2],solver.object_z+e['seat'][2]+(.1 if skill=='sit' else .94)]
                goal=float(np.linalg.norm(r[:3]-target))
                front=e['category']!='Chair' or bool(front_mask(r[None,:2],e['seat'][:2],e['facing'])[0])
                ok=bool(depth[0]>=-.015 and floor[0]>=-.02 and support[0] and not cross[0] and goal<=.20 and front and stats['root_delta']<=.02 and stats['joint_delta']<=.100001)
                row=dict(asset=e['id'],category=e['category'],skill=skill,index=int(i),clip=int(d['clip'][i]),static_accepted=ok,
                         clearance=float(depth[0]),floor=float(floor[0]),supported=bool(support[0]),foot_crossing=bool(cross[0]),goal=goal,**stats)
                report.append(row);print('IK_CANDIDATE',json.dumps(row),flush=True)
                if ok:
                    delta=q-d['dof'][i];shift=r[:3]-d['root_state'][i,:3]
                    d['ik_joint_delta'][i]=delta;d['ik_root_delta'][i]=shift
                    d['root_state'][i]=r;d['dof'][i]=q
                    # Constant pose offset has zero derivative; preserve reference velocity.
                    d['body_state'][i]=fk.state(r[None],q[None],d['dof_vel'][i:i+1])[0]
                    d['transform'][i,4:]+=shift;d['correction'][i]+=shift
                    d['clearance'][i]=depth[0];d['floor_clearance'][i]=floor[0];d['static_pass'][i]=True
                    d['ik_accepted'][i]=True
            # Audit only new IK candidates, keeping original good bank completely separate.
            d['static_pass'] &= d['ik_accepted']
            d={k:v for k,v in d.items() if not k.startswith('physics_') and k!='usable'}
            np.savez_compressed(folder/(skill+'.npz'),**d)
        (a.output/'ik_report.json').write_text(json.dumps(report,indent=2))
    m.update(ik_mode='offline_bounded_leg_contact_v1',source_bank=str(a.source.resolve()),
             training_enabled=False,ik_history='constant bounded joint offset on reference window; clamp + recompute velocities/FK; not yet integrated')
    (a.output/'manifest.json').write_text(json.dumps(m,indent=2))
    print('IK_DONE',json.dumps(dict(attempted=len(report),static_accepted=sum(x['static_accepted'] for x in report))),flush=True)
if __name__=='__main__':main()
