"""Offline mesh-contact arm IK for B20 carry RSI; source motions are read-only."""
from pathlib import Path
import argparse, json, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import open3d as o3d
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tokenhsi')]
from joint_carry.single_source import load_single
from rsi_native.repair import NativeFK
from rsi_native.geometry import NativeGeometry, collision_samples


def build(output, limit=0, grip_x=.50, grip_y=.65, backshift=.16):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    fk=NativeFK(ROOT/'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml')
    shapes=collision_samples(ROOT/'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml',fk.names)
    src=load_single('carryWith');arms=np.r_[np.arange(6,10),np.arange(10,14)]
    # Native 32DOF layout: torso0:3,head3:6,right arm6:10,left arm10:14.
    hands=np.array([fk.names.index('right_hand'),fk.names.index('left_hand')])
    feet=np.array([fk.names.index('right_foot'),fk.names.index('left_foot')])
    entries=[];report=[]
    for split in ('train','test'):
        for path in sorted((ROOT/'tokenhsi/data/dataset_dynamic_objects'/split/'Table_Circle').glob('*/config.json')):
            meta=json.loads(path.read_text());entries.append(dict(id=path.parent.name,path=str(path.parent.resolve()),split=split,**meta))
    if limit:entries=entries[:limit]
    for e in entries:
        geo=NativeGeometry(e['path'],shapes);rows=[]
        for frame in [24,36,48,60,72,84,96,108]:
            indices=np.arange(frame-9,frame+1)
            rotation=R.from_quat(src['box_rotation_xyzw'][frame]).inv()
            translation=-rotation.apply(src['box_position'][frame]);translation[2]=0 # retain world height
            roots=np.c_[src['root_position'][indices,0],src['root_rotation_xyzw'][indices,0],src['body_linear_velocity'][indices,0,0],src['body_angular_velocity'][indices,0,0]].copy()
            roots[:,:3]=rotation.apply(roots[:,:3])+translation
            roots[:,3:7]=(rotation*R.from_quat(roots[:,3:7])).as_quat()
            roots[:,7:10]=rotation.apply(roots[:,7:10]);roots[:,10:13]=rotation.apply(roots[:,10:13])
            joints=np.clip(src['dof_position'][indices,0],fk.lower,fk.upper)
            # Keep walking feet/root height; move the person slightly behind the wider tabletop.
            roots[:,1]-=backshift
            pos,quat=fk.forward(roots,joints)
            ids,points,radii=shapes
            footmask=np.isin(ids,feet)
            cloud=R.from_quat(quat[-1,ids[footmask]]).apply(points[footmask])+pos[-1,ids[footmask]]
            lift=max(0.,.003-float((cloud[:,2]-radii[footmask]).min()))
            roots[:,2]+=lift;pos,quat=fk.forward(roots,joints)
            # Preserve source carry height at the grip; table centre moves DOWN by half its height.
            top=float(src['body_position'][frame,0,hands,2].mean())+lift
            object_z=top-e['bbox'][2]/2+.065
            rx,ry=np.asarray(e['bbox'][:2])/2
            query=np.array([[rx*grip_x,-ry*grip_y,e['bbox'][2]],[-rx*grip_x,-ry*grip_y,e['bbox'][2]]],np.float32)
            rays=np.c_[query,np.tile([0.,0.,-1.],(2,1))].astype(np.float32)
            hit=geo.scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
            if not np.isfinite(hit).all():continue
            top_points=query.copy();top_points[:,2]-=hit
            rays[:,:3]=top_points;rays[:,2]-=.001
            second=geo.scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
            thickness=np.where(np.isfinite(second)&(second<.12),second+.001,.025)
            anchors=top_points.copy();anchors[:,2]-=thickness
            # Sphere hands support the underside, rather than relying on side friction.
            targets=anchors.copy();targets[:,2]+=object_z-.032
            base=joints[-1].copy();root=roots[-1:];target=targets.copy()
            def residual(x):
                q=base.copy();q[arms]=x;p,_=fk.forward(root,q[None])
                return np.r_[((p[0,hands]-target)/.003).ravel(),(x-base[arms])*.2]
            fit=least_squares(residual,np.clip(base[arms],fk.lower[arms]+1e-6,fk.upper[arms]-1e-6),bounds=(fk.lower[arms]+1e-7,fk.upper[arms]-1e-7),max_nfev=100)
            correction=fit.x-base[arms];raw=joints.copy();joints[:,arms]+=correction
            joints=np.clip(joints,fk.lower,fk.upper)
            # Constant correction across all ten frames, with velocities recomputed in native local joints.
            jv=np.gradient(joints,1/30,axis=0)
            body=fk.state(roots,joints,jv)
            obj=np.zeros((10,13));obj[:,3:7]=(rotation*R.from_quat(src['box_rotation_xyzw'][indices])).as_quat()
            obj[:,:3]=rotation.apply(src['box_position'][indices])+translation
            obj[:,2]+=object_z-src['box_position'][frame,2]
            obj[:,7:10]=rotation.apply(src['box_linear_velocity'][indices]);obj[:,10:13]=rotation.apply(src['box_angular_velocity'][indices])
            # Evaluate history against the MOVING table in its local frame.
            oq=R.from_quat(obj[:,3:7]).inv()
            localpos=np.stack([oq[i].apply(body[i,:,:3]-obj[i,:3]) for i in range(10)])
            localq=(R.from_quat(np.repeat(obj[:,3:7],len(fk.names),axis=0)).inv()*R.from_quat(body[:,:,3:7].reshape(-1,4))).as_quat().reshape(10,len(fk.names),4)
            depth,floor=geo.check(localpos,localq,0.)
            if not geo.watertight:
                # Open triangle meshes have undefined occupancy signs. Check two-sided
                # surface/capsule clearance here and require actual VHACD physics below.
                localcloud=np.stack([R.from_quat(localq[i,ids]).apply(points)+localpos[i,ids] for i in range(10)])
                distance=geo.scene.compute_distance(o3d.core.Tensor(localcloud.astype(np.float32).reshape(-1,3))).numpy().reshape(10,-1)
                depth=(distance-radii[None]).min(-1)
            # geo.check's floor is object-local, so compute actual human floor independently.
            world=np.stack([R.from_quat(body[i,ids,3:7]).apply(points)+body[i,ids,:3] for i in range(10)])
            ground=(world[:,:,2]-radii).min()
            hand_error=float(np.linalg.norm(body[-1,hands,:3]-target,axis=-1).max())
            distances=geo.scene.compute_distance(o3d.core.Tensor(localpos[:,hands].astype(np.float32).reshape(-1,3))).numpy().reshape(10,2)
            good=bool(hand_error<=.02 and depth.min()>=-.02 and ground>=-.025 and object_z-e['bbox'][2]/2>.1 and distances.max()<=.065)
            row=dict(frame=frame,hand_error=hand_error,clearance=float(depth.min()),floor=float(ground),mesh_watertight=geo.watertight,history_hand_surface_max=float(distances.max()),static_pass=good,arm_delta=float(abs(correction).max()))
            report.append(dict(asset=e['id'],split=e['split'],**row))
            if good:rows.append(dict(root_state=roots[-1],dof=joints[-1],dof_vel=jv[-1],body_state=body[-1],object_state=obj[-1],history_root=roots[:-1][::-1],history_dof=joints[:-1][::-1],history_dof_vel=jv[:-1][::-1],history_body=body[:-1][::-1],anchors=anchors,frame=np.asarray(frame),hand_error=np.asarray(hand_error)))
        folder=output/e['id'];folder.mkdir()
        if rows:np.savez_compressed(folder/'carryWith.npz',**{k:np.stack([v[k] for v in rows]).astype(np.float32) for k in rows[0]},physics_pass=np.zeros(len(rows),bool))
        print('TABLE_RSI',e['split'],e['id'],'static',len(rows),'of8',flush=True)
        (output/'report.json').write_text(json.dumps(report,indent=2))
    (output/'manifest.json').write_text(json.dumps(dict(assets=entries,body_names=fk.names,history_dt=1/30,history_steps=9,source='B20 carryWith; constant per-window native arm IK',physics_validated=False,grip_x=grip_x,grip_y=grip_y,backshift=backshift),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--grip-x',type=float,default=.50);p.add_argument('--grip-y',type=float,default=.65);p.add_argument('--backshift',type=float,default=.16);a=p.parse_args();build(a.output,a.limit,a.grip_x,a.grip_y,a.backshift)
