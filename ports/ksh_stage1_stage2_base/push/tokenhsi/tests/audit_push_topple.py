"""Isolated early-training replay; never attaches to or edits a live trainer.

Use run.py training arguments and a check output. This records episode type and
control-step states around the first >45-degree box tilt. Contact forces are net
body forces, not pair-specific collision reports.
"""
from isaacgym import gymapi
from collections import deque
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
import run as entry
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
from utils.stage1_push import box_tilt_degrees
from utils.door_reference import rotate

RESET = HumanoidMAStage1Push._reset_envs
PHYSICS = HumanoidMAStage1Push._physics_step
AUDIT = None


class Audit:
    def __init__(self, task):
        self.task = task
        self.out = Path(task.cfg['args'].output_path)
        self.out.mkdir(parents=True, exist_ok=True)
        self.age = torch.zeros(task.num_envs, 2, device=task.device, dtype=torch.long)
        self.episode = torch.zeros_like(self.age)
        self.seen = torch.zeros(task.num_envs, 2, 4, device=task.device, dtype=torch.bool)
        self.ring = deque(maxlen=16)
        self.pending = []
        self.cases = []
        self.steps = 0
        self.clearance = dict(enabled=os.environ.get('PUSH_AUDIT_CLEAR_FEET')=='1',shifted=0,max_shift_m=0.)
        self.geometry_stats = dict(checked_agent_resets=0,invalid_collider_resets=0,max_remaining_floor_shift_m=0.,loco_min_distance=float('inf'),loco_max_distance=0.)
        self.foot_vertices = {}
        self.initial = dict(step=0, age=torch.zeros_like(self.age),
                            box=task._box_states[:, :2].clone(),root=task._humanoid_root_states.clone(),
                            body=task._kinematic_humanoid_rigid_body_states[..., :3].clone(),
                            body_rot=task._kinematic_humanoid_rigid_body_states[..., 3:7].clone(),
                            force=torch.zeros_like(task._contact_forces),
                            box_force=torch.zeros_like(task._ontop_box_contact[:, :2]))
        self.names = task.gym.get_actor_rigid_body_names(task.envs[0], task.gym.get_actor_handle(task.envs[0], 0))
        xml = Path(task.cfg['env']['asset']['assetRoot'])/task.cfg['env']['asset']['assetFileName']
        tree = ET.parse(xml).getroot()
        for mesh in tree.findall('./asset/mesh'):
            if mesh.get('name') in ('left_foot_mesh','right_foot_mesh'):
                vertices = [[float(x) for x in line.split()[1:4]] for line in (xml.parent/mesh.get('file')).read_text().splitlines() if line.startswith('v ')]
                self.foot_vertices[mesh.get('name').replace('_mesh','')] = torch.tensor(vertices,device=task.device)
        self.summary = {kind: dict(episodes=0, agent_steps=0, first_crossings={str(x): 0 for x in (5,15,45,90)},
                                  first_45_before_step10=0, first_45_root_below_05=0)
                        for kind in ('loco','rsi')}

    def reset(self, ids):
        self.age[ids] = 0
        self.episode[ids] += 1
        self.seen[ids] = False
        self.initial['box'][ids] = self.task._box_states[ids, :2]
        self.initial['root'][ids] = self.task._humanoid_root_states[ids]
        self.initial['body'][ids] = self.task._kinematic_humanoid_rigid_body_states[ids, ..., :3]
        self.initial['body_rot'][ids] = self.task._kinematic_humanoid_rigid_body_states[ids, ..., 3:7]
        geometry=self.task._push_reset_geometry()
        if geometry is not None:
            pose=self.task._kinematic_humanoid_rigid_body_states[ids].reshape(len(ids)*2,-1,13)
            bad=geometry.invalid(pose[...,:3],pose[...,3:7],self.task._box_states[ids,:2].reshape(-1,13),self.task._box_size[ids,:2].reshape(-1,3))
            self.geometry_stats['checked_agent_resets']+=len(bad)
            self.geometry_stats['invalid_collider_resets']+=int(bad.sum())
            assert not bad.any()
            floor=geometry.floor_shift(pose[...,:3],pose[...,3:7])
            self.geometry_stats['max_remaining_floor_shift_m']=max(self.geometry_stats['max_remaining_floor_shift_m'],floor.max().item())
            assert floor.max()<1e-4
        mask=~self.task._push_rsi_interaction[ids]
        distance=(self.task._humanoid_root_states[ids,:,:2]-self.task._box_states[ids,:2,:2]).norm(dim=-1)[mask]
        if distance.numel():
            lo,hi=self.task.cfg['env']['pushTask']['loco_distance_range_m']
            assert distance.min()>=lo-.001 and distance.max()<=hi+.001
            self.geometry_stats['loco_min_distance']=min(self.geometry_stats['loco_min_distance'],distance.min().item())
            self.geometry_stats['loco_max_distance']=max(self.geometry_stats['loco_max_distance'],distance.max().item())
        for kind, mask in (('rsi', self.task._push_rsi_interaction[ids]), ('loco', ~self.task._push_rsi_interaction[ids])):
            self.summary[kind]['episodes'] += int(mask.sum())

    def clear_loco_feet(self, ids):
        """Counterfactual only: translate the complete human; preserve the 1--2m law."""
        if not self.clearance['enabled']:
            return
        t = self.task
        r,a = (~t._push_rsi_interaction[ids]).nonzero(as_tuple=True)
        r = ids[r]
        if not len(r):
            return
        states=t._kinematic_humanoid_rigid_body_states[r,a].clone()
        box=t._box_states[r,a];half=t._box_size[r,a]/2
        inv=box[:,3:7].clone();inv[:,:3]*=-1
        shift=torch.zeros(len(r),device=t.device)
        for name,vertices in self.foot_vertices.items():
            i=self.names.index(name)
            q=states[:,i,None,3:7].expand(-1,len(vertices),-1)
            world=rotate(q,vertices[None].expand(len(r),-1,-1))+states[:,i,None,:3]
            local=rotate(inv[:,None].expand_as(q),world-box[:,None,:3])
            near=(local[...,0].abs()<half[:,None,0]+.02)&(local[...,2].abs()<half[:,None,2]+.02)
            shift=torch.maximum(shift,torch.where(near,local[...,1]+half[:,None,1]+.02,0.).amax(-1).clamp_min(0))
        select=shift>1e-6
        if not select.any():
            return
        r,a,states,shift=r[select],a[select],states[select],shift[select]
        box=t._box_states[r,a]
        forward=rotate(box[:,3:7],box.new_tensor([0.,1.,0.]).expand(len(r),3))
        body=states[...,:3]-forward[:,None]*shift[:,None,None]
        root_distance=(body[:,0,:2]-box[:,:2]).norm(dim=-1)
        assert root_distance.min()>=1.-1e-4 and root_distance.max()<=2.+1e-4
        t._set_pose(r,a,body,states[...,3:7],states[...,7:10],states[...,10:13],t._dof_pos[r,a].clone(),t._dof_vel[r,a].clone())
        t._reset_env_tensors(ids);t._refresh_sim_tensors();t._reset_relation_history(ids);t._compute_observations(ids);t._init_amp_obs(ids)
        self.clearance['shifted']+=len(r)
        self.clearance['max_shift_m']=max(self.clearance['max_shift_m'],shift.max().item())

    def sample(self):
        t = self.task
        self.steps += 1
        self.age += 1
        snap = dict(step=self.steps, episode=self.episode.clone(), age=self.age.clone(),
                    box=t._box_states[:, :2].clone(), root=t._humanoid_root_states.clone(),
                    body=t._rigid_body_pos.clone(), force=t._contact_forces.clone(),
                    box_force=t._ontop_box_contact[:, :2].clone())
        self.ring.append(snap)
        for case in list(self.pending):
            e,a = case['env'],case['owner']
            if int(self.episode[e,a]) != case['episode']:
                case['ended_by_reset'] = True
                self.pending.remove(case)
            else:
                case['trace'].append(self.row(snap,e,a))
                if self.steps-case['event_step'] >= 16:
                    self.pending.remove(case)
        tilt = box_tilt_degrees(t._box_states[:, :2, 3:7])
        thresholds = tilt.new_tensor([5.,15.,45.,90.])
        crossings = (tilt[...,None] > thresholds) & ~self.seen
        self.seen |= tilt[...,None] > thresholds
        for kind,mask in (('rsi',t._push_rsi_interaction),('loco',~t._push_rsi_interaction)):
            s = self.summary[kind]
            s['agent_steps'] += int(mask.sum())
            for i,angle in enumerate((5,15,45,90)):
                s['first_crossings'][str(angle)] += int((crossings[...,i] & mask).sum())
            top = crossings[...,2] & mask
            s['first_45_before_step10'] += int((top & (self.age<=10)).sum())
            s['first_45_root_below_05'] += int((top & (t._humanoid_root_states[...,2]<.5)).sum())
        for e,a in crossings[...,2].nonzero().tolist():
            kind = 'rsi' if bool(t._push_rsi_interaction[e,a]) else 'loco'
            # Retain examples from each reset family, without copying all frames to CPU.
            if sum(c['kind']==kind for c in self.cases) >= 8:
                continue
            ep = int(self.episode[e,a])
            case = dict(env=e,owner=a,episode=ep,kind=kind,event_step=self.steps,
                        age=int(self.age[e,a]),box_size=t._box_size[e,a].cpu().tolist(),
                        motion_id=int(t._push_rsi_samples['motion_id'][e,a]),
                        motion_time=float(t._push_rsi_samples['time'][e,a]),
                        late=bool(t._push_rsi_samples['late_requested'][e,a]),trace=[])
            case['trace'] = [self.row(self.initial,e,a)] + [self.row(s,e,a) for s in self.ring if int(s['episode'][e,a])==ep]
            self.cases.append(case)
            self.pending.append(case)

    def row(self,s,e,a):
        t = self.task
        row = dict(step=s['step'],age=int(s['age'][e,a]),
                    tilt_deg=float(box_tilt_degrees(s['box'][e,a,3:7])),
                    box=s['box'][e,a].cpu().tolist(),root=s['root'][e,a].cpu().tolist(),
                    body=s['body'][e,a].cpu().tolist(),force=s['force'][e,a].cpu().tolist(),
                    box_force=s['box_force'][e,a].cpu().tolist(),
                    hands=s['body'][e,a,t._key_body_ids[:2]].cpu().tolist())
        if 'body_rot' in s:
            row['body_rot'] = s['body_rot'][e,a].cpu().tolist()
        return row

    def save(self):
        report = dict(method='isolated scratch training replay from the same saved environment and seed; not the live current policy',
                      control_dt=self.task.dt,control_steps=self.steps,body_names=self.names,
                      summary=self.summary,cases=self.cases,
                      foot_clearance_counterfactual=self.clearance,
                      reset_geometry_verification=self.geometry_stats,
                      caveat='Body forces are net forces; proximity is needed to infer box contact. No causal intervention is performed.')
        (self.out/'topple_audit.json').write_text(json.dumps(report,indent=2)+'\n')
        print('TOPPLE_AUDIT',json.dumps(self.summary),flush=True)


def reset(task,ids):
    global AUDIT
    result = RESET(task,ids)
    if len(ids):
        if AUDIT is None:
            AUDIT = Audit(task)
        AUDIT.clear_loco_feet(ids)
        AUDIT.reset(ids)
    return result


def physics(task):
    result = PHYSICS(task)
    if AUDIT is not None:
        AUDIT.sample()
    return result


if __name__=='__main__':
    HumanoidMAStage1Push._reset_envs = reset
    HumanoidMAStage1Push._physics_step = physics
    try:
        entry.main()
    finally:
        if AUDIT is not None:
            AUDIT.save()
