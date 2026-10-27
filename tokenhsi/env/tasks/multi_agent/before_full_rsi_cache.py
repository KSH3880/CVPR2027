"""Size-screened Stage1 frames for every solo bank and possible BEFORE binding."""
import gzip
import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import torch
from env.tasks.multi_agent.stage2_solo_rsi_cache import Stage2SoloRsiCache
from utils.before_full_rsi import SKILLS
from utils.size_rsi import SCREEN, size_key, late_frames, interaction_phase, screen_states


class BeforeFullRsiCache(Stage2SoloRsiCache):
    def __init__(self, env):
        self.env = env
        self.spec = dict(env.cfg['env']['sizeAwareRsi'])
        self.strict_geometry = True
        digest = hashlib.sha256()
        digest.update(json.dumps(dict(version=1, spec=self.spec, sim=env.cfg['sim'],
            dt=env.sim_params.dt, screen=SCREEN, density=100.,
            physics={k:env.cfg['env'][k] for k in ('asset','plane','powerScale','enableSelfCollisionDetection')}), sort_keys=True).encode())
        asset = env.cfg['env']['asset']
        digest.update((Path(asset['assetRoot'])/asset['assetFileName']).read_bytes())
        for name in ('loco','sit','climb','pickUp','carryWith','putDown'):
            lib = env._motion_lib[name]
            for v in (lib.gts,lib.grs,lib.lrs,lib.grvs,lib.gravs,lib.dvs,lib._motion_dt,lib._motion_weights):
                digest.update(v.cpu().numpy().tobytes())
            if hasattr(lib,'obj_gts'):
                digest.update(lib.obj_gts.cpu().numpy().tobytes()); digest.update(lib.obj_grs.cpu().numpy().tobytes())
        for path in (Path(__file__),Path('tokenhsi/utils/size_rsi.py'),Path('tokenhsi/utils/stage1_v2.py')):
            digest.update(path.read_bytes())
        output = Path(env.cfg['args'].output_path)
        root = output/'full_rsi_cache' if 'output_etc/check' in output.as_posix() else Path('output/rsi_cache_before_full')
        self.path = root/(digest.hexdigest()[:24]+'.pkl.gz')
        self.entries = {}
        if self.path.exists():
            with gzip.open(self.path,'rb') as f:self.entries=pickle.load(f)
        self.sizes = env._box_size.cpu().tolist()
        self._build_all()
        self._prepare_sampling()
        summary={kind:dict(profiles=sum(k[0]==kind for k in self.entries),
            empty=sum(not any(mask.any() for mask,phase in v) for k,v in self.entries.items() if k[0]==kind))
            for kind in self.pools}
        self.path.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('[BEFORE full RSI] ready',json.dumps(summary),flush=True)

    def _build_all(self):
        env=self.env; m=env.num_agents
        ranges=env.cfg['env']['box']['build']['taskSizeRanges']
        work={k:{} for k in ('sit','climb','pickUp','carryWith','putDown','putDownOnTop')}
        for row,sizes in enumerate(self.sizes):
            for owner in range(m):
                for kind,source in [('sit',owner),('sit',m+owner),('climb',owner),('climb',2*m+owner),
                                    ('pickUp',owner),('carryWith',owner),('putDown',owner)]:
                    if kind in ('sit','climb') and not all(lo-1e-6<=x<=hi+1e-6 for x,(lo,hi) in zip(sizes[source],ranges[kind.upper()])):continue
                    key=(kind,size_key(sizes[source]))
                    if key not in self.entries:work[kind].setdefault(key,[]).append((row,source,-1))
                for support in [3*m+owner]+[x for x in range(m) if x!=owner]:
                    key=('putDownOnTop',size_key(sizes[owner]),size_key(sizes[support]))
                    if key not in self.entries:work['putDownOnTop'].setdefault(key,[]).append((row,owner,support))
        print('[BEFORE full RSI] screening missing profiles',sum(map(len,work.values())),flush=True)
        changed=any(work.values())
        roots,dofs=env._root_states.clone(),env._dof_state.clone()
        try:
            for kind,remaining in work.items():
                while remaining:
                    batch={};used=set();self._screen_binding={}
                    for key,options in sorted(remaining.items(), key=lambda item: len(item[1])):
                        candidate=next((item for item in options if item[0] not in used), None)
                        if candidate is None:continue
                        row,source,support=candidate
                        used.add(row);batch[key]=row;self._screen_binding[row]=(source,support)
                        del remaining[key]
                    self._screen_profiles(kind,0,batch,roots,dofs)
                    self.path.parent.mkdir(parents=True,exist_ok=True)
                    tmp=self.path.with_suffix('.tmp.'+str(os.getpid()))
                    with gzip.open(tmp,'wb') as f:pickle.dump(self.entries,f,pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp,self.path)
        finally:
            env._root_states.copy_(roots);env._dof_state.copy_(dofs)
            if changed:
                from isaacgym import gymtorch
                env.gym.set_actor_root_state_tensor(env.sim,gymtorch.unwrap_tensor(env._root_states))
                env.gym.set_dof_state_tensor(env.sim,gymtorch.unwrap_tensor(env._dof_state))
                env.gym.set_dof_position_target_tensor(env.sim,gymtorch.unwrap_tensor(env._dof_pos.contiguous()))
                env._refresh_sim_tensors()

    def _prepare_sampling(self):
        self.pools={};self.profile_index={}
        for kind in ('sit','climb','pickUp','carryWith','putDown','putDownOnTop'):
            lib=self.env._motion_lib['putDown' if kind=='putDownOnTop' else kind]
            keys=[k for k in self.entries if k[0]==kind]
            # Sentinel profile zero represents an unavailable size/binding.
            starts=torch.zeros(len(keys)+1,lib.num_motions(),2,dtype=torch.long)
            counts=torch.zeros_like(starts);flat=[];total=0
            for i,key in enumerate(keys,1):
                self.profile_index[key]=i
                for clip,(mask,phase) in enumerate(self.entries[key]):
                    valid=torch.from_numpy(mask);late=late_frames(valid,torch.from_numpy(phase));all_frames=valid.nonzero().flatten()
                    if not len(late):late=all_frames
                    for j,frames in enumerate((all_frames,late)):
                        starts[i,clip,j]=total;counts[i,clip,j]=len(frames);flat.append(frames);total+=len(frames)
            weights=(counts[...,0]>0)*lib._motion_weights.cpu()
            self.pools[kind]=(starts.to(self.env.device),counts.to(self.env.device),
                (torch.cat(flat) if flat else torch.zeros(0,dtype=torch.long)).to(self.env.device),weights.to(self.env.device))
        # Binding lookup uses physical asset IDs, never assumes token order or role.
        n,m,p=self.env.num_envs,self.env.num_agents,self.env.num_physical_objects
        self.source_ids={}
        for kind in ('sit','climb','pickUp','carryWith','putDown'):
            self.source_ids[kind]=torch.tensor([[self.profile_index.get((kind,size_key(s)),0) for s in row] for row in self.sizes],device=self.env.device)
        self.top_ids=torch.zeros(n,m,m+1,dtype=torch.long,device=self.env.device)
        data=[[[self.profile_index.get(('putDownOnTop',size_key(row[a]),size_key(row[b if b<m else 3*m+a])),0)
                for b in range(m+1)] for a in range(m)] for row in self.sizes]
        self.top_ids.copy_(torch.tensor(data,device=self.env.device))

    def profile(self,kind,rows,owners):
        if kind=='putDownOnTop':
            pre=self.env._before_predecessor[rows,owners]
            return self.top_ids[rows,owners,torch.where(pre>=0,pre,self.env.num_agents)]
        source=self.env._scenario_physical_object(rows,owners)
        return self.source_ids[kind][rows,source]

    def availability(self,rows,owners,templates):
        result=torch.zeros(len(rows),len(SKILLS),device=rows.device,dtype=torch.bool);result[:,0]=True
        for j,skill in enumerate(SKILLS):
            if skill not in ('sit','climb','pickUp','carryWith','putDown'):continue
            for kind in (('putDown','putDownOnTop') if skill=='putDown' else (skill,)):
                selected=(templates==4) if kind=='putDownOnTop' else (templates!=4) if skill=='putDown' else torch.ones_like(rows,dtype=torch.bool)
                ids=self.profile(kind,rows,owners)
                result[selected,j]=self.pools[kind][3][ids[selected]].sum(-1)>0
        return result

    def sample(self,skill,rows,owners,templates):
        lib=self.env._motion_lib[skill]
        if skill=='loco':
            clips=lib.sample_motions(len(rows));return clips,lib.sample_time_rsi(clips)
        clips=torch.empty_like(rows);times=torch.empty(len(rows),device=rows.device)
        for kind in (('putDown','putDownOnTop') if skill=='putDown' else (skill,)):
            sel=(templates==4) if kind=='putDownOnTop' else (templates!=4) if skill=='putDown' else torch.ones_like(rows,dtype=torch.bool)
            if not sel.any():continue
            ids=self.profile(kind,rows[sel],owners[sel]);starts,counts,frames,weights=self.pools[kind]
            clip=torch.multinomial(weights[ids],1).flatten()
            late=(torch.rand(len(ids),device=rows.device)<self.spec['lateProbability']).long()
            count=counts[ids,clip,late]
            frame=frames[starts[ids,clip,late]+(torch.rand(len(ids),device=rows.device)*count).long()]
            clips[sel]=clip;times[sel]=frame*lib._motion_dt[clip]
            # Count the branch selection, not whether a full-pool sample happened to be late.
            if hasattr(self.env,'_full_rsi_frame_counts'):
                self.env._full_rsi_frame_counts += torch.stack(((late==0).sum(),(late==1).sum()))
        return clips,times

    def _screen_profiles(self, kind, owner, profiles, original_roots, original_dofs):
        from isaacgym import gymtorch
        env = self.env
        strict = self.spec.get('geometryFilter') == 'body_points_v1'
        if strict:
            from utils.stage1_v2 import pose_geometry_valid, sampled_phase
        skill = 'putDown' if kind == 'putDownOnTop' else kind
        lib = env._motion_lib[skill]
        rows = torch.tensor(list(profiles.values()), device=env.device)
        keys = list(profiles)
        source = torch.tensor([self._screen_binding[int(r)][0] for r in rows], device=env.device)
        support = torch.tensor([self._screen_binding[int(r)][1] for r in rows], device=env.device)
        sizes = env._box_size[rows, source]
        offset = env._agent_spawn_offsets[owner] + env._env_origins[rows]
        records = [[] for _ in keys]
        steps = math.ceil(SCREEN['seconds'] / env.sim_params.dt)
        for clip in range(lib.num_motions()):
            frames = int(lib._motion_num_frames[clip])
            ids = torch.full((frames,), clip, device=env.device, dtype=torch.long)
            times = torch.arange(frames, device=env.device) * lib._motion_dt[clip]
            root, rot, dofs, vel, ang, dof_vel, feet = lib.get_motion_state(ids, times)
            if strict:
                bodies = lib.get_motion_state_max(ids, times)[0]
            if skill in ('sit', 'climb'):
                pos, quat = lib.get_obj_motion_state_single_frame(ids[:1])
                box_pos = pos.expand(frames, -1)
                box_rot = quat.expand(frames, -1)
                # Phase is tied to the reference box, not the random training height.
                top = float(pos[0, 2]) * 2
                reference_size = torch.tensor([.5, .5, top], device=env.device)
                phase = interaction_phase(skill, root, feet[:, -2:], pos[0], reference_size)
            else:
                box_pos, box_rot = lib.get_obj_motion_state(ids, times)
                phase = torch.ones(frames, dtype=torch.bool, device=env.device)
            phases = (sampled_phase(skill, root, feet[:, -2:], box_pos[0, :2], sizes)
                      if strict and skill in ('sit', 'climb')
                      else phase[None].expand(len(rows), -1))
            end = lib.get_obj_motion_state(ids[:1], lib.get_motion_length(ids[:1]))[0] if kind == 'putDownOnTop' else None
            valid = torch.zeros(len(rows), frames, dtype=torch.bool, device=env.device)
            for frame in range(frames):
                env._root_states.copy_(original_roots)
                env._dof_state.copy_(original_dofs)
                env._humanoid_root_states[..., :2] = env._env_origins[:, None, :2] + 15.
                env._humanoid_root_states[..., 0] += torch.arange(env.num_agents, device=env.device)*3
                env._humanoid_root_states[..., 7:13] = 0.
                env._box_states[..., :2] = env._env_origins[:, None, :2] - 15.
                env._box_states[..., 0] -= torch.arange(env.num_physical_objects, device=env.device)*2
                env._box_states[..., 2] = env._box_size[..., 2]/2
                env._box_states[..., 3:7] = torch.tensor([0., 0., 0., 1.], device=env.device)
                env._box_states[..., 7:13] = 0.
                env._set_env_state(rows, torch.full_like(rows, owner), root[frame]+offset,
                                   rot[frame], dofs[frame], vel[frame], ang[frame], dof_vel[frame])
                env._box_states[rows, source, :3] = box_pos[frame]+offset
                env._box_states[rows, source, 2] = sizes[:, 2]/2 if skill in ('sit', 'climb') else torch.maximum(box_pos[frame, 2].expand(len(rows)), sizes[:, 2]/2)
                env._box_states[rows, source, 3:7] = box_rot[frame]
                checked = source[:, None]
                if end is not None:
                    env._box_states[rows, support, :2] = end[0, :2]+offset[:, :2]
                    checked = torch.stack((source, support), -1)
                initial_root = env._humanoid_root_states[rows, owner].clone()
                initial_boxes = env._box_states[rows[:, None], checked].clone()
                env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env._root_states))
                env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env._dof_state))
                targets = env._dof_pos.contiguous().clone()
                env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(targets))
                ok = torch.ones(len(rows), dtype=torch.bool, device=env.device)
                if strict:
                    ok &= pose_geometry_valid(bodies[frame]+offset[:, None],
                        feet[frame, -2:]+offset[:, None], initial_boxes,
                        env._box_size[rows[:, None], checked])
                for _ in range(steps):
                    env.gym.simulate(env.sim)
                    env.gym.fetch_results(env.sim, True)
                    env._refresh_sim_tensors()
                    ok &= screen_states(env._humanoid_root_states[rows, owner],
                                        env._box_states[rows[:, None], checked], initial_root, initial_boxes)
                    ok &= torch.isfinite(env._dof_state.view(env.num_envs, env.num_agents, -1)[rows, owner]).all(-1)
                    if strict:
                        body = env._rigid_body_pos[rows, owner]
                        ok &= pose_geometry_valid(body, body[:, env._key_body_ids[-2:]],
                            env._box_states[rows[:, None], checked], env._box_size[rows[:, None], checked])
                        # The cache is also used at progress=0, where runtime fall
                        # termination has a grace period. Screen fallen poses now.
                        low = body[..., 2] < env._termination_heights
                        low[:, env._contact_body_ids] = False
                        contact = env._contact_forces[rows, owner].clone()
                        contact[:, env._contact_body_ids] = 0
                        fallen = low.any(-1) & (contact.abs().sum(-1)>.1).any(-1)
                        ok &= ~fallen
                valid[:, frame] = ok
                if frame % 300 == 299:
                    print('[size RSI]', kind, 'owner', owner, 'clip', clip,
                          'frame', frame+1, '/', frames, flush=True)
            if strict and skill in ('sit', 'climb'):
                index = torch.arange(frames, device=env.device)[None]
                last = torch.where(phases, index, -1).amax(-1)
                valid &= (last[:, None] < 0) | (index <= last[:, None])
            elif skill in ('sit', 'climb') and phase.any():
                last = int(phase.nonzero()[-1])
                valid[:, last+1:] = False
            for record, mask, row_phase in zip(records, valid.cpu().numpy(), phases.cpu().numpy()):
                record.append((mask, row_phase))
            print('[size RSI]', kind, 'owner', owner, 'clip', clip,
                  'accepted', int(valid.sum()), '/', valid.numel(), flush=True)
        self.entries.update(zip(keys, records))

