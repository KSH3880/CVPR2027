from isaacgym.torch_utils import quat_from_angle_axis, quat_mul, quat_rotate
import torch

from env.tasks.multi_agent import square_sensor as sensor
from env.tasks.multi_agent import square_sensor_scene as scene
from utils import torch_utils


class CompactRsi:
    def _reset_ref_state_init(self, env, owner):
        super()._reset_ref_state_init(env, owner)
        selected = self._compact_kind[env,owner] > 0
        pending, people = env[selected], owner[selected]
        lib = self._motion_lib['loco']
        for _ in range(self._scene_cfg['bridge']['boundary_rsi']['attempts']):
            if not len(pending):
                break
            motions = lib.sample_motions(len(pending))
            times = lib.sample_time_rsi(motions)
            root,q,dof,vel,ang,dof_vel,_ = lib.get_motion_state(motions,times)
            heading = torch_utils.calc_heading(q)
            forward = torch.stack((heading.cos(),heading.sin()),-1)
            speed = vel[:,:2].norm(dim=-1)
            good = (speed >= .3) & (speed <= 1.5)
            good &= (vel[:,:2]*forward).sum(-1) >= speed*0.8660254
            es, humans = pending[good], people[good]
            if len(es):
                offset = self._agent_spawn_offsets[humans]+self._env_origins[es]
                self._set_env_state(es,humans,root[good]+offset,q[good],dof[good],vel[good],ang[good],dof_vel[good])
                body,bq,bv,ba = lib.get_motion_state_max(motions[good],times[good])
                self._kinematic_humanoid_rigid_body_states[es,humans] = torch.cat((body+offset[:,None],bq,bv,ba),-1)
                self._every_env_init_dof_pos[es,humans] = dof[good]
                slots = self._reset_ref_slots['loco']
                match = (slots[0][:,None] == es) & (slots[1][:,None] == humans)
                records, picked = match.nonzero(as_tuple=True)
                self._reset_ref_motion_ids['loco'][records] = motions[good][picked]
                self._reset_ref_motion_times['loco'][records] = times[good][picked]
            pending,people = pending[~good],people[~good]
        self._compact_failures[0] += len(pending)
        self._compact_kind[pending,people] = 0

    def _apply_boundary_rsi(self, ids):
        rows,people = ((self._compact_kind[ids] > 0) & ~self._place_failed[ids,None]).nonzero(as_tuple=True)
        env = ids[rows]
        if not len(env):
            return
        roots = self._humanoid_root_states[env,people].clone()
        bodies = self._kinematic_humanoid_rigid_body_states[env,people].clone()
        kind = self._compact_kind[env,people]
        source = self._start_surface[env,people]-1
        center = self._surface_center[env,1,source,0]
        landing = self._bridge_landing[env]
        sign = torch.sign(center-landing[:,0])
        size = self._box_size[env,self._bridge_slot[env]]
        yaw = self._structure_yaw[env,1]+torch.where(sign < 0,0.,torch.pi)
        old_yaw = torch_utils.calc_heading(roots[:,3:7])
        axis = torch.zeros(len(env),3,device=self.device)
        axis[:,2] = 1.
        q = quat_from_angle_axis(yaw-old_yaw,axis)
        qb = q[:,None].expand(-1,self.num_bodies,-1).reshape(-1,4)
        relative = quat_rotate(qb,(bodies[...,:3]-roots[:,None,:3]).reshape(-1,3)).view(-1,self.num_bodies,3)
        rotated = bodies.clone()
        rotated[...,3:7] = quat_mul(qb,bodies[...,3:7].reshape(-1,4)).view(-1,self.num_bodies,4)
        for start in (7,10):
            rotated[...,start:start+3] = quat_rotate(qb,bodies[...,start:start+3].reshape(-1,3)).view(-1,self.num_bodies,3)
        pending = torch.ones(len(env),device=self.device,dtype=torch.bool)
        for _ in range(self._scene_cfg['bridge']['boundary_rsi']['attempts']):
            distance = .35+.3*torch.rand(len(env),device=self.device)
            local = landing.clone()
            normal = sign*(self._bridge_gap[env]/2+distance)
            on = sign*(size[:,0]/2-distance)
            local[:,0] += torch.where(kind == 1,normal,torch.where(kind == 2,on,-on))
            xy = scene.to_world(local,self._structure_yaw[env,1],self._structure_shift[env,1],self._env_origins[env])
            moved = rotated.clone()
            position = roots[:,:3].clone()
            position[:,:2] = xy
            position[:,2] -= (kind > 1)*self._scene_cfg['bridge']['top_drop']
            moved[...,:3] = relative+position[:,None]
            good = self._boundary_supported(env,moved)
            others = self._humanoid_root_states[env,:,:2]
            separation = (xy[:,None]-others).norm(dim=-1)
            separation[torch.arange(len(env),device=self.device),people] = torch.inf
            good &= (separation >= self._bridge_curriculum['person_clearance']).all(-1)
            active = self._active_physical(env)
            active[torch.arange(len(env),device=self.device),self._bridge_slot[env]] = False
            boxes = self._box_states[env,:,:2]
            radius = self._box_size[env,:,:2].norm(dim=-1)/2
            good &= ~(((xy[:,None]-boxes).norm(dim=-1) < radius+.5) & active).any(-1)
            good &= pending & ~self._place_failed[env]
            es,humans = env[good],people[good]
            self._humanoid_root_states[es,humans,:3] = position[good]
            self._humanoid_root_states[es,humans,3:7] = quat_mul(q[good],roots[good,3:7])
            for start in (7,10):
                self._humanoid_root_states[es,humans,start:start+3] = quat_rotate(q[good],roots[good,start:start+3])
            self._kinematic_humanoid_rigid_body_states[es,humans] = moved[good]
            self._move_neutral_goal(es,humans,xy[good]-roots[good,:2],position[good,2]-roots[good,2])
            self._cross_phase[es,humans] = torch.where(kind[good] == 1,0,torch.where(kind[good] == 2,1,2))
            self._cross_progress[es,humans] = torch.where(kind[good] == 2,distance[good]/size[good,0],
                torch.where(kind[good] == 3,1-distance[good]/size[good,0],0.))
            pending &= ~good
            if not pending.any():
                break
        self._compact_failures[1] += pending.sum()
        self._compact_kind[env[pending],people[pending]] = 0

    def _boundary_supported(self, env, body):
        vertices = []
        heights = []
        for index,points in zip(self._foot_body_ids,self._foot_vertices):
            foot = body[:,index]
            rotation = sensor.rotation_matrix(foot[:,3:7])
            world = points[None]@rotation.transpose(-1,-2)+foot[:,None,:3]
            vertices.append(world)
            heights.append(world[...,2].amin(-1))
        bottom = torch.stack(heights,-1)
        valid = torch.ones(len(env),device=self.device,dtype=torch.bool)
        support = torch.zeros_like(valid)
        for index,world in enumerate(vertices):
            ground = self._terrain_height(world[...,:2],env)
            delta = world[...,2]-ground
            valid &= (delta >= -.005).all(-1)
            planted = bottom[:,index] <= bottom.amin(-1)+.03
            support |= planted & (delta.amin(-1) <= .05) & (ground > self._env_origins[env,2,None]+.5).all(-1)
        return valid & support
