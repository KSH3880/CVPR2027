"""Dynamic native chair, original carry reset distribution, existing HOLDING+AT reward."""
import os
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_from_angle_axis
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.humanoid_chair_ft import NoBoxRsi
from utils.hold_at_ft import object_catalog
from utils.unified_training import append_family

class HumanoidChairHoldAtFT(HumanoidMACarry):
    def __init__(self, *args, **kwargs):
        self._solo_cache_class = NoBoxRsi
        super().__init__(*args, **kwargs)
        if self.num_agents != 1 or self.num_objects != 1:
            raise ValueError('Chair HOLDING_AT requires one person and one dynamic object')

    def _load_box_asset(self):
        cfg = self.cfg['env']['holdAtFT']
        catalog = object_catalog(os.environ.get('HOLD_AT_OBJECT_ROOT', cfg['object_root']), cfg['asset_split'], cfg['category'])
        ids = torch.multinomial(torch.ones(len(catalog), device=self.device), self.num_envs, replacement=True)
        self._object_model_ids = ids
        selected = [catalog[i] for i in ids.cpu().tolist()]
        self._box_size = torch.tensor([x['bbox'] for x in selected], device=self.device)[:,None]
        self._object_front = torch.tensor([x['facing'] for x in selected], device=self.device, dtype=torch.float)
        self._object_ground_z = torch.tensor([-(x['center'][2]-x['bbox'][2]/2) for x in selected], device=self.device)
        self._box_density = torch.full((self.num_envs,), 100., device=self.device)
        self._box_scale = self._box_size / torch.tensor(self._build_base_size, device=self.device)
        # Eligibility only; do not exclude native furniture based on box RSI size limits.
        self._rsi_asset_sizes = torch.full_like(self._box_size, .45)
        self._box_assets = []; cache = {}
        for entry in selected:
            folder = entry['path']
            if folder not in cache:
                opt = gymapi.AssetOptions(); opt.fix_base_link = False
                opt.angular_damping = .01; opt.linear_damping = .01; opt.max_angular_velocity = 100.
                opt.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                opt.use_mesh_materials = True; opt.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
                opt.vhacd_enabled = True; opt.vhacd_params.resolution = 100000
                opt.vhacd_params.max_convex_hulls = 128; opt.vhacd_params.max_num_vertices_per_ch = 64
                opt.replace_cylinder_with_capsule = False
                cache[folder] = self.gym.load_asset(self.sim, folder, 'asset.urdf', opt)
                if cache[folder] is None or self.gym.get_asset_rigid_body_count(cache[folder]) != 1:
                    raise ValueError(f'Expected one rigid body: {folder}')
            self._box_assets.append(cache[folder])

    def _init_relation_runtime(self):
        self._size_template_probabilities.zero_(); self._size_template_probabilities[...,3] = 1
        return super()._init_relation_runtime()

    def _motion_skills_to_load(self, available):
        wanted = {'loco'} if self.cfg['args'].eval else {'loco', 'loco_more', 'omomo', 'pickUp', 'carryWith', 'putDown'}
        return [name for name in available if name in wanted]

    def _reset_ref_state_init(self, slot_env, slot_agent):
        # Bypass cuboid RSI screening/cache; original furniture adaptation starts from locomotion.
        saved = self._is_eval
        skills, probabilities = self._skill, self._skill_init_prob
        try:
            self._is_eval = True
            if not saved:
                self._skill = ['loco_more', 'loco']
                self._skill_init_prob = torch.tensor([.5, .5],device=self.device)
            return super()._reset_ref_state_init(slot_env, slot_agent)
        finally:
            self._is_eval = saved
            self._skill, self._skill_init_prob = skills, probabilities

    def _init_amp_obs(self, env_ids):
        super()._init_amp_obs(env_ids)
        if 'loco_more' in self._reset_ref_slots:
            self._init_amp_obs_ref(self._reset_ref_slots['loco_more'],self._reset_ref_motion_ids['loco_more'],self._reset_ref_motion_times['loco_more'],'loco_more')

    def _reset_boxes(self, env_ids):
        n = len(env_ids); device = self.device
        direction = torch.randn(n, 2, device=device)
        direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        distance = 1 + 4*torch.rand(n, 1, device=device)
        pos = torch.zeros(n, 3, device=device)
        pos[:,:2] = self._humanoid_root_states[env_ids,0,:2] + direction*distance
        pos[:,2] = self._object_ground_z[env_ids] + .05
        front = self._object_front[env_ids]
        target = self._humanoid_root_states[env_ids,0,:2] - pos[:,:2]
        yaw = torch.atan2(target[:,1], target[:,0]) - torch.atan2(front[:,1], front[:,0])
        axis = torch.zeros(n, 3, device=device); axis[:,2] = 1
        self._box_states[env_ids,0,:3] = pos
        self._box_states[env_ids,0,3:7] = quat_from_angle_axis(yaw, axis)
        self._box_states[env_ids,0,7:13] = 0
        self._platform_pos[env_ids] = self._platform_default_pos[env_ids]

    def _reset_task(self, env_ids):
        # Same original target square and squared-distance rejection threshold (<2.0).
        n = len(env_ids); root = self._humanoid_root_states[env_ids,0,:2]
        obj = self._box_states[env_ids,0,:2]; origin = self._env_origins[env_ids,:2]
        xy = origin + (torch.rand(n,2,device=self.device)*2-1)*4.5
        bad = ((xy-root).square().sum(-1)<2) | ((xy-obj).square().sum(-1)<2)
        for _ in range(1000):
            if not bad.any(): break
            xy[bad] = origin[bad] + (torch.rand(int(bad.sum()),2,device=self.device)*2-1)*4.5
            bad = ((xy-root).square().sum(-1)<2) | ((xy-obj).square().sum(-1)<2)
        if bad.any(): raise RuntimeError('Could not sample original furniture target')
        self._tar_pos[env_ids,0,:2] = xy
        self._tar_pos[env_ids,0,2] = self._object_ground_z[env_ids]
        self._tar_platform_pos[env_ids] = self._tar_platform_default_pos[env_ids]

    def _ontop_scene_infeasible(self, env_ids):
        # Match original furniture adaptation without additional cuboid body-point rejection.
        return torch.zeros(len(env_ids),dtype=torch.bool,device=self.device)

    def _fetch_conditioned_amp_demo(self, num_samples):
        names = ['loco_more','loco','pickUp','carryWith','putDown']
        probs = torch.tensor([.5,.2,.1,.1,.1],device=self.device)
        skills = torch.multinomial(probs,num_samples,replacement=True)
        result = torch.empty(num_samples,self.get_num_amp_obs(),device=self.device)
        for i,name in enumerate(names):
            rows = (skills==i).nonzero().flatten()
            if not len(rows): continue
            lib = self._motion_lib[name]; mid = lib.sample_motions(len(rows))
            truncate = self.dt*(self._num_amp_obs_steps-1)
            times = lib.sample_time(mid,truncate_time=truncate)+truncate
            obs = self.build_amp_obs_demo(mid,times,lib).reshape(len(rows),self._num_amp_obs_steps,-1)
            result[rows] = append_family(obs,torch.zeros(len(rows),self._num_amp_obs_steps,dtype=torch.long,device=self.device),3).flatten(1)
        return result
