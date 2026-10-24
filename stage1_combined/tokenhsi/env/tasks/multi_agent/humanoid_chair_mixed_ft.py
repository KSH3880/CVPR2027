"""Replay the original box tasks alongside separate chair SIT/CLIMB scenes."""
import json
from pathlib import Path
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate,quat_from_angle_axis
from utils.chair_front import front_allowed
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.humanoid_chair_ft import HumanoidChairFT
from env.tasks.multi_agent.size_rsi_cache import SizeRsiCache
from utils.chair_geometry import EXTRA
from utils.edge_scenario_spec import classify_templates


class MixedBoxRsiCache(SizeRsiCache):
    def __init__(self, env):
        # Profiles and geometry screening concern only the unchanged box scenes.
        actual = env._box_size
        try:
            env._box_size = env._original_box_sizes
            super().__init__(env)
        finally:
            env._box_size = actual

    def _profiles(self, kind, owner, pending):
        self.eligible[self.env._chair_env_mask] = 0
        return super()._profiles(kind, owner, pending)


class HumanoidChairMixedFT(HumanoidChairFT):
    def __init__(self, *args, **kwargs):
        self._solo_cache_class = MixedBoxRsiCache
        HumanoidMACarry.__init__(self, *args, **kwargs)
        if self.num_agents != 2 or self.num_objects != 4:
            raise ValueError('Chair mixed FT requires 2 humans and 4 objects')

    def _load_box_asset(self):
        HumanoidMACarry._load_box_asset(self)
        self._original_box_sizes = self._box_size.clone()
        fraction = float(self.cfg['env']['chairFT']['chair_environment_fraction'])
        if not 0 < fraction < 1 and not (fraction == 1 and self.cfg['env']['chairFT'].get('sit_only',False)):
            raise ValueError('chair_environment_fraction must be between zero and one')
        count = max(1, min(self.num_envs - 1, round(self.num_envs * fraction))) if self.num_envs > 1 else 1
        if fraction == 1: count = self.num_envs
        self._chair_env_mask = torch.arange(self.num_envs, device=self.device) < count
        self._native_asset_keys = [['']*self.num_objects for _ in range(self.num_envs)]
        self._chair_features = torch.zeros(self.num_envs, self.num_objects, EXTRA, device=self.device)
        self._chair_seats = torch.zeros(self.num_envs, self.num_objects, 3, device=self.device)
        self._chair_seat_extents = torch.zeros(self.num_envs, self.num_objects, 2, device=self.device)
        self._chair_kinds = torch.zeros(self.num_envs, self.num_objects, dtype=torch.long, device=self.device)
        self._chair_kinds[self._chair_env_mask, :self.num_agents] = 2
        self._climb_metadata_heights = torch.zeros(self.num_envs, self.num_objects, device=self.device)
        self._chair_table_ft = self.cfg['env']['chairFT'].get('target_categories') == ['chair', 'table']
        if self._chair_table_ft:
            first = torch.randint(1, 3, (count,), device=self.device)
            self._chair_kinds[:count, 0] = first
            self._chair_kinds[:count, 1] = 3-first
        root = Path(__file__).resolve().parents[4] / 'chair_assets'
        catalog = json.loads((root / 'catalog.json').read_text())
        entries = catalog[self.cfg['env']['chairFT'].get('asset_split', 'train')]
        indices = torch.randint(len(entries), (count, self.num_agents)).tolist()
        cache = {}
        for row in range(count):
            for owner in range(self.num_agents):
                kind = int(self._chair_kinds[row, owner])
                entry = entries[indices[row][owner]] if kind == 2 else catalog['tables'][indices[row][owner] % len(catalog['tables'])]
                self._native_asset_keys[row][owner] = Path(entry['path']).name
                metadata = json.loads((Path(entry['path']) / 'config.json').read_text())
                self._climb_metadata_heights[row, owner] = metadata['tarSitPos'][2]
                key = (kind, entry['path'])
                if key not in cache:
                    options = gymapi.AssetOptions()
                    options.fix_base_link = True
                    options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                    options.use_mesh_materials = True
                    options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
                    options.vhacd_enabled = True
                    options.vhacd_params.resolution = 100000
                    options.vhacd_params.max_convex_hulls = 128
                    options.vhacd_params.max_num_vertices_per_ch = 64
                    asset = self.gym.load_asset(self.sim, entry['path'], 'asset.urdf', options)
                    if asset is None or self.gym.get_asset_rigid_body_count(asset) != 1 or self.gym.get_asset_dof_count(asset) != 0:
                        raise ValueError('Chair must be a single rigid body without DOFs')
                    cache[key] = asset
                self._box_assets[row*self.num_objects + owner] = cache[key]
                self._box_size[row, owner] = torch.tensor(entry['bbox'], device=self.device)
                seat = entry['seat'] if kind == 2 else [0., 0., entry['bbox'][2]/2]
                feature = entry['feature'] if kind == 2 else [1., 0., 0., *seat]
                extent = entry['seat_extent'] if kind == 2 else entry['bbox'][:2]
                self._chair_features[row, owner] = torch.tensor(feature, device=self.device)
                self._chair_seats[row, owner] = torch.tensor(seat, device=self.device)
                self._chair_seat_extents[row, owner] = torch.tensor(extent, device=self.device)
        self._box_scale = self._box_size / torch.tensor(self._build_base_size, device=self.device)
        self._rsi_asset_sizes = self._original_box_sizes.clone()
        self._rsi_asset_sizes[self._chair_env_mask, :self.num_agents] = .45
        print(f'[chair mixed FT] chair scenes={count}; original box scenes={self.num_envs-count}', flush=True)

    def _init_relation_runtime(self):
        if self.cfg['env']['chairFT'].get('training_tasks') == ['sit', 'climb']:
            self._size_template_probabilities.zero_()
            self._size_template_probabilities[..., 1:3] = .5
        self._size_template_probabilities[self._chair_env_mask] = 0
        self._size_template_probabilities[self._chair_env_mask, :, 1:3] = .5
        return super()._init_relation_runtime()

    def _reset_ref_state_init(self, slot_env, slot_agent):
        chair = self._chair_env_mask[slot_env]
        if (~chair).any():
            HumanoidMACarry._reset_ref_state_init(self, slot_env[~chair], slot_agent[~chair])
        if chair.any():
            # Save box reference slots: the loco entry must retain BOTH scene groups.
            slots = dict(self._reset_ref_slots)
            ids = dict(self._reset_ref_motion_ids)
            times = dict(self._reset_ref_motion_times)
            HumanoidChairFT._reset_ref_state_init(self, slot_env[chair], slot_agent[chair])
            # Chair training always samples loco from the base evaluation distribution.
            for skill in self._skill:
                old = slots.get(skill)
                new = self._reset_ref_slots.get(skill)
                if old is not None and new is not old:
                    self._reset_ref_slots[skill] = tuple(torch.cat([a, b]) for a, b in zip(old, new))
                    self._reset_ref_motion_ids[skill] = torch.cat([ids[skill], self._reset_ref_motion_ids[skill]])
                    self._reset_ref_motion_times[skill] = torch.cat([times[skill], self._reset_ref_motion_times[skill]])

    def _evaluate_ontop_context(self, env_ids=None):
        box_phi, box_diag = HumanoidMACarry._evaluate_ontop_context(self, env_ids)
        chair_phi, chair_diag = HumanoidChairFT._evaluate_ontop_context(self, env_ids)
        mask = self._chair_env_mask if env_ids is None else self._chair_env_mask[env_ids]
        def merge(old, new):
            if torch.is_tensor(old) and old.ndim and old.shape[0] == len(mask):
                return torch.where(mask.reshape(-1, *([1]*(old.ndim-1))), new, old)
            return old
        phi = merge(box_phi, chair_phi)
        diag = {key: merge(value, chair_diag[key]) for key, value in box_diag.items()}
        if env_ids is None:
            self._ontop_last_diag = diag
        return phi, diag

    def _reset_envs(self, env_ids):
        result = HumanoidMACarry._reset_envs(self, env_ids)
        if not hasattr(self, '_mixed_scene_saved') and len(env_ids) == self.num_envs:
            row = env_ids[:, None].expand(-1, self.num_agents).reshape(-1)
            owner = torch.arange(self.num_agents, device=self.device).repeat(len(env_ids))
            kinds = classify_templates(self.relation_runtime.graph, row, owner, True)
            chair = self._chair_env_mask[row]
            if self.cfg['env']['chairFT'].get('training_tasks') == ['sit', 'climb']:
                assert ((kinds == 1) | (kinds == 2)).all(), 'Unexpected task in SIT/CLIMB-only scene'
            assert ((kinds[chair] == 1) | (kinds[chair] == 2)).all()
            assert torch.equal(self._box_size[~self._chair_env_mask], self._original_box_sizes[~self._chair_env_mask])
            report = {'num_envs':self.num_envs, 'chair_envs':int(self._chair_env_mask.sum()),
                      'box_envs':int((~self._chair_env_mask).sum()),
                      'box_template_counts':torch.bincount(kinds[~chair], minlength=5).cpu().tolist(),
                      'chair_template_counts':torch.bincount(kinds[chair], minlength=5).cpu().tolist(),
                      'native_target_counts_box_table_chair':torch.bincount(self._chair_kinds[self._chair_env_mask, :self.num_agents].flatten(), minlength=3).cpu().tolist(),
                      'box_sizes_unchanged':True, 'box_rsi':'original_size_screened', 'chair_rsi':'loco_only'}
            folder = Path(self.cfg['args'].output_path)
            folder.mkdir(parents=True, exist_ok=True)
            (folder/'chair_mixed_scene_validation.json').write_text(json.dumps(report, indent=2))
            self._mixed_scene_saved = True
            if hasattr(self,'_native_pool'):
                self._save_native_rsi_snapshot(folder)
        return result

    def _reset_boxes(self, env_ids):
        super()._reset_boxes(env_ids)
        if self.cfg['env']['chairFT'].get('chair_front_only',False):
            rows=env_ids.repeat_interleave(self.num_agents);owners=torch.arange(self.num_agents,device=self.device).repeat(len(env_ids));physical=self._scenario_physical_object(rows,owners);chair=self._chair_kinds[rows,physical]==2
            r,o,p=rows[chair],owners[chair],physical[chair]
            if len(r):
                direction=self._humanoid_root_states[r,o,:2]-self._box_states[r,p,:2]
                facing=-self._chair_features[r,p,1:3]
                yaw=torch.atan2(direction[:,1],direction[:,0])-torch.atan2(facing[:,1],facing[:,0])
                axis=torch.zeros(len(r),3,device=self.device);axis[:,2]=1
                self._box_states[r,p,3:7]=quat_from_angle_axis(yaw,axis)
        options=self.cfg['env']['chairFT'].get('native_rsi')
        if options and (not self._is_eval or options.get('viewer',False)):
            if not hasattr(self,'_native_pool'):
                from utils.native_rsi import NativeRsiPool
                self._native_pool=NativeRsiPool(self,options)
            self._native_pool.apply(env_ids)

    def _init_amp_obs(self, env_ids):
        super()._init_amp_obs(env_ids)
        if hasattr(self,'_native_pool'):self._native_pool.init_history(env_ids)

    def _ontop_scene_infeasible(self, ids):
        bad=super()._ontop_scene_infeasible(ids)
        if hasattr(self,'_native_pool'):
            active=self._native_pool.active[ids].any(-1)
            bodies=self._kinematic_humanoid_rigid_body_states[ids,...,:3]
            distance=torch.cdist(bodies[:,0],bodies[:,1]).amin(dim=(-1,-2))
            bad|=active&(distance<.15)
        if self.cfg['env']['chairFT'].get('native_rsi',{}).get('require_support',False):
            from utils.rsi_support import load_support_points,box_supported
            if not hasattr(self,'_support_shapes'):
                self._support_shapes=load_support_points('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml',self.device)
                self._box_unsupported_rejections=0
            for skill in ['sit','climb']:
                slots=self._reset_ref_slots.get(skill)
                if slots is None:continue
                r,o=slots;p=self._scenario_physical_object(r,o);keep=self._chair_kinds[r,p]==0
                r,o,p=r[keep],o[keep],p[keep]
                if not len(r):continue
                ok=box_supported(self._kinematic_humanoid_rigid_body_states[r,o],self._box_states[r,p],self._box_size[r,p],self._support_shapes)
                local=torch.searchsorted(ids,r)
                bad[local[~ok]]=True;self._box_unsupported_rejections+=int((~ok).sum())
        if self.cfg['env']['chairFT'].get('chair_front_only',False):
            rows=ids.repeat_interleave(self.num_agents);owners=torch.arange(self.num_agents,device=self.device).repeat(len(ids));physical=self._scenario_physical_object(rows,owners);chair=self._chair_kinds[rows,physical]==2
            r,o,p=rows[chair],owners[chair],physical[chair]
            inv=self._box_states[r,p,3:7].clone();inv[:,:3]*=-1
            local=quat_rotate(inv,self._humanoid_root_states[r,o,:3]-self._box_states[r,p,:3])
            ok=front_allowed(local,self._chair_seats[r,p],-self._chair_features[r,p,1:3])
            bad[torch.searchsorted(ids,r[~ok])]=True
        options=self.cfg['env']['chairFT'].get('native_rsi',{})
        if options.get('strict_foot_collision',False):
            from utils.native_foot_guard import NativeFootGuard
            if not hasattr(self,'_native_foot_guard'):self._native_foot_guard=NativeFootGuard(self,options['bank'])
            bad|=torch.tensor(self._native_foot_guard.rejected(ids),device=self.device)
        return bad

    def _save_native_rsi_snapshot(self, folder):
        import numpy as np
        pool=self._native_pool
        rows=torch.arange(self.num_envs,device=self.device).repeat_interleave(self.num_agents)
        owners=torch.arange(self.num_agents,device=self.device).repeat(self.num_envs)
        physical=self._scenario_physical_object(rows,owners)
        tasks=classify_templates(self.relation_runtime.graph,rows,owners,True)
        actual=self._chair_kinds[rows,physical]
        _,diag=self._evaluate_ontop_context()
        chosen=[]
        for category in [0,1,2]:
            for task in [1,2]:
                valid=(actual==category)&(tasks==task)
                if category:
                    valid&=pool.active[rows,owners]
                    preferred=valid&(pool.lowering[rows,owners] if task==1 else ~pool.phase[rows,owners])
                    if preferred.any():valid=preferred
                else:
                    advanced=torch.zeros(self.num_envs,self.num_agents,device=self.device,dtype=torch.bool)
                    for skill in ['sit','climb']:
                        slots=self._reset_ref_slots.get(skill)
                        if slots is not None:advanced[slots]=True
                    valid&=advanced[rows,owners]
                if category:
                    base=(actual==category)&(tasks==task)&pool.active[rows,owners]
                    early=base&~pool.phase[rows,owners]
                    if task==1 and (early&pool.lowering[rows,owners]).any():early&=pool.lowering[rows,owners]
                    for stage,mask in [('approach',early),('late',base&pool.phase[rows,owners])]:
                        found=torch.nonzero(mask).flatten()
                        if len(found):chosen.append((int(rows[found[0]]),category,task,int(owners[found[0]]),stage))
                else:
                    found=torch.nonzero(valid).flatten()
                    if len(found):chosen.append((int(rows[found[0]]),category,task,int(owners[found[0]]),'box_reference'))
        selected=torch.tensor([x[0] for x in chosen],device=self.device)
        def cpu(x):return x.detach().cpu().numpy()
        np.savez_compressed(folder/'rsi_reset_snapshot.npz',humans=cpu(self._kinematic_humanoid_rigid_body_states[selected]),root=cpu(self._humanoid_root_states[selected]),dof=cpu(self._dof_pos[selected]),dof_vel=cpu(self._dof_vel[selected]),objects=cpu(self._box_states[selected]),sizes=cpu(self._box_size[selected]),kinds=cpu(self._chair_kinds[selected]),goal=cpu(diag['target'][selected]),goal_valid=cpu(self.relation_runtime.graph.edge_valid[selected]),amp=cpu(self._amp_obs_buf[selected]))
        metadata={'scenes':[dict(env=x[0],category=x[1],task=x[2],owner=x[3],stage=x[4],asset_keys=self._native_asset_keys[x[0]]) for x in chosen], 'native_counts':pool.counts,'active_native_slots':int(pool.active.sum()),'active_nonphase':int((pool.active&~pool.phase).sum()),'active_sit_lowering':int(pool.lowering.sum()),'native_amp_history_max_error':float((self._hist_amp_obs_buf[pool.active]-pool.history[pool.active]).abs().max()),'amp_finite':bool(torch.isfinite(self._amp_obs_buf).all()),'native_foot_scene_rejections':getattr(getattr(self,'_native_foot_guard',None),'rejections',0),'chair_front_only':self.cfg['env']['chairFT'].get('chair_front_only',False),'box_unsupported_rejections':getattr(self,'_box_unsupported_rejections',0),'source':self.cfg['env']['chairFT']['source_checkpoint']}
        (folder/'rsi_reset_validation.json').write_text(json.dumps(metadata,indent=2))
