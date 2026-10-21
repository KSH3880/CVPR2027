"""Private native-object fine-tuning; installed only by this port's launcher."""
import hashlib
import json
from pathlib import Path
import types

PORT = Path(__file__).resolve().parent


def install(output, split):
    import torch
    from isaacgym import gymapi, gymtorch
    from isaacgym.torch_utils import quat_rotate
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    from env.tasks.multi_agent import size_rsi_cache
    from utils import task_role_spec
    from utils.edge_scenario_spec import sample_size_conditioned_graph
    from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges
    from utils.edge_ontop_spec import select_graph

    old_load = HumanoidMACarry._load_box_asset
    old_sample = HumanoidMACarry._sample_episode_graph
    old_evaluate = HumanoidMACarry._evaluate_ontop_context
    old_reset_boxes = HumanoidMACarry._reset_boxes
    old_build_box = HumanoidMACarry._build_box
    active = {}

    def probabilities(sizes, carry_only=False):
        task = active['task']
        if sizes.shape[:2] != task._ft_probabilities.shape[:2]:
            raise ValueError('Unexpected native task probability shape')
        return task._ft_probabilities.clone()

    task_role_spec.task_size_probabilities = probabilities
    def cache_probabilities(sizes):
        p = probabilities(sizes)
        p[..., 0] = p[..., 3]  # carry skill cache uses HOLDING eligibility
        initialization = active['task'].cfg['env']['objectFineTune']['initialization']
        if initialization == 'loco_only':
            p.zero_()
        elif initialization == 'joint_sit_carry':
            p[..., 0] = 0.
            p[..., 3] = 0.
        return p
    size_rsi_cache.task_probabilities = cache_probabilities

    def retarget(self, rows, owners, skill, clips, times, update_kinematic=True):
        delta = torch.zeros(len(rows), 3, device=self.device)
        if skill == 'sit':
            lib = self._motion_lib[skill]
            reference_seat = []
            for clip in clips.tolist():
                start = int(lib.length_starts[clip]); count = int(lib._motion_num_frames[clip])
                # The source object_state has a floor-origin Z, not a seat Z.
                # Anchor each clip to its low, seated root phase instead.
                reference_seat.append(lib.gts[start:start+count, 0, 2].quantile(.1) - .12)
            reference_seat = torch.stack(reference_seat)
            box = self._box_states[rows, owners]
            target = self._ft_seats[rows, owners].clone()
            target[:, 2] -= .10
            seat_world = quat_rotate(box[:, 3:7], target) + box[:, :3]
            delta[:, 2] = seat_world[:, 2] - reference_seat
        elif skill in ('pickUp', 'carryWith', 'putDown'):
            ref, _ = self._motion_lib[skill].get_obj_motion_state(clips, times)
            delta[:, 2] = (self._box_states[rows, owners, 2] - ref[:, 2]).clamp_min(0)
        self._humanoid_root_states[rows, owners, :3] += delta
        if update_kinematic:
            self._kinematic_humanoid_rigid_body_states[rows, owners, :, :3] += delta[:, None]

    def flush(self):
        # Clear mesh teleport contact manifolds without changing candidate states.
        roots, dofs = self._root_states.clone(), self._dof_state.clone()
        self._box_states[:, :self.num_agents, 2] += 50.
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
        for _ in range(2):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        self._refresh_sim_tensors()
        self._root_states.copy_(roots)
        self._dof_state.copy_(dofs)

    def load(self):
        old_load(self)
        cfg = self.cfg['env']['objectFineTune']
        from utils import unified_training
        # TokenHSI has loco_more/loco aliases; this checkpoint has one loco slot.
        unified_training.EXPERT_PROBS = ((.7,0.,0.,0.,0.,.1,.1,.1),
                                         (.2,.8,0.,0.,0.,0.,0.,0.),
                                         (1/3,0.,1/3,1/3,0.,0.,0.,0.))
        self._ft_amp_family_probs = ((.5,.5,0.) if cfg['task']=='joint' else
            (0.,1.,0.) if cfg['task']=='sit' else (1.,0.,0.))
        N, M, O = self.num_envs, self.num_agents, self.num_objects
        if (M, O) != (2, 4):
            raise ValueError('Native SIT/CARRY fine-tune requires 2 agents/4 object slots')
        self._ft_seats = torch.zeros(N, O, 3, device=self.device)
        self._ft_sit = torch.zeros(N, M, dtype=torch.bool, device=self.device)
        self._ft_probabilities = torch.zeros(N, M, 5, device=self.device)
        self._ft_profile_keys = [[None] * M for _ in range(N)]
        self._ft_profile_specs = {}
        pools = []
        dataset = 'dataset_sit' if cfg['task'] == 'sit' else 'dataset_dynamic_objects'
        recipes = [('dataset_sit', 'StraightChair_Normal', 'sit'),
                   ('dataset_dynamic_objects', 'ArmChair_Normal', 'carry'),
                   ('dataset_dynamic_objects', 'Table_Circle', 'carry')] if cfg['task']=='joint' else [(dataset,cfg['category'],cfg['task'])]
        for dataset, category, task in recipes:
            dirs = sorted((PORT / 'tokenhsi/data' / dataset / 'objects' / split / category).glob('*/config.json'))
            if dataset == 'dataset_dynamic_objects':
                dirs = sorted((PORT / 'tokenhsi/data' / dataset / split / category).glob('*/config.json'))
            limit = cfg['assets_per_category']
            if limit:
                dirs = dirs[:limit]
            if not dirs:
                raise FileNotFoundError((dataset, split, category))
            for p in dirs:
                spec = json.loads(p.read_text())
                from xml.etree import ElementTree as ET
                xml = ET.parse(p.parent / 'asset.urdf').getroot()
                if len(xml.findall('link')) != 1 or xml.findall('joint'):
                    raise ValueError('One-body native assets required')
                fingerprint = hashlib.sha256()
                for f in sorted(p.parent.rglob('*')):
                    if f.is_file(): fingerprint.update(f.read_bytes())
                key = category + '/' + p.parent.name + '/' + fingerprint.hexdigest()[:16]
                import shutil
                dest = output / 'assets' / p.parent.name
                if not dest.exists(): shutil.copytree(p.parent, dest)
                opts = gymapi.AssetOptions()
                opts.angular_damping = opts.linear_damping = .01
                opts.max_angular_velocity = 100.
                if task == 'sit':
                    # TokenHSI SIT uses fixed objects. Use a resettable dynamic
                    # fixture (as in this runtime's shelves), not a static mesh
                    # whose PhysX collider cannot follow tensor teleport resets.
                    opts.disable_gravity = True
                    opts.max_linear_velocity = 0.
                    opts.max_angular_velocity = 0.
                opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                opts.vhacd_enabled = True
                opts.vhacd_params.resolution = 100000
                opts.vhacd_params.max_convex_hulls = 128
                opts.vhacd_params.max_num_vertices_per_ch = 64
                opts.override_inertia = True
                opts.use_mesh_materials = True
                handle = self.gym.load_asset(self.sim, str(dest), 'asset.urdf', opts)
                if self.gym.get_asset_rigid_body_count(handle) != 1:
                    raise ValueError('Native actor layout changed')
                pools.append((task, key, spec, handle))
                self._ft_profile_specs[key] = dict(bbox=spec['bbox'], seat=spec.get('tarSitPos'),
                    source=str(p.parent), fingerprint=fingerprint.hexdigest())
        for env in range(N):
            for owner in range(M):
                category = ('StraightChair_Normal' if env % 4 < 2 else
                            'ArmChair_Normal' if env % 4 == 2 else 'Table_Circle') if cfg['task']=='joint' else cfg['category']
                sit = category == 'StraightChair_Normal'
                preset = self._task_graph_preset
                if preset == 'sit': sit = True
                elif preset in ('carry_at', 'holding_at'): sit = False
                elif preset != 'random_scenario': raise ValueError('Supported tasks: random_scenario/sit/carry_at')
                self._ft_sit[env, owner] = sit
                self._ft_probabilities[env, owner, 1 if sit else 3] = 1
                replay = cfg['box_replay_fraction'] > 0 and env % 4 == 0
                if replay:
                    size = [.5, .5, .45] if sit else [.35, .35, .35]
                    opts = gymapi.AssetOptions(); opts.density = 100.
                    opts.angular_damping = opts.linear_damping = .01
                    handle = self.gym.create_box(self.sim, *size, opts)
                    key = 'replay_sit' if sit else 'replay_carry'
                    spec = dict(bbox=size, tarSitPos=[0., 0., size[2]/2 + .10])
                    self._ft_profile_specs[key] = spec
                else:
                    choices = [p for p in pools if p[0] == ('sit' if sit else 'carry') and p[1].startswith(category+'/')]
                    _, key, spec, handle = choices[((env // 4)*M + owner if cfg['task']=='joint' else env // 2 + owner) % len(choices)]
                self._ft_profile_keys[env][owner] = key
                self._box_assets[env * O + owner] = handle
                self._box_size[env, owner] = torch.tensor(spec['bbox'], device=self.device)
                if sit: self._ft_seats[env, owner] = torch.tensor(spec['tarSitPos'], device=self.device)
        self._box_scale = self._box_size / torch.tensor(self._build_base_size, device=self.device)
        self._ft_retarget_rsi = types.MethodType(retarget, self)
        self._ft_flush_contacts = types.MethodType(flush, self)
        active['task'] = self
        (output / 'native_profiles.json').write_text(json.dumps(dict(split=split, profiles=self._ft_profile_specs), indent=2))

    def build_box(self, env_id, object_id, env_ptr):
        old_build_box(self, env_id, object_id, env_ptr)
        if object_id < self.num_agents and bool(self._ft_sit[env_id, object_id]):
            handle = self._box_handles[env_id*self.num_objects+object_id]
            bodies = self.gym.get_actor_rigid_body_properties(env_ptr,handle)
            bodies[0].mass = 1.e10
            self.gym.set_actor_rigid_body_properties(env_ptr,handle,bodies,recomputeInertia=True)

    def sample(self, ids):
        probabilities_saved = self._size_template_probabilities
        self._size_template_probabilities = self._ft_probabilities
        try: return old_sample(self, ids)
        finally: self._size_template_probabilities = probabilities_saved

    def reset_boxes(self, ids):
        old_reset_boxes(self, ids)
        for skill in ('sit', 'pickUp', 'carryWith', 'putDown'):
            if self._reset_ref_slots.get(skill) is not None:
                rows, owners = self._reset_ref_slots[skill]
                self._ft_retarget_rsi(rows, owners, skill, self._reset_ref_motion_ids[skill],
                                      self._reset_ref_motion_times[skill])

    def evaluate(self, env_ids=None):
        # Keep policy bbox native; only SIT reward/goal uses annotated seat height.
        rows = slice(None) if env_ids is None else env_ids
        objects = self._logical_box_values(self._box_states, env_ids).clone()
        sizes = self._logical_box_values(self._box_size, env_ids).clone()
        seats = self._logical_box_values(self._ft_seats, env_ids)
        mask = self._ft_sit[rows]
        local = torch.zeros_like(seats[:, :self.num_agents])
        local[..., :2] = seats[:, :self.num_agents, :2]
        local[..., 2] = (seats[:, :self.num_agents, 2] - .10 - sizes[:, :self.num_agents, 2]/2)/2
        delta = quat_rotate(objects[:, :self.num_agents, 3:7].reshape(-1,4), local.reshape(-1,3)).reshape_as(local)
        objects[:, :self.num_agents, :3] += delta * mask[..., None]
        new_height = seats[:, :self.num_agents, 2] - .10 + sizes[:, :self.num_agents, 2]/2
        sizes[:, :self.num_agents, 2] = torch.where(mask, new_height, sizes[:, :self.num_agents, 2])
        bodies = self._rigid_body_pos if env_ids is None else self._kinematic_humanoid_rigid_body_states[env_ids, ..., :3]
        graph = self.relation_runtime.graph if env_ids is None else select_graph(self.relation_runtime.graph, env_ids)
        result = evaluate_interaction_edges(bodies[..., self._key_body_ids[:2], :],
            bodies[..., self._key_body_ids[2:], :], bodies[..., 0, :], objects, sizes,
            self._tar_pos[rows], graph, self._relation_cfg, self._char_h)
        if env_ids is None: self._ontop_last_diag = result[1]
        return result

    HumanoidMACarry._load_box_asset = load
    HumanoidMACarry._build_box = build_box
    HumanoidMACarry._sample_episode_graph = sample
    HumanoidMACarry._reset_boxes = reset_boxes
    HumanoidMACarry._evaluate_ontop_context = evaluate
