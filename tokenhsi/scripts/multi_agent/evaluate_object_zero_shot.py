#!/usr/bin/env python3
"""Evaluation-only native object swaps for the read-only KSH distill policy.

The upstream runtime is imported without bytecode writes. All evaluation hooks,
asset copies, configuration snapshots, and results live in this checkout.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = Path('/home/hwanhee/ksh/stage1_task')
RUN = ('approach_scenario_stage1_unified_size_rsi_task_embedding_distill/'
       'ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingDistill_07-15-20-48')
CHECKPOINT = UPSTREAM / 'output' / RUN / 'nn/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingDistill_00039000.pth'
ENV_CFG = 'tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill.yaml'
TRAIN_CFG = 'tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_task_embedding_distill.yaml'
CASES = {
    'sit_chair': ('sit', 'dataset_sit/objects/test/StraightChair_Normal'),
    'climb_box': ('climb', 'dataset_amass_climb/objects/test/Box'),
    'climb_cabinet': ('climb', 'dataset_amass_climb/objects/test/Cabinet'),
    'climb_table': ('climb', 'dataset_amass_climb/objects/test/Table_Square'),
    'climb_chair': ('climb', 'dataset_sit/objects/test/StraightChair_Normal'),
    'carry_chair': ('carry_at', 'dataset_dynamic_objects/test/ArmChair_Normal'),
    'carry_table': ('carry_at', 'dataset_dynamic_objects/test/Table_Circle'),
    'baseline_sit': ('sit', None),
    'baseline_climb': ('climb', None),
    'baseline_carry': ('carry_at', None),
}
DEFAULT_CASES = ('sit_chair', 'climb_cabinet', 'climb_table', 'carry_chair', 'carry_table')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def size_in_range(size, ranges):
    """Inclusive axis-wise size check; only roundoff-sized boundary slack."""
    return all(lo - 1e-6 <= value <= hi + 1e-6 for value, (lo, hi) in zip(size, ranges))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def source_hashes():
    paths = [UPSTREAM / 'tokenhsi/run.py']
    for directory in ('env', 'learning', 'utils'):
        paths.extend((UPSTREAM / 'tokenhsi' / directory).rglob('*.py'))
    paths.extend((UPSTREAM / 'tokenhsi/data/cfg').rglob('*.yaml'))
    return {str(p.relative_to(UPSTREAM)): digest(p) for p in sorted(set(paths))}


def asset_spec(directory):
    """Reject assets that break the original one-body-per-object tensor layout."""
    cfg = json.loads((directory / 'config.json').read_text())
    bbox = cfg['bbox']
    if len(bbox) != 3 or not all(0 < float(v) < 10 for v in bbox):
        raise ValueError('Invalid native bbox: ' + str(directory))
    if any(abs(float(v)) > 1e-6 for v in cfg['center']):
        raise ValueError('Only centered native assets are supported: ' + str(directory))
    xml = ET.parse(directory / 'asset.urdf').getroot()
    if len(xml.findall('link')) != 1 or xml.findall('joint'):
        raise ValueError('Evaluation requires one rigid body and no joints: ' + str(directory))
    if not xml.findall('.//collision/geometry/mesh') and not xml.findall('.//collision/geometry/box'):
        raise ValueError('An actual mesh or box collider is required: ' + str(directory))
    for mesh in xml.findall('.//mesh'):
        if not (directory / mesh.attrib['filename']).is_file():
            raise FileNotFoundError(mesh.attrib['filename'])
    return dict(id=directory.name, source=str(directory), bbox=bbox,
                target_local=cfg.get('tarSitPos'),
                mass_kg=float(xml.find('.//mass').attrib['value']))


def seat_scoring_geometry(objects, sizes, targets, num_agents):
    """Return a seat-height scoring proxy; never change policy/physics tensors."""
    import torch
    result_objects, result_sizes = objects.clone(), sizes.clone()
    surface_z = targets[:, :num_agents, 2] - .10
    bottom_z = -sizes[:, :num_agents, 2] / 2
    if (surface_z <= bottom_z).any():
        raise ValueError('Seat surface must be above the bbox bottom')
    center = torch.zeros_like(objects[:, :num_agents, :3])
    center[..., 2] = (surface_z + bottom_z) / 2
    q = objects[:, :num_agents, 3:7]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    uv = torch.cross(q[..., :3], center, dim=-1)
    rotated = center + 2 * (q[..., 3:4] * uv + torch.cross(q[..., :3], uv, dim=-1))
    result_objects[:, :num_agents, :3] += rotated
    result_sizes[:, :num_agents, 2] = surface_z - bottom_z
    return result_objects, result_sizes


def support_proxy_geometry(objects, sizes, targets, num_agents, xy_fraction=1.):
    """Ground-to-seat virtual box, centered on metadata XY; never a collider.

    tarSitPos includes 10cm pelvis clearance. XY extent is a bbox heuristic,
    not an annotated seat polygon. Preserve native origin-to-ground distance.
    """
    import torch
    if not 0 < xy_fraction <= 1:
        raise ValueError('Seat XY fraction must be in (0, 1]')
    proxy, proxy_sizes = seat_scoring_geometry(objects, sizes, targets, num_agents)
    local = targets[:, :num_agents].clone()
    local[..., 2] = (targets[:, :num_agents, 2] - .10 - sizes[:, :num_agents, 2] / 2) / 2
    q = objects[:, :num_agents, 3:7]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    uv = torch.cross(q[..., :3], local, dim=-1)
    delta = local + 2 * (q[..., 3:4] * uv + torch.cross(q[..., :3], uv, dim=-1))
    proxy[:, :num_agents, :3] = objects[:, :num_agents, :3] + delta
    proxy[:, :num_agents, 7:10] = objects[:, :num_agents, 7:10] + torch.cross(
        objects[:, :num_agents, 10:13], delta, dim=-1)
    proxy_sizes[:, :num_agents, :2] *= xy_fraction
    return proxy, proxy_sizes


def proxy_policy_observation(obs, objects, sizes, proxy, proxy_sizes, num_agents):
    """Replace O30 geometry/velocity and GTA object origins together, pre-RMS.

    Human nodes, goals, rotations, graph suffix and physical tensors stay native.
    Exact H223/O30/T1/pose7 layout is specific to the audited distill checkpoint.
    """
    import torch
    B, O = sizes.shape[:2]
    M = num_agents
    if obs.shape != (B, M * 223 + O * 30 + M + (2 * M + O) * 7 + 10):
        raise ValueError('Unexpected distill observation layout')
    result = obs.clone()
    nodes = result[:, M * 223:M * 223 + O * 30].reshape(B, O, 30)
    q = objects[..., 3:7]
    vector = proxy[..., 7:10]
    # Inverse rotation, normalized quaternions (xyzw).
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    xyz = -q[..., :3]
    uv = torch.cross(xyz, vector, dim=-1)
    nodes[:, :M, :3] = (vector + 2 * (q[..., 3:4] * uv + torch.cross(xyz, uv, dim=-1)))[:, :M]
    signs = sizes.new_tensor([[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1],
                             [1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]])
    nodes[:, :M, 6:] = (proxy_sizes[:, :M, None] / 2 * signs).reshape(B, M, 24)
    pose_start = M * 223 + O * 30 + M
    poses = result[:, pose_start:pose_start + (2 * M + O) * 7].reshape(B, 2 * M + O, 7)
    poses[:, M:M + M, :3] += proxy[:, :M, :3] - objects[:, :M, :3]
    return result


def climb_ascent_success(feet, roots, objects, sizes, foot_forces):
    """Both feet on native top footprint, pelvis above it; no center target.

    Contact is net foot contact plus top geometry, not actor-pair attribution.
    Foot body origins use the original 7cm surface tolerance. A fallen object's
    side is not its native climb surface. All inputs are world-frame tensors.
    """
    import torch
    q = objects[..., 3:7]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    def local(points):
        v = points - objects[..., None, :3]
        xyz = -q[..., None, :3].expand_as(v)
        uv = torch.cross(xyz, v, dim=-1)
        return v + 2 * (q[..., None, 3:4] * uv + torch.cross(xyz, uv, dim=-1))
    local_feet = local(feet)
    local_root = local(roots.unsqueeze(-2)).squeeze(-2)
    half = sizes / 2
    inside = (local_feet[..., :2].abs() <= half[..., None, :2]).all(-1).all(-1)
    at_top = ((local_feet[..., 2] - half[..., None, 2]).abs() <= .07).all(-1)
    standing = local_root[..., 2] >= half[..., 2] + .45
    up_z = 1 - 2 * (q[..., 0].square() + q[..., 1].square())
    geometric = inside & at_top & standing & (up_z >= .9)
    supported = geometric & (foot_forces[..., 2].clamp_min(0).sum(-1) >= 20.)
    return geometric, supported


def output_directory(path, visualize=False):
    output = Path(path).resolve()
    if ROOT not in output.parents or output == ROOT:
        raise ValueError('Output must be a new directory inside this checkout')
    if output.exists():
        if not visualize:
            raise FileExistsError('Use a new output directory: ' + str(output))
        for index in range(1, 100000):
            candidate = output.with_name(output.name + '_run{:03d}'.format(index))
            if not candidate.exists():
                return candidate
        raise FileExistsError('Too many visualization runs: ' + str(output))
    return output


def policy_observation(obs):
    """CommonPlayer needs a dictionary; env_step returns a plain tensor."""
    return obs if isinstance(obs, dict) else {'obs': obs}


def viewer_events(task, events):
    """Intercept only new controls; leave the original viewer shortcuts intact."""
    remaining = []
    for event in events:
        if event.action == 'object_reset':
            if event.value > 0:
                task._object_viewer_reset = True
                print('[object viewer] N/R received: reset requested (same asset, new placement)', flush=True)
        elif event.action == 'object_pause':
            if event.value > 0:
                task._object_viewer_pause = not task._object_viewer_pause
        else:
            remaining.append(event)
    return remaining


class ViewerGymProxy:
    """Forward to the unmodified Gym renderer, including its native UI events."""
    def __init__(self, gym, task):
        self._gym, self._task = gym, task

    def __getattr__(self, name):
        return getattr(self._gym, name)

    def query_viewer_action_events(self, viewer):
        return viewer_events(self._task, self._gym.query_viewer_action_events(viewer))


def prepare(options):
    requested = options.output.resolve()
    output = output_directory(requested, getattr(options, 'visualize', False))
    options.output = output
    if output != requested:
        print('Preserving existing output; using:', output, flush=True)
    if not options.checkpoint.is_file():
        raise FileNotFoundError(options.checkpoint)
    cases = {}
    policy_bbox = getattr(options, 'policy_bbox', 'full')
    climb_surface = getattr(options, 'climb_surface', 'full')
    seat_fraction = getattr(options, 'seat_xy_fraction', .75)
    climb_ascent = getattr(options, 'climb_ascent', False)
    range_only = getattr(options, 'climb_training_range', False)
    object_geometry = getattr(options, 'object_geometry', 'native')
    import yaml
    native_config = yaml.safe_load((UPSTREAM / ENV_CFG).read_text())
    climb_ranges = native_config['env']['box']['build']['taskSizeRanges']['CLIMB']
    for name in options.cases:
        task, relative = CASES[name]
        if climb_ascent and task != 'climb':
            raise ValueError('--climb-ascent only supports CLIMB cases')
        if (range_only or object_geometry != 'native') and (task != 'climb' or relative is None):
            raise ValueError('Size-range and matched-box options require native CLIMB cases')
        if policy_bbox != 'full' and name not in ('sit_chair', 'climb_chair'):
            raise ValueError('Seat proxies are only supported for sit_chair/climb_chair')
        if relative is None:
            cases[name] = dict(task=task, baseline=True, available_assets=None,
                              assets=[dict(id='training_distribution_box')])
            continue
        all_dirs = sorted(p.parent for p in (ROOT / 'tokenhsi/data' / relative).glob('*/asset.urdf'))
        if not all_dirs:
            raise FileNotFoundError(relative)
        total_native = len(all_dirs)
        if range_only:
            all_dirs = [p for p in all_dirs if size_in_range(asset_spec(p)['bbox'], climb_ranges)]
        all_dirs = all_dirs[options.asset_index:]
        chosen = all_dirs[:options.assets_per_category] if options.assets_per_category else all_dirs
        if not chosen:
            raise ValueError('asset-index exceeds the available objects: ' + name)
        if len(chosen) > options.envs * 2:
            raise ValueError('Increase --envs so all chosen assets occupy at least one source slot')
        cases[name] = dict(task=task, available_assets=len(all_dirs),
                           total_native_assets=total_native,
                           assets=[asset_spec(p) for p in chosen])
        if policy_bbox != 'full' or (task == 'climb' and climb_surface == 'seat'):
            if name not in ('sit_chair', 'climb_chair'):
                raise ValueError('Seat proxies are only supported for sit_chair/climb_chair')
            if any(a['target_local'] is None for a in cases[name]['assets']):
                raise ValueError('Seat metadata is required')
            if task == 'climb' and climb_surface != 'seat':
                raise ValueError('CLIMB seat proxy requires --climb-surface seat')
    output.mkdir(parents=True)
    shutil.copy2(UPSTREAM / ENV_CFG, output / 'original_env.yaml')
    shutil.copy2(UPSTREAM / TRAIN_CFG, output / 'original_train.yaml')
    for case in cases.values():
        if case.get('baseline'):
            continue
        for asset in case['assets']:
            dest = output / 'assets' / asset['id']
            if not dest.exists():
                shutil.copytree(asset['source'], dest)
            asset['local'] = str(dest)
    manifest = dict(checkpoint=str(options.checkpoint.resolve()), checkpoint_sha256=digest(options.checkpoint),
                    evaluation_code_sha256=digest(Path(__file__)),
                    upstream=str(UPSTREAM), upstream_hashes=source_hashes(),
                    protocol=dict(envs=options.envs, agents=2, objects=4, repeats=options.repeats,
                                  episode_length=options.episode_length, seed=options.seed,
                                  initial_skill='loco', scale=1.0, native_urdf_mass=True,
                                  movable_objects=True, policy_bbox=('full native object bbox' if policy_bbox == 'full'
                                      else 'virtual ground-to-seat bbox; physical mesh unchanged'),
                                  policy_bbox_mode=policy_bbox, climb_surface=climb_surface,
                                  seat_xy_fraction=seat_fraction,
                                  climb_ascent=climb_ascent,
                                  primary_metric=('ascent_ever_rate' if climb_ascent else 'final_success_rate'),
                                  climb_training_range_only=range_only, climb_training_size_ranges=climb_ranges,
                                  object_geometry=object_geometry,
                                  box_friction=getattr(options, 'box_friction', None),
                                  mass_rule='native URDF mass; matched-box density=mass/bbox_volume',
                                  inertia_rule='computed from actual collider; not matched across geometry',
                                  ascent_definition=dict(both_feet_inside_native_top_xy=True,
                                      foot_origin_height_tolerance_m=.07, pelvis_min_above_top_m=.45,
                                      object_local_z_world_z_dot_min=.9, net_feet_upward_force_min_N=20.,
                                      center_target_required=False, end_of_episode_required=False,
                                      contact_attribution='net foot forces plus bbox top geometry; not pairwise contacts'),
                                  policy_observation_size=634, object_features=30,
                                  visualize=options.visualize, viewer_episodes=options.viewer_episodes,
                                  record_video=getattr(options, 'record_video', False),
                                  scene_mode=options.scene_mode,
                                  viewer_style=options.viewer_style, viewer_camera=options.viewer_camera,
                                  reset_contact_cache_flush='park mesh actors for two physics ticks, then restore all root/DOF states before timed rollout',
                                  sit_scoring='native tarSitPos - 0.10 + trained pelvis clearance 0.12',
                                  sit_xy_region='full bbox with original inner margin; not a seat polygon',
                                  climb_scoring=('metadata seat surface; full bbox XY heuristic' if climb_surface == 'seat'
                                                 else 'original full bbox top'),
                                  carry_scoring='original AT goal; lift and elevated travel reported separately'),
                    cases=cases)
    save(output / 'manifest.json', manifest)
    return manifest


def install_runtime_hooks(case, manifest, output):
    # This function only runs in the evaluation child process, after Isaac Gym import.
    import torch
    from isaacgym import gymapi
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges
    from env.tasks.multi_agent.edge_ontop_reward import vertical_extent
    from utils import task_role_spec

    original_build = HumanoidMACarry._build_box

    def build_box(self, env_id, object_id, env_ptr):
        original_build(self, env_id, object_id, env_ptr)
        friction = manifest['protocol'].get('box_friction')
        if friction is None:
            return
        if not (self.cfg['args'].test and self.cfg['args'].eval):
            raise ValueError('Friction override is evaluation-only')
        handle = self._box_handles[-1]
        shapes = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
        for shape in shapes:
            shape.friction = friction
        self.gym.set_actor_rigid_shape_properties(env_ptr, handle, shapes)
        actual = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
        if not all(abs(shape.friction - friction) < 1e-6 for shape in actual):
            raise RuntimeError('Physical box friction override failed')
        if env_id == 0:
            path = output / 'friction_override.json'
            audit = json.loads(path.read_text()) if path.exists() else dict(
                evaluation_only=True, box_friction=friction, plane=self.cfg['env']['plane'], boxes=[])
            audit['boxes'].append(dict(object_id=object_id,
                actual_shape_friction=[shape.friction for shape in actual]))
            save(path, audit)

    HumanoidMACarry._build_box = build_box

    original_load = HumanoidMACarry._load_box_asset
    original_evaluate = HumanoidMACarry._evaluate_ontop_context
    original_clean_obs = HumanoidMACarry._compute_clean_scene_obs
    task_name = case['task']
    assets = case['assets']
    baseline = case.get('baseline', False)
    protocol = manifest['protocol']
    policy_bbox = protocol.get('policy_bbox_mode', 'full')
    seat_climb = task_name == 'climb' and protocol.get('climb_surface', 'full') == 'seat'

    def clean_obs(self, env_ids=None):
        obs = original_clean_obs(self, env_ids)
        if policy_bbox == 'full' or baseline:
            return obs
        objects = self._logical_box_values(self._box_states, env_ids)
        sizes = self._logical_box_values(self._box_size, env_ids)
        targets = self._logical_box_values(self._zero_shot_target_local, env_ids)
        fraction = protocol['seat_xy_fraction'] if policy_bbox == 'seat-inset' else 1.
        proxy, proxy_sizes = support_proxy_geometry(objects, sizes, targets, self.num_agents, fraction)
        result = proxy_policy_observation(obs, objects, sizes, proxy, proxy_sizes, self.num_agents)
        if not hasattr(self, '_seat_proxy_audited'):
            save(output / 'policy_proxy_audit.json', dict(mode=policy_bbox,
                observation_shape=list(obs.shape),
                changed_columns=(result[0] != obs[0]).nonzero().flatten().cpu().tolist(),
                native_sizes=sizes[0].cpu().tolist(), proxy_sizes=proxy_sizes[0].cpu().tolist(),
                native_states=objects[0].cpu().tolist(), proxy_states=proxy[0].cpu().tolist()))
            self._seat_proxy_audited = True
        return result

    def fixed_probabilities(sizes, carry_only=False):
        # The original function filters out task/size pairs unseen during training.
        # Fixed evaluation tasks must remain testable for the native object sizes.
        probs = torch.zeros(*sizes.shape[:-1], 5, device=sizes.device)
        probs[..., {'sit': 1, 'climb': 2, 'carry_at': 3}[task_name]] = 1
        return probs

    task_role_spec.task_size_probabilities = fixed_probabilities

    def load_objects(self):
        if not (self.cfg['args'].test and self.cfg['args'].eval):
            raise ValueError('Native object hooks are evaluation-only')
        original_load(self)
        N, M, O = self.num_envs, self.num_agents, self.num_objects
        self._zero_shot_asset_ids = torch.arange(N * M, device=self.device).reshape(N, M) % len(assets)
        self._zero_shot_target_local = torch.zeros(N, O, 3, device=self.device)
        if baseline:
            return
        loaded = []
        for asset in assets:
            opts = gymapi.AssetOptions()
            opts.fix_base_link = False
            opts.angular_damping = .01
            opts.linear_damping = .01
            opts.max_angular_velocity = 100.
            opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            opts.use_mesh_materials = True
            opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            opts.vhacd_enabled = True
            opts.vhacd_params.resolution = 100000
            opts.vhacd_params.max_convex_hulls = 128
            opts.vhacd_params.max_num_vertices_per_ch = 64
            opts.override_inertia = True
            if protocol.get('object_geometry', 'native') == 'matched-box':
                x, y, z = asset['bbox']
                opts.density = asset['mass_kg'] / (x * y * z)
                handle = self.gym.create_box(self.sim, x, y, z, opts)
            else:
                handle = self.gym.load_asset(self.sim, asset['local'], 'asset.urdf', opts)
            if handle is None or self.gym.get_asset_rigid_body_count(handle) != 1:
                raise RuntimeError('Object actor must have exactly one rigid body')
            if self.gym.get_asset_rigid_shape_count(handle) < 1:
                raise RuntimeError('Object has no collision shapes')
            loaded.append(handle)
        for env in range(N):
            for agent in range(M):
                idx = int(self._zero_shot_asset_ids[env, agent])
                self._box_assets[env * O + agent] = loaded[idx]
                self._box_size[env, agent] = torch.tensor(assets[idx]['bbox'], device=self.device)
                if task_name == 'sit' or seat_climb:
                    self._zero_shot_target_local[env, agent] = torch.tensor(assets[idx]['target_local'], device=self.device)
        self._box_scale = self._box_size / torch.tensor(self._build_base_size, device=self.device)

    def evaluate(self, env_ids=None):
        if (task_name != 'sit' and not seat_climb) or baseline:
            return original_evaluate(self, env_ids)
        from utils.edge_ontop_spec import select_graph
        bodies = self._rigid_body_pos if env_ids is None else self._kinematic_humanoid_rigid_body_states[env_ids, ..., :3]
        objects = self._logical_box_values(self._box_states, env_ids)
        sizes = self._logical_box_values(self._box_size, env_ids)
        targets = self._logical_box_values(self._zero_shot_target_local, env_ids)
        # tarSitPos contains a 0.10m clearance in the data generator. Recover
        # the seat surface, then let the original evaluator add trained 0.12m.
        if seat_climb or policy_bbox != 'full' or protocol.get('climb_surface') == 'seat':
            objects, sizes = support_proxy_geometry(objects, sizes, targets, self.num_agents)
        else:
            objects, sizes = seat_scoring_geometry(objects, sizes, targets, self.num_agents)
        graph = self.relation_runtime.graph if env_ids is None else select_graph(self.relation_runtime.graph, env_ids)
        goals = self._tar_pos if env_ids is None else self._tar_pos[env_ids]
        result = evaluate_interaction_edges(bodies[..., self._key_body_ids[[0, 1]], :],
            bodies[..., self._key_body_ids[[2, 3]], :], bodies[..., 0, :], objects, sizes,
            goals, graph, self._relation_cfg, self._char_h)
        if env_ids is None:
            self._ontop_last_diag = result[1]
        return result

    HumanoidMACarry._load_box_asset = load_objects
    HumanoidMACarry._evaluate_ontop_context = evaluate
    HumanoidMACarry._compute_clean_scene_obs = clean_obs

    from learning.multi_agent import edge_context_eval
    from env.tasks.multi_agent.edge_context_reward import goal_success
    from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success

    def clear_reset_contacts(task):
        from isaacgym import gymtorch
        saved_roots = task._root_states.clone()
        saved_dofs = task._dof_state.clone()
        mesh_actor_ids = task._box_actor_ids[:, :task.num_agents].reshape(-1).long()
        task._root_states[mesh_actor_ids, 2] += 50.
        task.gym.set_actor_root_state_tensor(task.sim, gymtorch.unwrap_tensor(task._root_states))
        for _ in range(2):
            task.gym.simulate(task.sim)
            task.gym.fetch_results(task.sim, True)
        task.gym.refresh_actor_root_state_tensor(task.sim)
        task.gym.refresh_dof_state_tensor(task.sim)
        task._root_states.copy_(saved_roots)
        task._dof_state.copy_(saved_dofs)
        task.gym.set_actor_root_state_tensor(task.sim, gymtorch.unwrap_tensor(task._root_states))
        task.gym.set_dof_state_tensor(task.sim, gymtorch.unwrap_tensor(task._dof_state))

    def visualize(player):
        task = player.env.task
        if task.viewer is None or task.num_envs != 1:
            raise ValueError('Visualization requires HEADLESS=0 and --envs 1')
        task.gym.subscribe_viewer_keyboard_event(task.viewer, gymapi.KEY_N, 'object_reset')
        task.gym.subscribe_viewer_keyboard_event(task.viewer, gymapi.KEY_R, 'object_reset')
        task.gym.subscribe_viewer_keyboard_event(task.viewer, gymapi.KEY_SPACE, 'object_pause')
        task._object_viewer_reset = False
        task._object_viewer_pause = False
        task.gym = ViewerGymProxy(task.gym, task)

        def render(self, sync_frame_time=False):
            if self.gym.query_viewer_has_closed(self.viewer):
                raise SystemExit(0)
            for event in self.gym.query_viewer_action_events(self.viewer):
                if event.value <= 0:
                    continue
                if event.action == 'QUIT':
                    raise SystemExit(0)
                if event.action == 'object_reset':
                    self._object_viewer_reset = True
                elif event.action == 'object_pause':
                    self._object_viewer_pause = not self._object_viewer_pause
                elif event.action == 'follow':
                    self._camera_follow_flag = not self._camera_follow_flag
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            if self._camera_follow_flag:
                points = torch.stack([self._humanoid_root_states[0, 0, :3], self._box_states[0, 0, :3]])
                if task_name == 'carry_at':
                    points = torch.cat([points, self._tar_pos[0, :1]], dim=0)
                center = points.mean(0).detach().cpu().tolist()
                radius = max(float((points.amax(0)-points.amin(0))[:2].norm()) * .65, 2.)
                self.gym.viewer_camera_look_at(self.viewer, None,
                    gymapi.Vec3(center[0]+radius, center[1]-radius*1.35, 1.4+radius*.65),
                    gymapi.Vec3(center[0], center[1], .7))
            self.gym.draw_viewer(self.viewer, self.sim, True)

        if manifest['protocol'].get('viewer_style', 'native') == 'focus':
            task.render = types.MethodType(render, task)
        episode = 0
        limit = manifest['protocol'].get('viewer_episodes', 0)
        frame_seconds = task.gym.get_sim_params(task.sim).dt * task.control_freq_inv
        manual = manifest['protocol'].get('scene_mode', 'manual') == 'manual'
        print('[object viewer] task={} asset={} Space=pause, N/R=new placement, F=camera follow, Esc=quit'.format(
            task_name, assets[0]['id']), flush=True)
        while not limit or episode < limit:
            rows = torch.arange(task.num_envs, device=player.device) * task.num_agents
            obs = policy_observation(player.env_reset(rows))
            player.get_batch_size(obs['obs'], 1)
            clear_reset_contacts(task)
            task._object_viewer_reset = False
            task._object_viewer_pause = False
            print('[object viewer] scene={} reset complete; asset={}'.format(episode+1, assets[0]['id']), flush=True)
            step = 0
            recording = manifest['protocol'].get('record_video', False)
            video = None
            frame_path = output / '.record_frame.png'
            if recording:
                import cv2
            try:
                while True:
                    start = time.monotonic()
                    if task._object_viewer_pause or (manual and step >= task.max_episode_length):
                        task.render()
                        if task._object_viewer_reset:
                            break
                        time.sleep(max(0., frame_seconds - (time.monotonic() - start)))
                        continue
                    action = player.get_action(obs, player.is_determenistic)
                    if not torch.isfinite(action).all():
                        raise RuntimeError('Nonfinite policy action')
                    obs, _, done, _ = player.env_step(player.env, action)
                    obs = policy_observation(obs)
                    if recording:
                        task.gym.write_viewer_image_to_file(task.viewer, str(frame_path))
                        frame = cv2.imread(str(frame_path))
                        if frame is None:
                            raise RuntimeError('Could not read viewer recording frame')
                        if video is None:
                            height, width = frame.shape[:2]
                            video = cv2.VideoWriter(str(output / ('scene_%03d.mp4' % (episode + 1))),
                                cv2.VideoWriter_fourcc(*'mp4v'), 1. / frame_seconds, (width, height))
                            if not video.isOpened():
                                raise RuntimeError('Could not open viewer MP4 writer')
                        video.write(frame)
                    if episode == 0 and step in (0, 60):
                        task.gym.write_viewer_image_to_file(task.viewer, str(output / ('initial.png' if step == 0 else 'preview.png')))
                    step += 1
                    if task._object_viewer_reset or (not manual and (done.any() or step >= task.max_episode_length)):
                        break
                    time.sleep(max(0., frame_seconds - (time.monotonic() - start)))
            finally:
                if video is not None:
                    video.release()
                    print('[record] saved ' + str(output / ('scene_%03d.mp4' % (episode + 1))), flush=True)
                if recording and frame_path.exists():
                    frame_path.unlink()
            episode += 1
        return {}

    @torch.no_grad()
    def run_eval(player):
        task = player.env.task
        N, M = task.num_envs, task.num_agents
        audit = dict(observation_size=task.num_obs, object_features=task.get_clean_object_obs_size(),
                     bbox=task._box_size.detach().cpu().tolist(), assets=[],
                     actor_count=task.gym.get_actor_count(task.envs[0]))
        if (audit['observation_size'], audit['object_features']) != (634, 30):
            raise RuntimeError('Unexpected policy input dimensions: ' + str(audit))
        for asset, handle in zip(assets, loaded_handles(task)):
            audit['assets'].append(dict(id=asset['id'], rigid_bodies=task.gym.get_asset_rigid_body_count(handle),
                collision_shapes=task.gym.get_asset_rigid_shape_count(handle)))
        audit['physical_sources_env0'] = []
        for agent in range(M):
            actor = task._box_handles[agent]
            body = task.gym.get_actor_rigid_body_properties(task.envs[0], actor)[0]
            shapes = task.gym.get_actor_rigid_shape_properties(task.envs[0], actor)
            audit['physical_sources_env0'].append(dict(mass=body.mass,
                inertia=[body.inertia.x.x, body.inertia.y.y, body.inertia.z.z],
                friction=[p.friction for p in shapes]))
        audit['physical_sources_by_asset'] = {}
        if not baseline:
            for env in range(N):
                for agent in range(M):
                    idx = int(task._zero_shot_asset_ids[env, agent])
                    asset = assets[idx]
                    if asset['id'] in audit['physical_sources_by_asset']:
                        continue
                    actor = task._box_handles[env * task.num_objects + agent]
                    body = task.gym.get_actor_rigid_body_properties(task.envs[env], actor)[0]
                    shapes = task.gym.get_actor_rigid_shape_properties(task.envs[env], actor)
                    audit['physical_sources_by_asset'][asset['id']] = dict(mass=body.mass,
                        bbox=asset['bbox'], friction=sorted({float(p.friction) for p in shapes}),
                        com=[body.com.x, body.com.y, body.com.z],
                        inertia_diagonal=[body.inertia.x.x, body.inertia.y.y, body.inertia.z.z])
        save(output / 'runtime_audit.json', audit)
        if manifest['protocol'].get('visualize', False):
            return visualize(player)
        results, trials = {}, []
        for repeat in range(task.cfg['args'].eval_repeats):
            obs = player.env_reset()
            player.get_batch_size(obs['obs'], 1)
            # Convex compounds with a nonzero COM can retain pre-teleport contact
            # manifolds in GPU PhysX. Remove those contacts before the rollout;
            # restore the complete scene and DOF state, preserving the start.
            clear_reset_contacts(task)
            initial = task._box_states[:, :M, :3].clone()
            if repeat == 0:
                save(output / 'initial_state.json', dict(
                    objects=task._box_states.detach().cpu().tolist(),
                    goals=task._tar_pos.detach().cpu().tolist(),
                    humanoid_roots=task._humanoid_root_states.detach().cpu().tolist()))
            previous = initial.clone()
            alive = torch.ones(N, device=player.device, dtype=torch.bool)
            ever = torch.zeros(N, M, device=player.device, dtype=torch.bool)
            raw_ever = torch.zeros_like(ever)
            elevated_travel = torch.zeros(N, M, device=player.device)
            max_bottom = torch.zeros_like(elevated_travel)
            high_feet_steps = torch.zeros_like(elevated_travel)
            live_steps = torch.zeros(N, device=player.device)
            ascent_ever = torch.zeros_like(ever)
            ascent_geometry_ever = torch.zeros_like(ever)
            ascent_streak = torch.zeros(N, M, device=player.device, dtype=torch.long)
            ascent_max_streak = torch.zeros_like(ascent_streak)
            ascent_first_step = torch.full_like(ascent_streak, -1)
            resets = []
            trace = []
            for step in range(task.max_episode_length + 1):
                obs = player.env_reset(resets)
                if not torch.isfinite(obs['obs']).all():
                    raise RuntimeError('Nonfinite policy observation')
                action = player.get_action(obs, player.is_determenistic)
                if not torch.isfinite(action).all():
                    raise RuntimeError('Nonfinite policy action')
                obs, _, done, info = player.env_step(player.env, action)
                current = task.relation_runtime.done.clone()
                raw_phi, raw_diag = original_evaluate(task)
                raw_success = interaction_own_success(raw_phi, raw_diag['z_error'], raw_diag['feet_height_error'],
                    task.relation_runtime.graph, task._relation_cfg, raw_diag['region_error'])
                raw_current = goal_success(raw_success, task.relation_runtime.graph)
                ever |= current & alive[:, None]
                raw_ever |= raw_current & alive[:, None]
                state = task._box_states[:, :M]
                if protocol.get('climb_ascent', False):
                    feet_ids = task._key_body_ids[[2, 3]]
                    source_objects = task._assigned_box_values(task._box_states)
                    source_sizes = task._assigned_box_values(task._box_size)
                    # Optional seat-surface diagnostics use the same support
                    # geometry as scoring, without changing the collider/policy.
                    if seat_climb:
                        source_targets = task._assigned_box_values(task._zero_shot_target_local)
                        source_objects, source_sizes = support_proxy_geometry(
                            source_objects, source_sizes, source_targets, M)
                    geometry, supported = climb_ascent_success(
                        task._rigid_body_pos[..., feet_ids, :], task._rigid_body_pos[..., 0, :],
                        source_objects, source_sizes, task._contact_forces[..., feet_ids, :])
                    supported &= alive[:, None]
                    ascent_first_step = torch.where(supported & ~ascent_ever,
                        torch.full_like(ascent_first_step, step + 1), ascent_first_step)
                    ascent_ever |= supported
                    ascent_geometry_ever |= geometry & alive[:, None]
                    ascent_streak = torch.where(supported, ascent_streak + 1, torch.zeros_like(ascent_streak))
                    ascent_max_streak = torch.maximum(ascent_max_streak, ascent_streak)
                if case.get('baseline') is not True and (task_name == 'sit' or seat_climb):
                    from isaacgym.torch_utils import quat_rotate_inverse
                    feet = task._rigid_body_pos[..., task._key_body_ids[[2, 3]], :]
                    chair = task._assigned_box_values(task._box_states)
                    targets = task._assigned_box_values(task._zero_shot_target_local)
                    q = chair[..., 3:7].unsqueeze(-2).expand(-1, -1, 2, -1)
                    local_feet = quat_rotate_inverse(q.reshape(-1, 4),
                        (feet - chair[..., None, :3]).reshape(-1, 3)).reshape(N, M, 2, 3)
                    size = task._assigned_box_values(task._box_size)
                    inside = (local_feet[..., :2].abs() <= size[..., None, :2] / 2).all(-1)
                    high = (local_feet[..., 2] > targets[..., None, 2] - .10 + .15) & inside
                    high_feet_steps += high.any(-1) * alive[:, None]
                    live_steps += alive
                if step < 12:
                    trace.append(dict(step=step+1, sources=state.detach().cpu().tolist(),
                        contact_force=task._ontop_box_contact[:, :M].detach().cpu().tolist()))
                bottom = state[..., 2] - vertical_extent(state[..., 3:7], task._box_size[:, :M] / 2)
                max_bottom = torch.where(alive[:, None], torch.maximum(max_bottom, bottom), max_bottom)
                distance = (state[..., :2] - previous[..., :2]).norm(dim=-1)
                elevated_travel += distance * (bottom > .05) * alive[:, None]
                previous.copy_(state[..., :3])
                ending = done.reshape(N, M).bool().any(-1)
                collect = ending & alive
                for env in collect.nonzero(as_tuple=False).flatten().tolist():
                    for agent in range(M):
                        asset = assets[int(task._zero_shot_asset_ids[env, agent])]
                        trials.append(dict(repeat=repeat, env=env, agent=agent, asset=asset['id'],
                            final_success=bool(current[env, agent]), ever_success=bool(ever[env, agent]),
                            bbox_rule_final_success=bool(raw_current[env, agent]),
                            bbox_rule_ever_success=bool(raw_ever[env, agent]),
                            max_bbox_bottom_m=float(max_bottom[env, agent]),
                            elevated_xy_travel_m=float(elevated_travel[env, agent]),
                            final_xy_goal_distance_m=float((state[env, agent, :2]-task._tar_pos[env, agent, :2]).norm()),
                            final_z_goal_error_m=float((state[env, agent, 2]-task._tar_pos[env, agent, 2]).abs()),
                            final_xy_displacement_m=float((state[env, agent, :2]-initial[env, agent, :2]).norm()),
                            terminated=bool(info['terminate'].reshape(N, M)[env].bool().any()), end_step=step+1))
                        if not baseline and (task_name == 'sit' or seat_climb):
                            trials[-1]['foot_above_seat_15cm_fraction'] = float(
                                high_feet_steps[env, agent] / live_steps[env].clamp_min(1))
                        if protocol.get('climb_ascent', False):
                            trials[-1].update(ascent_ever=bool(ascent_ever[env, agent]),
                                ascent_geometry_ever=bool(ascent_geometry_ever[env, agent]),
                                ascent_first_step=int(ascent_first_step[env, agent]),
                                ascent_max_consecutive_steps=int(ascent_max_streak[env, agent]),
                                ascent_3steps=bool(ascent_max_streak[env, agent] >= 3))
                alive &= ~ending
                resets = ending.nonzero(as_tuple=False).flatten() * M
                if not alive.any():
                    break
            if alive.any():
                raise RuntimeError('Evaluation exceeded horizon')
            rows = [r for r in trials if r['repeat'] == repeat]
            metrics = {k + '_rate': sum(r[k] for r in rows) / len(rows)
                       for k in ('final_success', 'ever_success', 'bbox_rule_final_success', 'bbox_rule_ever_success', 'terminated')}
            metrics['num_agent_trials'] = len(rows)
            if protocol.get('climb_ascent', False):
                for key in ('ascent_ever', 'ascent_geometry_ever', 'ascent_3steps'):
                    metrics[key + '_rate'] = sum(r[key] for r in rows) / len(rows)
                metrics['ascent_without_strict_ever_rate'] = sum(
                    r['ascent_ever'] and not r['ever_success'] for r in rows) / len(rows)
            metrics['lifted_5cm_ever_rate'] = sum(r['max_bbox_bottom_m'] > .05 for r in rows) / len(rows)
            metrics['elevated_xy_travel_mean_m'] = sum(r['elevated_xy_travel_m'] for r in rows) / len(rows)
            if rows and 'foot_above_seat_15cm_fraction' in rows[0]:
                metrics['foot_above_seat_15cm_fraction'] = sum(r['foot_above_seat_15cm_fraction'] for r in rows) / len(rows)
            if task_name == 'carry_at':
                metrics['final_xy_within_10cm_rate'] = sum(r['final_xy_goal_distance_m'] <= .1 for r in rows) / len(rows)
                metrics['final_xy_goal_distance_mean_m'] = sum(r['final_xy_goal_distance_m'] for r in rows) / len(rows)
                metrics['final_z_goal_error_mean_m'] = sum(r['final_z_goal_error_m'] for r in rows) / len(rows)
            results['repeat_' + str(repeat)] = metrics
            print('[object zero shot]', metrics, flush=True)
            save(output / 'results.json', results)
            save(output / 'trials.json', trials)
            save(output / ('first_steps_repeat_' + str(repeat) + '.json'), trace)
        return results

    def loaded_handles(task):
        # All selected assets occupy at least one of the two source slots.
        found = {}
        for env in range(task.num_envs):
            for agent in range(task.num_agents):
                found[int(task._zero_shot_asset_ids[env, agent])] = task._box_assets[env * task.num_objects + agent]
        return [found[i] for i in range(len(assets))]

    edge_context_eval.run_edge_context_eval = run_eval


def worker(output, name, headless):
    sys.dont_write_bytecode = True
    manifest = json.loads((output / 'manifest.json').read_text())
    # Match the checkpoint's runtime without modifying or writing to KSH.
    sys.path.insert(0, str(UPSTREAM))
    sys.path.insert(0, str(UPSTREAM / 'tokenhsi'))
    from isaacgym import gymapi  # Isaac Gym must precede torch.
    import runpy
    case = manifest['cases'][name]
    destination = output / name
    destination.mkdir(exist_ok=False)
    install_runtime_hooks(case, manifest, destination)
    p = manifest['protocol']
    sys.argv = ['run.py', '--task', 'HumanoidMACarry', '--cfg_train', str(output / 'original_train.yaml'),
        '--cfg_env', str(output / 'original_env.yaml'), '--motion_file', 'tokenhsi/data/dataset_loco_sit_carry_climb.yaml',
        '--checkpoint', manifest['checkpoint'], '--num_agents', '2', '--num_objects', '4',
        '--num_envs', str(p['envs']), '--eval_repeats', str(p['repeats']), '--episode_length', str(p['episode_length']),
        '--seed', str(p['seed']), '--eval_skills', 'loco', '--eval_skill_probs', '1.0',
        '--task_graph', case['task'], '--task_camera', p.get('viewer_camera', 'agent'), '--output_path', str(destination),
        '--no_video', '--test', '--eval']
    if headless:
        sys.argv.append('--headless')
    os.chdir(ROOT)
    runpy.run_path(str(UPSTREAM / 'tokenhsi/run.py'), run_name='__main__')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/object_zero_shot_39000')
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(DEFAULT_CASES))
    parser.add_argument('--assets-per-category', type=int, default=1, help='0 selects all test assets; default is a representative pilot')
    parser.add_argument('--asset-index', type=int, default=0, help='First native test asset index, sorted by ID')
    parser.add_argument('--policy-bbox', choices=('full', 'seat-height', 'seat-inset'), default='full',
                        help='Evaluation-only policy geometry; native collider remains unchanged')
    parser.add_argument('--seat-xy-fraction', type=float, default=.75,
                        help='Heuristic XY fraction for seat-inset; not an annotated seat polygon')
    parser.add_argument('--climb-surface', choices=('full', 'seat'), default='full',
                        help='Seat uses tarSitPos minus 10cm for CLIMB scoring in every comparison arm')
    parser.add_argument('--climb-ascent', action='store_true',
                        help='Report ever reached supported top footprint without a pelvis center target')
    parser.add_argument('--climb-training-range', action='store_true',
                        help='Select native CLIMB assets already inside saved training XYZ ranges; no scaling')
    parser.add_argument('--object-geometry', choices=('native', 'matched-box'), default='native',
                        help='matched-box replaces collider with a box of identical bbox and native mass')
    parser.add_argument('--box-friction', type=float, default=None,
                        help='Evaluation-only physical box shape friction; default preserves upstream')
    parser.add_argument('--visualize', action='store_true', help='Replay only, without success aggregation')
    parser.add_argument('--record-video', action='store_true',
                        help='Save each viewer scene as an MP4, using the actual viewer camera')
    parser.add_argument('--viewer-episodes', type=int, default=0, help='Visualization episodes; 0 loops until closed')
    parser.add_argument('--scene-mode', choices=('manual', 'loop'), default='manual',
                        help='manual keeps the scene until N; loop automatically restarts after termination')
    parser.add_argument('--viewer-style', choices=('native', 'focus'), default='native',
                        help='native preserves the original Isaac Gym renderer and keyboard shortcuts')
    parser.add_argument('--viewer-camera', choices=('stack', 'agent'), default='stack')
    parser.add_argument('--envs', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--episode-length', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', choices=list(CASES), help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        worker(options.output.resolve(), options.worker, options.headless)
        return
    if options.episode_length is None:
        options.episode_length = 18000 if options.visualize else 600
    if min(options.envs, options.repeats, options.episode_length) < 1 or options.assets_per_category < 0:
        parser.error('Trial counts must be positive; assets-per-category must be nonnegative')
    if options.asset_index < 0 or options.viewer_episodes < 0:
        parser.error('Asset index and viewer episode count must be nonnegative')
    if not 0 < options.seat_xy_fraction <= 1:
        parser.error('Seat XY fraction must be in (0, 1]')
    if options.box_friction is not None and (not math.isfinite(options.box_friction) or options.box_friction < 0):
        parser.error('Box friction must be finite and nonnegative')
    if options.record_video and not options.visualize:
        parser.error('--record-video requires --visualize (VNC wrapper adds it)')
    if options.visualize and (options.headless or options.envs != 1):
        parser.error('Visualization requires HEADLESS=0 and --envs 1')
    manifest = prepare(options)
    if options.prepare_only:
        print('Prepared:', options.output)
        return
    output = options.output.resolve()
    child_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    statuses = {}
    try:
        for name in options.cases:
            cmd = [sys.executable, str(Path(__file__).resolve()), '--output', str(output), '--worker', name]
            if options.headless:
                cmd.append('--headless')
            print('Visualizing:' if options.visualize else 'Evaluating:', name, flush=True)
            with (output / (name + '.log')).open('w') as log:
                result = subprocess.run(cmd, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT)
            statuses[name] = result.returncode
            if result.returncode:
                print((output / (name + '.log')).read_text()[-4500:], flush=True)
                raise RuntimeError('{} failed: {} (log: {})'.format(
                    'Viewer' if options.visualize else 'Evaluation', name, output / (name + '.log')))
    finally:
        after = source_hashes()
        integrity = dict(upstream_code_and_configs_unchanged=after == manifest['upstream_hashes'],
                         checkpoint_unchanged=digest(options.checkpoint) == manifest['checkpoint_sha256'])
        save(output / 'integrity.json', integrity)
        save(output / 'statuses.json', statuses)
        if not all(integrity.values()):
            raise RuntimeError('Read-only source/checkpoint integrity check failed')
    if options.visualize:
        return
    summary = {name: json.loads((output / name / 'results.json').read_text()) for name in options.cases}
    save(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
