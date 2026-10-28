"""Reject unsafe final starts for every agent pair, independently of task/group/RSI."""
import torch
from utils.jointpush_geometry import rotate

MODE = 'all_agent_pairs_v1'


def invalid_initialization(body_states, geometry, min_distance=.7, margin=.005):
    roots = body_states[:, :, 0, :3]
    distance = torch.cdist(roots[..., :2], roots[..., :2])
    eye = torch.eye(roots.shape[1], device=roots.device, dtype=torch.bool)[None]
    bad = ((distance < min_distance) & ~eye).any((1, 2))
    bad |= ~torch.isfinite(body_states).flatten(1).all(-1)
    bodies = body_states[:, :, geometry['ids']]
    points = bodies[..., :3] + rotate(bodies[..., 3:7], geometry['offsets'].expand(*bodies.shape[:-1], 3))
    radii = geometry['radii']
    low = (points-radii[None,None,:,None]).amin(-2)
    high = (points+radii[None,None,:,None]).amax(-2)
    threshold = radii[:, None]+radii[None, :]+margin
    for a in range(roots.shape[1]):
        for b in range(a):
            # Broad phase keeps the covering-sphere check small for distant pairs.
            nearby = ((low[:,a] <= high[:,b]+margin) & (low[:,b] <= high[:,a]+margin)).all(-1)
            rows = (nearby & ~bad).nonzero().flatten()
            if len(rows):
                gap = torch.cdist(points[rows,a], points[rows,b])
                bad[rows] |= (gap < threshold).any((1, 2))
    return bad


def install(env):
    spec = env.get('stage2JointPush', {}).get('initialization_safety')
    if spec is None:
        return
    if spec['mode'] != MODE or spec['minimum_root_distance'] != .7 or spec['body_margin'] != .005:
        raise ValueError('Unknown initialization-safety contract')
    from env.tasks.multi_agent.humanoid_ma_stage2_jointpush_rsi import HumanoidMAStage2JointPushRsi as Task
    from env.tasks.multi_agent.edge_ontop_task import SampledOnTopTaskMixin
    from utils import stage2_jointpush_scene as scene
    from utils.jointpush_rsi_geometry import geometry
    from pathlib import Path
    original_reset = Task._reset_ontop_context_envs
    original_configure = Task._configure_ontop_targets
    original_infeasible = Task._ontop_scene_infeasible

    def reset(self, ids):
        if self._push_rsi_enabled:
            return original_reset(self, ids)
        # Unlike the legacy loco-only loop, this retries the human references
        # too. Keep group/region assignment fixed throughout rejection sampling.
        return SampledOnTopTaskMixin._reset_ontop_context_envs(self, ids)

    def configure(self, ids):
        original_configure(self, ids)
        if not self._push_rsi_enabled:
            self._layout_failed[ids] = scene.sample_push(self, ids)
            scene.place_distractors(self, ids)

    def infeasible(self, ids):
        bad = original_infeasible(self, ids)
        if not self._push_rsi_enabled:
            bad |= self._layout_failed[ids]
        if not hasattr(self, '_initialization_safety_geometry'):
            if self._bank is not None:
                self._initialization_safety_geometry = self._bank.geometry
            else:
                asset = self.cfg['env']['asset']
                lib = self._motion_lib['push']
                self._initialization_safety_geometry = geometry(
                    Path(asset['assetRoot'])/asset['assetFileName'],
                    lib._motions[0].skeleton_tree.node_names, self.device)
        return bad | invalid_initialization(self._kinematic_humanoid_rigid_body_states[ids],
            self._initialization_safety_geometry, spec['minimum_root_distance'], spec['body_margin'])

    Task._reset_ontop_context_envs = reset
    Task._configure_ontop_targets = configure
    Task._ontop_scene_infeasible = infeasible
    print('KSH_INITIALIZATION_SAFETY', spec, flush=True)
