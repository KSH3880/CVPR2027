"""Opt-in jhh Partial CONTACT shaping; AT, gate and scene initialization stay native.

The two geometry functions below are copied verbatim from the read-only jhh
snapshot and partial/reward.py. No imports or writes to those trees at runtime.
"""
import hashlib
from pathlib import Path
import torch
from isaacgym.torch_utils import quat_rotate, quat_conjugate

MODE = 'coupled_jhh_solo_ksh_contact_v1'


def contract():
    return dict(mode=MODE, scope='coupled_only', contact_height=1.04, midpoint_only=True,
                approach_stand_off=1., clearance=.45,
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def contact_geometry(hands, box, size, hand_force, box_force, midpoint_only=False, contact_height=.6175):
    q = box[..., 3:7]
    normal = quat_rotate(q.reshape(-1, 4), torch.tensor([0., 1., 0.], device=q.device).expand(q.numel()//4, -1)).reshape_as(box[..., :3])
    local_point = torch.zeros_like(box[..., :3])
    local_point[..., 1] = -size[..., 1]/2
    local_point[..., 2] = contact_height - size[..., 2]/2
    point = box[..., :3] + quat_rotate(q.reshape(-1, 4), local_point.reshape(-1, 3)).reshape_as(local_point)
    local_hands = quat_rotate(quat_conjugate(q)[..., None, :].expand(*hands.shape[:-1], 4).reshape(-1, 4),
                              (hands-box[..., None, :3]).reshape(-1, 3)).reshape_as(hands)
    excess_x = (local_hands[..., 0].abs()-size[..., None, 0]/2).clamp_min(0)
    excess_z = (local_hands[..., 2].abs()-size[..., None, 2]/2).clamp_min(0)
    face_distances = (excess_x.square()+excess_z.square()+(local_hands[..., 1]+size[..., None, 1]/2).square()).sqrt()
    delta = hands.mean(-2)-point
    error_xy = delta[..., :2].norm(dim=-1)
    if not midpoint_only:
        error_xy = torch.maximum(error_xy, face_distances.amax(-1))
    error_z = (delta[..., 2].abs()-.0325).clamp_min(0)
    phi = torch.exp(-10*error_xy.square()-(error_z/.04).square())
    hit = ((face_distances <= .18).all(-1) & (delta[..., :2].norm(dim=-1) <= .18)
           & (delta[..., 2].abs() <= .0325+1e-6)
           & ((hand_force * -normal[..., None, :]).sum(-1) >= .5).all(-1)
           & ((box_force*normal).sum(-1) >= .5))
    return phi, hit, point, normal


def approach_distance(roots, centers, normals, half_size, target_x, stand_off=1., clearance=.45):
    """Baseline obstacle-aware approach route, ending at the assigned rear region."""
    delta=roots[..., :2]-centers[..., :2]
    tangent=torch.stack((normals[..., 1],-normals[..., 0]),-1)
    start=torch.stack(((delta*tangent).sum(-1),(delta*normals[..., :2]).sum(-1)),-1)
    extent=half_size[..., :2]+clearance
    target=torch.stack((target_x,-half_size[..., 1]-stand_off),-1)
    left=torch.stack((-extent[..., 0],-extent[..., 1]),-1)
    right=torch.stack((extent[..., 0],-extent[..., 1]),-1)
    back_left=left.clone();back_left[...,1]*=-1
    back_right=right.clone();back_right[...,1]*=-1
    destinations=torch.stack((target,left,right,back_left,back_right),-2)
    remaining=torch.stack((torch.zeros_like(target_x),(left-target).norm(dim=-1),(right-target).norm(dim=-1),2*extent[...,1]+(left-target).norm(dim=-1),2*extent[...,1]+(right-target).norm(dim=-1)),-1)
    segment=destinations-start[...,None,:]
    parallel=segment.abs()<1e-7
    denominator=torch.where(parallel,torch.ones_like(segment),segment)
    lo=(-extent[...,None,:]+1e-4-start[...,None,:])/denominator
    hi=(extent[...,None,:]-1e-4-start[...,None,:])/denominator
    inside=start[...,None,:].abs()<extent[...,None,:]-1e-4
    enter=torch.where(parallel,torch.where(inside,-torch.inf,torch.inf),torch.minimum(lo,hi)).amax(-1)
    leave=torch.where(parallel,torch.where(inside,torch.inf,-torch.inf),torch.maximum(lo,hi)).amin(-1)
    blocked=torch.maximum(enter,torch.zeros_like(enter))<torch.minimum(leave,torch.ones_like(leave))
    paths=(segment.norm(dim=-1)+remaining).masked_fill(blocked,torch.inf)
    distance=paths.amin(-1)
    return torch.where(torch.isfinite(distance),distance,(start-target).norm(dim=-1))


def values(task, ids, bodies):
    from utils.stage2_jointpush import region_states
    boxes = task._assigned_box_values(task._box_states, ids)
    sizes = task._assigned_box_values(task._box_size, ids)
    regions, cells = region_states(boxes, sizes, task._members[ids], task._regions[ids])
    hands = bodies[:, :, task._key_body_ids[:2]]
    hand_force = task._contact_forces[ids][:, :, task._key_body_ids[:2]]
    box_force = task._assigned_box_values(task._ontop_box_contact, ids)
    state, hit, point, normal = contact_geometry(
        hands, regions, cells, hand_force, box_force,
        midpoint_only=True, contact_height=1.04)
    target_x = cells[..., 0] * (task._regions[ids] + .5) - sizes[..., 0] / 2
    distance = approach_distance(bodies[:, :, 0], boxes[..., :3], normal,
                                 sizes / 2, target_x, 1., .45)
    delta = hands.mean(-2) - point
    return state, hit, 1 / (1 + distance), delta[..., :2].norm(dim=-1), delta[..., 2].abs(), distance


def install(env, task_class=None):
    config = env.get('stage2JointPush', {}).get('contact_reward')
    if config is None:
        return
    if config != contract():
        raise ValueError('Unknown or stale legacy CONTACT contract')
    if task_class is None:
        from env.tasks.multi_agent.humanoid_ma_stage2_jointpush import HumanoidMAStage2JointPush
        task_class = HumanoidMAStage2JointPush
    if getattr(task_class, '_legacy_contact_installed', False):
        return
    evaluate = task_class._evaluate_ontop_context

    def evaluate_contact(self, env_ids=None):
        phi, diag = evaluate(self, env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        bodies = (self._rigid_body_pos[ids] if env_ids is None
                  else self._kinematic_humanoid_rigid_body_states[ids, ..., :3])
        state, _, progress, xy, z, distance = values(self, ids, bodies)
        owner = self.relation_runtime.graph.edge_owner[ids]
        mask = ((self.relation_runtime.graph.edge_relation[ids] == 12)
                & (self._members[ids] > 1).gather(1, owner))
        phi = torch.where(mask, state.gather(1, owner), phi)
        for key, value in (('progress', progress), ('distance', distance),
                           ('distance_xy', xy), ('z_error', z)):
            diag[key] = torch.where(mask, value.gather(1, owner), diag[key])
        return phi, diag

    def physics_contact(self):
        ids = torch.arange(self.num_envs, device=self.device)
        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self._refresh_sim_tensors()
            # Preserve the native KSH success condition for solo PUSH.
            distance, normal, _, _, _ = self._contact_geometry(ids, self._rigid_body_pos)
            reaction = -(self._contact_forces[:, :, self._key_body_ids[:2], :2]
                         * normal[:, :, None]).sum(-1)
            force = (self._assigned_box_values(self._ontop_box_contact)[..., :2]
                     * normal).sum(-1)
            cfg = self._joint['contact']
            native_hit = ((torch.exp(-cfg['near_distance_scale'] * distance.square())
                           > cfg['satisfaction_threshold'])
                          & (reaction >= cfg['min_hand_force']).all(-1)
                          & (force >= cfg['min_box_force']))
            legacy_hit = values(self, ids, self._rigid_body_pos)[1]
            self._push_hits |= torch.where(self._members > 1, legacy_hit, native_hit)

    task_class._evaluate_ontop_context = evaluate_contact
    task_class._physics_step = physics_contact
    task_class._legacy_contact_installed = True
    print('KSH_LEGACY_CONTACT', config, flush=True)
