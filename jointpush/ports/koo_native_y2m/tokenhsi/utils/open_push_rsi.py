"""Task-matched, PD-screened human references with explicit object proxies."""
import hashlib
import json
import math
from pathlib import Path
import torch
from utils.door_reference import yaw_from_quat, rotate, multiply


def validate_rsi(spec):
    if set(spec) - {'pose_mode','excluded_motion_ids','max_support_sole_height_m'} != {'bank_file', 'interaction_probability', 'late_probability', 'late_fraction', 'stand_off'}:
        raise ValueError('Invalid OPEN/PUSH RSI fields')
    if spec.get('pose_mode', 'legacy') not in ('legacy', 'native_phase'):
        raise ValueError('Invalid RSI pose mode')
    if 'max_support_sole_height_m' in spec and not 0 < spec['max_support_sole_height_m'] <= .02:
        raise ValueError('Invalid initial foot sole support tolerance')
    if any(type(v) is not int or v < 0 for v in spec.get('excluded_motion_ids', [])):
        raise ValueError('Invalid excluded RSI motion IDs')
    if not isinstance(spec['bank_file'], str) or not spec['bank_file']:
        raise ValueError('RSI requires a screened candidate bank')
    for key in ('interaction_probability', 'late_probability', 'late_fraction'):
        v = spec[key]
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 < v <= 1:
            raise ValueError('Invalid RSI probability/fraction: ' + key)
    if isinstance(spec['stand_off'], bool) or not isinstance(spec['stand_off'], (float, int)) or not 0 < spec['stand_off'] < .12:
        raise ValueError('Invalid RSI hand stand-off')


def bank_contract(env):
    """Bind the PD screen to reference files and physical/reward settings."""
    manifest = Path(env['motion_file'])
    import yaml
    motions = yaml.safe_load(manifest.read_text())['motions']
    hashes = {skill: [hashlib.sha256((manifest.parent / row['file']).read_bytes()).hexdigest()
                       for row in motions[skill]] for skill in ('doorOpen', 'push')}
    contract=dict(revision=3, push_arm_retarget='shoulder_rotation_v1', motion_hashes=hashes, door=env['door'], hold=env['doorHold'],
                push_box=env['pushBox'], reward=env['relationReward'], stand_off=env['openPushRSI']['stand_off'])
    if env['relationReward'].get('push_at',{}).get('hand_reference')=='goal_opposite_surface_xy':
        contract['push_rsi_alignment']='goal_opposite_surface_xy_v1'
    if env['relationReward'].get('contact_at'):
        contract['push_rsi_alignment']='goal_contact_face_xy_v1'
    return contract


def geometry_candidates(libraries, hands, feet, history):
    """Propose grounded upright frames; actual stability is checked in PhysX."""
    result = {}
    for skill in ('doorOpen', 'push'):
        lib = libraries[skill]; rows = []
        for mid in range(lib.num_motions()):
            start = int(lib.length_starts[mid]); n = int(lib._motion_num_frames[mid])
            pos = lib.gts[start:start+n]; rot = lib.grs[start:start+n, 0]
            yaw = yaw_from_quat(rot)
            forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), -1)
            reach = ((pos[:, hands, :2]-pos[:, :1, :2])*forward[:, None]).sum(-1)
            height = pos[:, hands, 2]
            hand_ok = (reach > .2) & ((height-1.).abs() < .18) if skill == 'doorOpen' else (reach > .2) & (height > .5) & (height < 1.18)
            upright = (1-2*(rot[:, 0].square()+rot[:, 1].square())) > math.cos(.7 if skill=='doorOpen' else 1.)
            valid = (pos[:, 0, 2] > .7) & (pos[:, feet, 2].amin(-1) < (.18 if skill=='doorOpen' else .3)) & upright & hand_ok.any(-1)
            dt = float(lib._motion_dt[mid]); window = math.ceil(history/dt)+1
            eligible = torch.zeros_like(valid)
            if n >= window:
                eligible[window-1:] = valid.unfold(0, window, 1).all(-1)
            frames = eligible.nonzero().flatten()[::4]
            for frame in frames.cpu().tolist():
                phase = frame/max(1, n-1)
                rows.append(dict(motion_id=mid, time=frame*dt, phase=phase,
                    angle=.4+1.2*phase if skill == 'doorOpen' else 0.,
                    box_progress=0.85*phase if skill == 'push' else 0.))
        if not rows:
            raise ValueError('No geometry RSI proposals: '+skill)
        result[skill] = rows
    return result


class OpenPushRSIBank:
    def __init__(self, spec, libraries, contract, proposals=None):
        validate_rsi(spec); self.spec = spec; self.rows = {}; self.pools = {}
        if proposals is None:
            data = json.loads(Path(spec['bank_file']).read_text())
            if data['contract'] != contract or data.get('screen') != 'PD_12_steps':
                raise ValueError('OPEN/PUSH RSI bank screen/contract mismatch')
            proposals = data['candidates']
        for skill in ('doorOpen', 'push'):
            rows = sorted(proposals[skill], key=lambda r: (r['motion_id'], r['time']))
            lib = libraries[skill]; device = lib.gts.device
            weights = torch.zeros(len(rows), device=device); late = torch.zeros_like(weights)
            for mid in range(lib.num_motions()):
                indices = [i for i, r in enumerate(rows) if r['motion_id'] == mid]
                if not indices:
                    raise ValueError('No PD-valid RSI candidates for '+skill+' clip '+str(mid))
                for i in indices:
                    r=rows[i]
                    if not all(math.isfinite(r[k]) for k in ('time','phase','angle','box_progress')) or not 0 <= r['time'] <= float(lib._motion_lengths[mid]) or not 0 <= r['phase'] <= 1 or not 0 <= r['angle'] <= 1.6 or not 0 <= r['box_progress'] <= .85:
                        raise ValueError('Invalid RSI candidate')
                weight = float(lib._motion_weights[mid])
                weights[indices] = weight / len(indices)
                tail = indices[int(len(indices)*(1-spec['late_fraction'])):]
                late[tail] = weight / len(tail)
            self.rows[skill] = rows
            self.pools[skill] = (weights, late)

    def sample(self, skill, count):
        weights, late = self.pools[skill]
        chosen = torch.multinomial(weights, count, replacement=True)
        requested = torch.rand(count, device=weights.device) < self.spec['late_probability']
        if requested.any():
            chosen[requested] = torch.multinomial(late, int(requested.sum()), replacement=True)
        rows = self.rows[skill]
        values = {key: torch.tensor([r[key] for r in rows], device=weights.device)[chosen]
                  for key in ('motion_id', 'time', 'phase', 'angle', 'box_progress')}
        values['late_requested'] = requested
        return values


def align_pose(body, body_rot, body_vel, body_ang, root_rot, target, angle, hands, stand_off):
    """Face the target surface normal; align a forward-reaching hand to it."""
    yaw = math.pi/2 + angle-yaw_from_quat(root_rot)
    q = torch.zeros_like(root_rot); q[:, 2] = torch.sin(yaw/2); q[:, 3] = torch.cos(yaw/2)
    expanded = q[:, None].expand_as(body_rot)
    relative = rotate(expanded, body-body[:, :1])
    normal = torch.stack((-torch.sin(angle), torch.cos(angle), torch.zeros_like(angle)), -1)
    reaches = (relative[:, hands]*normal[:, None]).sum(-1)
    score = (body[:, hands, 2]-target[:, None, 2]).abs().masked_fill(reaches < .2, float('inf'))
    if not torch.isfinite(score.amin(-1)).all():
        raise ValueError('No forward-reaching RSI hand')
    hand = torch.tensor(hands, device=body.device)[score.argmin(-1)]
    wanted = target-normal*stand_off
    root = body[:, 0].clone()
    root[:, :2] = wanted[:, :2]-relative[torch.arange(len(body), device=body.device), hand, :2]
    return (relative+root[:, None], multiply(expanded, body_rot),
            rotate(expanded, body_vel), rotate(expanded, body_ang), q)


def align_push_pose(body, body_rot, body_vel, body_ang, targets, hands, stand_off):
    """Fit both hand XY points with one rigid yaw/translation; retain foot Z."""
    delta=body[:,hands[0],:2]-body[:,hands[1],:2]
    wanted_delta=targets[:,0,:2]-targets[:,1,:2]
    yaw=torch.atan2(wanted_delta[:,1],wanted_delta[:,0])-torch.atan2(delta[:,1],delta[:,0])
    q=body_rot[:,0].new_zeros(len(body),4);q[:,2]=torch.sin(yaw/2);q[:,3]=torch.cos(yaw/2)
    expanded=q[:,None].expand_as(body_rot)
    relative=rotate(expanded,body-body[:,:1])
    right=wanted_delta/wanted_delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    normal=torch.stack((-right[:,1],right[:,0]),-1)
    root=body[:,0].clone()
    root[:,:2]=targets[:,:,:2].mean(1)-relative[:,hands,:2].mean(1)-stand_off*normal
    return (relative+root[:,None],multiply(expanded,body_rot),
            rotate(expanded,body_vel),rotate(expanded,body_ang),q)


def lower_push_arms(body, rotations, velocities, angular, dof, dof_vel,
                    parents, hands, dof_bodies, offsets, height):
    """Retarget reset shoulders only; preserve torso, feet and arm lengths.

    AMP experts remain original. Reset shoulder velocity is zero for the added
    pose offset; descendants retain relative motion in the new orientation.
    """
    from utils import torch_utils
    body, rotations, velocities, angular, dof, dof_vel = (
        x.clone() for x in (body, rotations, velocities, angular, dof, dof_vel))
    for hand in hands:
        shoulder = parents[parents[hand]]
        children = []
        for node in range(len(parents)):
            ancestor = node
            while ancestor >= 0 and ancestor != shoulder:
                ancestor = parents[ancestor]
            if ancestor == shoulder:
                children.append(node)
        pivot = body[:, shoulder].clone()
        vector = body[:, hand]-pivot
        length = vector.norm(dim=-1).clamp_min(1e-6)
        dz = torch.clamp(body.new_full(length.shape, height)-pivot[:, 2],
                         min=-length+.01, max=length-.01)
        horizontal = vector[:, :2]/vector[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        wanted = torch.cat((horizontal*(length.square()-dz.square()).clamp_min(0).sqrt()[:, None], dz[:, None]), -1)
        delta = torch.cat((torch.cross(vector, wanted, dim=-1),
                           (length.square()+(vector*wanted).sum(-1))[:, None]), -1)
        delta = delta/delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        expanded = delta[:, None].expand(-1, len(children), -1)
        relative=body[:,children]-pivot[:,None]
        new_relative=rotate(expanded,relative)
        old_angular=angular[:,shoulder,None].clone()
        parent_angular=angular[:,parents[shoulder],None].clone()
        relative_velocity=velocities[:,children]-velocities[:,shoulder,None]-torch.cross(
            old_angular.expand_as(relative),relative,dim=-1)
        velocities[:, children] = velocities[:, shoulder, None]+torch.cross(
            parent_angular.expand_as(new_relative),new_relative,dim=-1)+rotate(expanded,relative_velocity)
        angular[:, children] = parent_angular+rotate(expanded,angular[:,children]-old_angular)
        body[:, children] = pivot[:, None]+new_relative
        rotations[:, children] = multiply(expanded, rotations[:, children])
        parent_rot = rotations[:, parents[shoulder]].clone(); parent_rot[:, :3] *= -1
        local = multiply(parent_rot, rotations[:, shoulder])
        index = dof_bodies.index(shoulder); lo, hi = offsets[index:index+2]
        if hi-lo != 3:
            raise ValueError('PUSH shoulder retarget requires a spherical joint')
        dof[:, lo:hi] = torch_utils.quat_to_exp_map(local)
        dof_vel[:, lo:hi] = 0.
    return body, rotations, velocities, angular, dof, dof_vel
