"""PUSH-only extension of the imported Stage1 scene and native AT objective."""
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import torch
import yaml
from utils.door_reference import rotate
from utils.edge_scenario_spec import validate_sampler


def validate_env(env):
    native = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml').read_text())['env']
    if env['relationReward'] != native['relationReward']:
        raise ValueError('PUSH must preserve the imported Stage1 AT and reward contract')
    if (env['numAgents'], env['numObjects']) != (2, 4):
        raise ValueError('Stage1 PUSH requires two humans and four object inputs')
    validate_sampler(env['relationGraph'])
    if list(env['relationGraph']['template_probabilities'].values()) != [0., 0., 0., 1., 0.]:
        raise ValueError('PUSH experiment requires 100% contact + AT pairs')
    if env['skill'] != ['loco', 'push'] or env['templateRsi'] is not None:
        raise ValueError('PUSH requires its own loco/push reset path')
    surface = env['pushTask'].get('contact_objective') == 'common_face_xy_v1'
    midpoint = env['pushTask'].get('contact_objective') in ('midpoint_only_v1', 'common_face_xy_v1')
    rect = env['pushTask']['revision'] in (3, 4)
    if env['pushTask']['revision'] == 4:
        expected_box = dict(size_range_m=[1.2, 1.5], size_interval_m=.05, size_z_m=1.3,
                            mass=20. if midpoint else 25., friction=.6, restitution=0.)
        if midpoint:
            if 'size_y_m' in env['pushBox']:
                expected_box['size_y_m'] = 1.8
                if surface and (env['pushBox'].get('size_y_m'), env['pushBox'].get('mass')) == (2., 25.):
                    expected_box.update(size_y_m=2., mass=25.)
                if surface and (env['pushBox'].get('size_y_m'), env['pushBox'].get('mass'), env['pushBox'].get('friction')) == (2., 20., .4):
                    expected_box.update(size_y_m=2., mass=20., friction=.4)
            else:
                expected_box['tie_xy'] = True
    elif env['pushTask']['revision'] == 3:
        expected_box = dict(size_range_m=[1.2, 1.5], size_interval_m=.05,
                            mass=20., friction=.6, restitution=0.)
    else:
        expected_box = dict(size=[1.3]*3, mass=20., friction=.6, restitution=0.)
    if env['pushBox'] != expected_box:
        raise ValueError('PUSH physical box contract differs')
    if 'curriculum' in env['pushGoalRandomization']:
        from utils.push_goal_curriculum import validate_goal
        validate_goal(env['pushGoalRandomization'])
        if not midpoint or env['envSpacing'] < 32:
            raise ValueError('Goal curriculum requires midpoint PUSH and envSpacing >= 32')
    elif env['pushGoalRandomization'] != dict(sampling='box_forward_uniform', distance_range=[1., 4. if midpoint else 5.]):
        raise ValueError('PUSH goal distance/initial face-normal contract differs')
    if env['box']['reset']['randomHeight'] or env['box']['build']['randomSize']:
        raise ValueError('PUSH boxes must start grounded with fixed size')
    expected_reward = (dict(z_tolerance_m=.03, z_reward_sigma_m=.04) if rect
                       else dict(height_band_m=[.585, .650], z_reward_sigma_m=.04))
    if 'contact_objective' in env['pushTask'] and not midpoint:
        raise ValueError('Unknown PUSH contact objective')
    if midpoint:
        if env['pushTask']['revision'] != 4 or 'hand_distance_m' in env['pushTask']:
            raise ValueError('Midpoint-only CONTACT requires native RSI without a face-distance gate')
        expected_reward = dict(z_reward_sigma_m=.04)
        if 'xy_reward_scale' in env['pushPointReward']:
            expected_reward = {key: env['pushPointReward'].get(key)
                               for key in ('xy_reward_scale', 'z_reward_sigma_m')}
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value <= 0 for value in expected_reward.values()):
                raise ValueError('Midpoint state decay parameters must be finite positive numbers')
    if surface:
        expected_reward = {}
    if env['pushPointReward'] != expected_reward:
        raise ValueError('PUSH contact height reward differs')
    spec = env['pushTask']
    if 'rsi_max_hand_midpoint_height_m' in spec:
        limit=spec['rsi_max_hand_midpoint_height_m']
        if (not midpoint or spec.get('rsi_sampling')!='native_size_cache' or spec.get('rsi_retarget')!='none'
                or isinstance(limit,bool) or not isinstance(limit,(int,float)) or not math.isfinite(limit)
                or not 0 < limit <= env['pushBox']['size_z_m']):
            raise ValueError('PUSH RSI height selection requires native unchanged poses and a finite positive limit')
    if 'at_success_max_tilt_deg' in spec:
        limit = spec['at_success_max_tilt_deg']
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or not 0 <= limit <= 180:
            raise ValueError('PUSH AT success tilt must be finite degrees in [0, 180]')
    if 'reset_geometry_filter' in spec:
        if (spec['reset_geometry_filter']!='collider_bounds_v1' or spec['reset_geometry_margin_m']!=.02
                or spec['reset_floor_clearance_m']!=.002
                or len(spec['reset_geometry_sha256'])!=64):
            raise ValueError('Unknown PUSH collider reset contract')
    if midpoint and 'size_y_m' in env['pushBox'] and spec['loco_distance_range_m'] != [1., 5.]:
        raise ValueError('Depth180 PUSH uses the configured center-distance range 1--5m')
    if spec['objective'] != 'push_contact_plus_native_at' or spec['probability'] != 1.:
        raise ValueError('Unknown PUSH objective')
    if spec['revision'] == 1:
        if spec.get('require_contact_force', True) is not True:
            raise ValueError('Original PUSH revision requires contact force')
    elif spec['revision'] in (2, 3, 4):
        if midpoint and spec.get('require_contact_force') is True:
            if any(isinstance(spec.get(key), bool) or not isinstance(spec.get(key), (int, float))
                   or not math.isfinite(spec[key]) or spec[key] <= 0
                   for key in ('min_hand_force_n', 'min_box_force_n')):
                raise ValueError('Midpoint CONTACT force thresholds must be finite positive Newtons')
        elif spec.get('require_contact_force') is not False or any(key in spec for key in ('min_hand_force_n', 'min_box_force_n')):
            raise ValueError('No-force PUSH revision uses geometry only')
        expected_retarget = 'none' if spec['revision'] == 4 else 'preserve_hand_height_difference'
        relaxed = midpoint and 'xy_reward_scale' in env['pushPointReward']
        mode = spec.get('height_mode', 'fixed')
        if mode not in ('fixed', 'amp_reference') or (mode == 'amp_reference' and not relaxed):
            raise ValueError('Unknown midpoint height mode')
        height = spec.get('height_m')
        if mode == 'amp_reference' and 'height_m' in spec:
            raise ValueError('AMP midpoint height must come from the reference, not a fixed height')
        if relaxed and mode == 'fixed':
            if (isinstance(height, bool) or not isinstance(height, (int, float))
                    or not math.isfinite(height) or not 0 < height < env['pushBox']['size_z_m']):
                raise ValueError('Midpoint height must be inside the box face')
        if rect and ((not surface and not relaxed and height != .65) or spec.get('rsi_retarget') != expected_retarget):
            raise ValueError('Rectangular PUSH hand midpoint/retarget contract differs')
        if spec['revision'] == 4:
            if (spec.get('rsi_sampling') != 'native_size_cache' or spec.get('rsi_box_progress_scale') != .85
                    or any(k in spec for k in ('bank_file', 'bank_sha256'))):
                raise ValueError('AMP RSI samples the original motion/time without a candidate bank')
            if (spec['late_probability'] != env['sizeAwareRsi']['lateProbability']
                    or spec['late_fraction'] != env['sizeAwareRsi']['lateFraction']
                    or spec['late_fraction'] != .3):
                raise ValueError('PUSH must use native SIT/CARRY late-frame sampling settings')
    else:
        raise ValueError('Unknown PUSH objective revision')
    for key in ('interaction_probability', 'late_probability', 'late_fraction'):
        if not 0 < spec[key] <= 1:
            raise ValueError('Invalid PUSH RSI probability')
    if spec['minimum_rsi_remaining_m'] <= env['relationReward']['at']['success_distance']:
        raise ValueError('RSI must leave more than the AT success distance')
    if env['envSpacing'] < 16 or spec['station_separation_m'] != 12.:
        raise ValueError('PUSH swept paths need separated workspaces')


def sample_box_sizes(count, device, env):
    spec = env['pushBox']
    if 'size' in spec:
        return torch.tensor(spec['size'], device=device).expand(count, 4, 3).clone()
    lo, hi = spec['size_range_m']; step = spec['size_interval_m']
    levels = round((hi-lo)/step)+1
    sizes = lo+step*torch.randint(levels, (count, 4, 3), device=device)
    if spec.get('tie_xy', False):
        sizes[..., 1] = sizes[..., 0]
    if 'size_y_m' in spec:
        sizes[..., 1] = float(spec['size_y_m'])
    if 'size_z_m' in spec:
        sizes[..., 2] = float(spec['size_z_m'])
    return sizes


def box_size_profiles(env):
    """Enumerate exactly the physical size distribution used by the sampler."""
    spec = env['pushBox']
    if 'size' in spec:
        return [tuple(spec['size'])]
    lo, hi = spec['size_range_m']; step = spec['size_interval_m']
    levels = [lo+step*i for i in range(round((hi-lo)/step)+1)]
    z_levels = [spec['size_z_m']] if 'size_z_m' in spec else levels
    y_levels = [spec['size_y_m']] if 'size_y_m' in spec else levels
    xy = [(value, value) for value in levels] if spec.get('tie_xy', False) else itertools.product(levels, y_levels)
    return [(x, y, z) for x, y in xy for z in z_levels]


def hand_box_overlap(hands, boxes, sizes, radius=.04):
    """Exact signed-distance overlap of MJCF hand spheres with oriented boxes.

    Prefix is rows; hands is rows x 2 x 3. Returns sphere overlap depth and
    whether each hand center lies strictly inside the physical box.
    """
    inv = boxes[:, 3:7].clone(); inv[:, :3] *= -1
    local = rotate(inv[:, None].expand(-1, hands.shape[1], -1), hands-boxes[:, None, :3])
    q = local.abs()-sizes[:, None]/2
    signed = q.clamp_min(0).norm(dim=-1)+q.amax(-1).clamp_max(0)
    return (radius-signed).clamp_min(0), q.amax(-1) < 0


def box_tilt_degrees(quaternion):
    q = quaternion/quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    return torch.atan2(torch.hypot(2*(x*z+w*y), 2*(y*z-w*x)), 1-2*(x*x+y*y))*180/math.pi


def sample_boxes_and_goals(count, device, origins, env, sizes=None, epoch=0):
    """Four real objects, two independent workspaces, no hidden/padded tokens."""
    states = torch.zeros(count, 4, 13, device=device)
    centers = states.new_tensor([[-8., -8.], [8., 8.], [-8., 8.], [8., -8.]])
    if 'curriculum' in env['pushGoalRandomization']:
        centers *= 2  # Keep both 10m goal corridors away from other real objects.
    states[..., :2] = centers + (torch.rand(count, 4, 2, device=device)-.5)
    states[..., :2] += origins[:, None, :2]
    if sizes is None:
        if 'size' not in env['pushBox']:
            raise ValueError('Rectangular reset requires the actual physical actor sizes')
        sizes = states.new_tensor(env['pushBox']['size']).expand(count, 4, 3)
    states[..., 2] = sizes[..., 2]/2
    yaw = torch.rand(count, 4, device=device)*2*math.pi
    states[..., 5] = (yaw/2).sin(); states[..., 6] = (yaw/2).cos()
    from utils.push_goal_curriculum import distance_range
    lo, hi = distance_range(env, epoch)
    distances = lo+(hi-lo)*torch.rand(count, 2, device=device)
    forward = rotate(states[:, :2, 3:7], states.new_tensor([0., 1., 0.]).expand(count, 2, 3))
    goals = states[:, :2, :3]+forward*distances[..., None]
    return states, goals, forward, distances


def sample_rsi_rows(bank, library, count, spec):
    rows = sorted(bank['candidates'], key=lambda row: (row['motion_id'], row['time']))
    device = library.gts.device
    weights = torch.zeros(len(rows), device=device); late = torch.zeros_like(weights)
    for mid in range(library.num_motions()):
        ids = [i for i, row in enumerate(rows) if row['motion_id'] == mid]
        if not ids: continue
        weights[ids] = float(library._motion_weights[mid])/len(ids)
        tail = ids[int(len(ids)*(1-spec['late_fraction'])):]
        late[tail] = float(library._motion_weights[mid])/len(tail)
    chosen = torch.multinomial(weights, count, replacement=True)
    requested = torch.rand(count, device=device) < spec['late_probability']
    if requested.any(): chosen[requested] = torch.multinomial(late, int(requested.sum()), replacement=True)
    result = {key: torch.tensor([row[key] for row in rows], device=device)[chosen]
              for key in ('motion_id', 'time', 'phase', 'box_progress')}
    result['late_requested'] = requested
    return result


def check_checkpoint(weights, env):
    saved = weights.get('relation_experiment_config', {}).get('env', {})
    if 'pushTask' not in saved and 'pushTask' not in env: return
    keys = ('pushTask', 'pushBox', 'pushGoalRandomization', 'pushPointReward', 'relationReward',
            'relationGraph', 'numAgents', 'numObjects', 'skill', 'skillDiscProb')
    if any(saved.get(key) != env.get(key) for key in keys):
        raise ValueError('PUSH checkpoint objective/config differs; original HOLDING is separate')


def load_bank(env):
    manifest = Path('tokenhsi/data/dataset_stage1_push.yaml')
    motions = yaml.safe_load(manifest.read_text())['motions']['push']
    hashes = [hashlib.sha256((manifest.parent/row['file']).read_bytes()).hexdigest() for row in motions]
    if env['pushTask']['revision'] == 4:
        if len(hashes) != 21 or hashes != env['pushTask']['motion_hashes']:
            raise ValueError('AMP RSI original motion files differ')
        return None
    bank = json.loads(Path(env['pushTask']['bank_file']).read_text())
    if hashes != bank['motion_hashes'] or not bank['candidates']:
        raise ValueError('PUSH screened poses/motion data differ')
    if env['pushTask']['revision'] == 3:
        digest = hashlib.sha256(Path(env['pushTask']['bank_file']).read_bytes()).hexdigest()
        if (bank.get('push_revision') != 3 or bank.get('height_m') != .65
                or bank.get('screen_status') != 'passed_all_rectangular_sizes'
                or digest != env['pushTask'].get('bank_sha256')):
            raise ValueError('Rectangular PUSH requires its separately screened RSI bank')
    elif len(bank['candidates']) != 11:
        raise ValueError('Original PUSH screened candidate set differs')
    return bank
