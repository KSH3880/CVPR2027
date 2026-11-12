import numpy as np
import torch

from utils.evaluation_protocol import spatial_success, shared_center_success


THRESHOLDS = (.06, .08, .12)


def surface_distance(hands, boxes, sizes):
    delta = hands - boxes[:, None, None, :3]
    xyz = -boxes[:, None, None, 3:6]
    uv = torch.cross(xyz.expand_as(delta), delta, dim=-1)
    local = delta + 2*(boxes[:, None, None, 6:7]*uv + torch.cross(xyz.expand_as(delta), uv, dim=-1))
    excess = local.abs() - sizes[:, None, None]/2
    signed = excess.clamp_min(0).norm(dim=-1) + excess.amax(-1).clamp_max(0)
    return signed.abs()


class Table2Metrics:
    def __init__(self, initial, setting, device, thresholds=THRESHOLDS, start_displacement_m=.2, start_hand_distance_m=None):
        self.setting = setting
        self.thresholds = tuple(thresholds)
        self.start_displacement_m = start_displacement_m
        self.start_hand_distance_m = start_hand_distance_m
        self.packet = initial['packet'].to(device)
        self.groups = initial['groups'].to(device)
        self.sizes = initial['sizes'].to(device)
        self.goals = initial['goals'].to(device)
        self.center_goals = initial['shared_center_goals'].to(device)
        self.origin = initial['shared_centers'][:, 0].to(device)
        n = len(self.packet)
        self.first_success = torch.full((n,), -1, dtype=torch.long, device=device)
        self.start = torch.full_like(self.first_success, -1)
        self.frames = torch.zeros_like(self.first_success)
        self.contacts = torch.zeros(n, len(self.thresholds), dtype=torch.long, device=device)
        self.step = 0

    def update(self, roots, boxes, alive, physical=None, physical_sizes=None, hands=None):
        self.step += 1
        if self.setting.startswith('enable_'):
            current = spatial_success(self.packet, roots, boxes, self.sizes, self.goals)[..., 1]
            success = (current & alive).all(-1)
        else:
            centers = physical[:, None, :3].expand(-1, alive.shape[1], -1)
            success = shared_center_success(centers, self.center_goals, self.groups, alive).any(-1)
            distance = surface_distance(hands.double(), physical.double(), physical_sizes.double())
            moved = (physical[:, :2].double()-self.origin[:, :2].double()).norm(dim=-1) >= self.start_displacement_m
            if self.start_hand_distance_m is not None:
                nearby = ((distance < self.start_hand_distance_m) & alive[..., None]).any(-1).any(-1)
                moved &= nearby
            self.start[(self.start < 0) & moved] = self.step
        self.first_success[(self.first_success < 0) & success] = self.step
        if self.setting.startswith('coupled_'):
            window = (self.start >= 0) & ((self.first_success < 0) | (self.first_success == self.step))
            for column, threshold in enumerate(self.thresholds):
                contact = ((distance < threshold) & alive[..., None]).all(-1).all(-1)
                self.contacts[:, column] += window & contact
            self.frames += window

    def arrays(self):
        return {key: value.cpu().numpy() for key, value in dict(first_success=self.first_success,
            start=self.start, window_frames=self.frames, contact_frames=self.contacts).items()}

    def summary(self, dt):
        successes = self.first_success >= 0
        times = self.first_success[successes].double()*dt
        result = dict(setting=self.setting, trials=len(successes), successes=int(successes.sum()),
            group_success_rate=float(successes.double().mean()),
            completion_time_s=float(times.mean()) if len(times) else None,
            xy_extent_percent=100, control_dt=dt, enabling_success='simultaneous')
        if self.setting.startswith('coupled_'):
            ratio = self.contacts.double()/self.frames[:, None].clamp_min(1)
            result['cooperative_time_ratio_percent'] = {format(t*100,'.12g'):float(ratio[:, i].mean()*100)
                                                       for i, t in enumerate(self.thresholds)}
            result.update(distance_reference='hand center to physical box surface; 3D',
                contact_force_used=False, start_displacement_m=self.start_displacement_m,
                window=f'first displacement >={self.start_displacement_m:g}m through first success, inclusive; failure through recorded episode end',
                failed_trials_included=True, no_start_percent=0., original_participants_retained=True)
            if self.start_hand_distance_m is not None:
                result['start_hand_distance_m'] = self.start_hand_distance_m
                result['window'] = (f'first displacement >={self.start_displacement_m:g}m and any valid hand surface distance '
                                    f'<{self.start_hand_distance_m:g}m in same frame through first success, inclusive; failure through recorded episode end')
        return result


def replay(path, thresholds=THRESHOLDS, start_displacement_m=.2, start_hand_distance_m=None):
    import json
    torch.set_num_threads(1)
    initial = torch.load(path/'initial.pt', map_location='cpu', weights_only=False)
    recorded = json.loads((path/'table2.json').read_text())
    metric = Table2Metrics(initial, recorded['setting'], 'cpu', thresholds, start_displacement_m, start_hand_distance_m)
    coupled = recorded['setting'].startswith('coupled_')
    with np.load(path/'trace.npz') as trace:
        raw = {key:trace[key] for key in ('roots', 'boxes', 'valid')}
    if coupled:
        with np.load(path/'cooperation.npz') as archive:
            cooperation = {key:torch.from_numpy(archive[key]) for key in ('hands', 'physical_boxes', 'physical_sizes')}
    for frame in range(len(raw['roots'])):
        options = dict(physical=cooperation['physical_boxes'][frame], physical_sizes=cooperation['physical_sizes'],
                       hands=cooperation['hands'][frame]) if coupled else {}
        metric.update(torch.from_numpy(raw['roots'][frame]), torch.from_numpy(raw['boxes'][frame]),
                      torch.from_numpy(raw['valid'][frame]), **options)
    computed = metric.summary(recorded['control_dt'])
    with np.load(path/'table2_trials.npz') as archive:
        for name, actual in metric.arrays().items():
            changed_start = (start_displacement_m != recorded.get('start_displacement_m', .2)
                             or start_hand_distance_m != recorded.get('start_hand_distance_m'))
            if name != 'first_success' and changed_start:
                continue
            if name == 'contact_frames' and tuple(thresholds) != THRESHOLDS:
                continue
            if not np.array_equal(actual, archive[name]):
                raise ValueError('CPU/GPU Table2 mismatch: '+name)
    if computed['group_success_rate'] != recorded['group_success_rate']:
        raise ValueError('CPU/GPU Table2 success rate mismatch')
    return computed
