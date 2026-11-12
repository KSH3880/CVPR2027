import ast
import hashlib
from pathlib import Path

import torch
import yaml
from isaacgym import torch_utils as gym_torch
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from utils import torch_utils
from utils.before_object_alias import encoder_packet, goal_aligned_scene


ROOT = Path(__file__).resolve().parents[1]
LIMITATION = ('TokenHSI retains its original fixed 397D self/assigned-task input. '
              'Other people and distractor object tokens do not enter its encoder; '
              'the common all-object-token attention requirement is unsupported.')


def native_function(filename, name):
    path = ROOT/'tokenhsi/env/tasks/basic_interaction_skills'/filename
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(gym_torch.__dict__, torch=torch, torch_utils=torch_utils)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


carry_obs = native_function('humanoid_carry.py', 'compute_location_observations')
sit_obs = native_function('humanoid_sit.py', 'compute_location_observations')
climb_obs = native_function('humanoid_climb.py', 'compute_location_observations')


def project_scene(observation, people, objects):
    scene = observation[:, :-people]
    packet = encoder_packet(scene[:, -5*people:], people, objects).reshape(-1, people, 5).long()
    scene = goal_aligned_scene(scene, people, objects)
    count = len(scene)
    human = scene[:, :223*people].reshape(count, people, 223)
    intrinsic = scene[:, 223*people:223*people+30*objects].reshape(count, objects, 30)
    start = 223*people+30*objects+people
    poses = scene[:, start:start+7*(2*people+objects)].reshape(count, -1, 7)
    human = human.clone()
    human[..., 0] = 0
    heading = poses[:, :people, 3:7].reshape(-1, 4)
    for offset in (43, 46):
        human[..., offset:offset+3] = gym_torch.quat_rotate(heading, human[..., offset:offset+3].reshape(-1, 3)).reshape(count, people, 3)
    batch = torch.arange(count, device=scene.device)[:, None].expand(-1, people)
    valid, kind, actor, payload, target = packet.unbind(-1)
    if not valid.bool().all() or not torch.equal(actor.sort(-1).values, torch.arange(people, device=scene.device).expand_as(actor)):
        raise ValueError('Expected exactly one active task per person')
    root = scene.new_zeros(count, people, 13)
    root[..., :7] = poses[batch, actor]
    selected = torch.where(kind >= 2, payload, target)-people
    box = scene.new_zeros(count, people, 13)
    box[..., :7] = poses[batch, people+selected]
    nodes = intrinsic[batch, selected]
    box[..., 7:10] = gym_torch.quat_rotate(box[..., 3:7].reshape(-1, 4), nodes[..., :3].reshape(-1, 3)).reshape(count, people, 3)
    box[..., 10:13] = gym_torch.quat_rotate(box[..., 3:7].reshape(-1, 4), nodes[..., 3:6].reshape(-1, 3)).reshape(count, people, 3)
    bps = nodes[..., 6:30].reshape(count, people, 8, 3)
    size = bps.amax(-2)-bps.amin(-2)
    support = poses[batch, target.clamp(people, people+objects-1)]
    support_bps = intrinsic[batch, (target-people).clamp(0, objects-1), 6:30].reshape(count, people, 8, 3)
    support_size = support_bps.amax(-2)-support_bps.amin(-2)
    from env.tasks.multi_agent.edge_ontop_reward import vertical_extent
    goal = support[..., :3].clone()
    goal[..., 2] += vertical_extent(support[..., 3:7], support_size/2)
    goal[..., 2] += torch.where(kind == 0, .10, torch.where(kind == 1, .94, size[..., 2]/2))
    point_task = (kind == 2) | (kind == 4)
    goal = torch.where(point_task[..., None], poses[batch, target.clamp(0, poses.shape[1]-1), :3], goal)
    result = scene.new_zeros(count, people, 397)
    result[batch, actor, :223] = human[batch, actor]
    for task_id, start, width in ((0, 243, 38), (1, 323, 27), (2, 281, 42), (3, 281, 42), (4, 350, 42)):
        rows, records = (kind == task_id).nonzero(as_tuple=True)
        if not len(rows):
            continue
        r, b, corners, g = [v[rows, records] for v in (root, box, bps, goal)]
        if task_id == 0:
            facing = torch.zeros_like(g)
            facing[:, 0] = 1
            value = sit_obs(r, g, b, corners, facing)
        elif task_id == 1:
            value = climb_obs(r, b, corners, g)
        else:
            value = carry_obs(r, b, corners, g, True)
        owners = actor[rows, records]
        result[rows, owners, start:start+width] = value
        label = (1, 3, 2, 2, 4)[task_id]
        result[rows, owners, 392+label] = 1
    return result.flatten(0, 1)


class TokenHSIPolicy:
    def __init__(self, checkpoint, device):
        from tokenhsi.learning.transformer.amp_network_builder_transformer import AMPTransformerMultiTaskBuilder
        self.path = Path(checkpoint).resolve(strict=True)
        self.saved = torch.load(self.path, map_location='cpu', weights_only=False)
        builder = AMPTransformerMultiTaskBuilder()
        builder.load(yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/push_kd.yaml').read_text())['params']['network'])
        info = dict(onehot_size=5, tota_subtask_obs_size=169,
                    each_subtask_obs_size=[20, 38, 42, 27, 42],
                    each_subtask_obs_indx=[0, 20, 58, 100, 127, 169],
                    each_subtask_name=['traj', 'sit', 'carry', 'climb', 'push'], enable_task_mask_obs=True)
        self.model = torch.nn.Module()
        self.model.a2c_network = builder.build('amp', actions_num=32, input_shape=(397,),
            amp_input_shape=(1340,), self_obs_size=223, task_obs_size=174,
            multi_task_info=info, device=device)
        self.model.load_state_dict(self.saved['model'], strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.rms = RunningMeanStd((397,)).to(device)
        self.rms.load_state_dict(self.saved['running_mean_std'], strict=True)
        self.rms.eval().requires_grad_(False)
        self.calls = 0

    def normalize(self, raw):
        normalized = self.rms(raw)
        normalized[..., -5:] = raw[..., -5:]
        return normalized

    @torch.no_grad()
    def action(self, raw):
        mean, logstd = self.model.a2c_network.eval_actor(self.normalize(raw), raw)
        if not torch.isfinite(mean).all():
            raise ValueError('Nonfinite TokenHSI action')
        self.calls += 1
        return mean.clamp(-1, 1)

    def finish(self):
        for live, saved in ((self.model.state_dict(), self.saved['model']),
                            (self.rms.state_dict(), self.saved['running_mean_std'])):
            if any(not torch.equal(value.cpu(), saved[key].cpu()) for key, value in live.items()):
                raise ValueError('TokenHSI weights or normalization changed')
        return dict(checkpoint=str(self.path), checkpoint_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                    epoch=self.saved.get('epoch'), action_calls=self.calls, deterministic=True,
                    weights_unchanged=True, normalization_unchanged=True, other_policy_actions=0,
                    input_conversion='Assigned task; goal-aligned enable support; individual PUSH region',
                    protocol_limitations=[LIMITATION])
