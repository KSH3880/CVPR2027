import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tokenhsi'))
import isaacgym
import torch
from env.tasks.multi_agent import humanoid_ma_stage2 as stage2
from env.tasks.multi_agent import stage2_layout as layout
from learning.multi_agent import formal_eval
from utils import coupled_push_spec as coupled
from utils import stage2_eval_sizes
from utils.before_goal_sampling import ontop_distances


def prepare_new_bank(sizes, masses, env, preset):
    if preset != 'random_scenario':
        raise ValueError('This check requires random_scenario')
    count, people = sizes.shape[0], env['numAgents']
    groups = torch.arange(people).expand(count, -1).clone()
    parent = torch.full_like(groups, -1)
    tasks = torch.empty_like(groups)
    for row in range(count):
        order = torch.randperm(people).tolist()
        cursor = 0
        while cursor < people:
            remaining = people - cursor
            kind = (0 if os.environ.get('STAGE2_SCENE_MODE', 'mixed') == 'independent'
                    else int(torch.randint(3 if remaining >= 2 else 1, (1,))))
            width = int(torch.randint(2, min(4, remaining)+1, (1,))) if kind == 2 else (2 if kind == 1 else 1)
            selected = order[cursor:cursor+width]
            leader = selected[0]
            if kind == 0:
                tasks[row, leader] = torch.randint(1, 6, (1,))
            elif kind == 1:
                follower = selected[1]
                tasks[row, leader] = 3
                tasks[row, follower] = (1, 2, 4)[int(torch.randint(3, (1,)))]
                parent[row, follower] = leader
                groups[row, follower] = leader
            else:
                groups[row, selected] = leader
                tasks[row, selected] = 5
            cursor += width
    groups, parent, tasks = [value.to(sizes.device) for value in (groups, parent, tasks)]
    rows, followers = (parent >= 0).nonzero(as_tuple=True)
    leaders = parent[rows, followers]
    density = masses[rows, leaders] / sizes[rows, leaders].prod(-1)
    for axis, (lo, hi) in enumerate(((8, 12), (8, 12), (7, 10))):
        sizes[rows, leaders, axis] = torch.randint(lo, hi+1, (len(rows),), device=sizes.device)/20
    masses[rows, leaders] = sizes[rows, leaders].prod(-1)*density
    has_before = (parent >= 0).any(-1)
    has_coupled = ((coupled.group_sizes(groups) > 1) & (tasks == 5)).any(-1)
    domain = torch.where(has_before & has_coupled, 2, torch.where(has_coupled, 1, 0))
    return domain, groups, parent, tasks


stage2_eval_sizes.prepare_intersection_bank = prepare_new_bank
original_graph = stage2.HumanoidMAStage2._sample_episode_graph


def sample_graph(self, ids):
    original_graph(self, ids)
    self._before_near_pairs[ids] = False


stage2.HumanoidMAStage2._sample_episode_graph = sample_graph


def sample_layout(goal, leader_route, templates, near, margin=.001):
    approach, pickup = ontop_distances(torch.full_like(leader_route, 10.))
    angle = torch.rand_like(leader_route)*2*torch.pi
    direction = torch.stack((angle.cos(), angle.sin()), -1)
    payload = goal+approach[:, None]*direction
    minimum = 2.+margin
    cosine = ((minimum**2-approach.square()-pickup.square())/(2*approach*pickup)).clamp(-1., 1.)
    relative = (2*torch.rand_like(leader_route)-1)*cosine.acos()
    heading = angle+relative
    top_root = payload+pickup[:, None]*torch.stack((heading.cos(), heading.sin()), -1)
    distance = 2.+torch.rand_like(leader_route)*8.
    follower = goal+distance[:, None]*direction
    return torch.where((templates == 4)[:, None], top_root, follower), payload


layout.sample_layout = sample_layout
original_invalid = stage2.invalid_before_layout


def invalid_layout(task, ids):
    bad = original_invalid(task, ids)
    rows, env, follower, leader, source, goal = layout.pair_values(task, ids)
    distance = (task._humanoid_root_states[env, follower, :2]-task._tar_pos[env, goal, :2]).norm(dim=-1)
    bad[rows[distance < 2.]] = True
    return bad


stage2.invalid_before_layout = invalid_layout
original_eval = formal_eval.run_formal_eval


def capture(task):
    ids = torch.arange(task.num_envs, device=task.device)
    row, env, follower, leader, source, goal = layout.pair_values(task, ids)
    roots, boxes = task._humanoid_root_states, task._box_states
    target = task._tar_pos[env, goal, :2]
    payload = task._scenario_physical_object(env, follower)
    distances = torch.stack(((roots[env, leader, :2]-boxes[env, source, :2]).norm(dim=-1),
        (boxes[env, source, :2]-target).norm(dim=-1),
        (roots[env, follower, :2]-target).norm(dim=-1),
        (roots[env, follower, :2]-boxes[env, payload, :2]).norm(dim=-1),
        (boxes[env, payload, :2]-target).norm(dim=-1)), -1)
    top = task._templates[env, follower] == 4
    shared = task._box_size[env, source]
    eps = .002
    if ((distances[:, :2] < 1.-eps) | (distances[:, :2] > 5.+eps)).any():
        raise ValueError('Leader distance violation')
    if ((distances[:, 2] < 2.-eps) | (distances[:, 2] > 10.+eps)).any():
        raise ValueError('Follower goal distance violation')
    if ((distances[top, 3:] < 1.-eps) | (distances[top, 3:] > 5.+eps)).any():
        raise ValueError('ON_TOP distance violation')
    lo, hi = shared.new_tensor([.4,.4,.35]), shared.new_tensor([.6,.6,.5])
    if ((shared < lo-eps) | (shared > hi+eps)).any() or task._before_near_pairs.any():
        raise ValueError('Common size or near50 violation')
    data = dict(templates=task._templates.cpu(), predecessor=task._before_predecessor.cpu(),
                groups=task._groups.cpu(), shared_sizes=shared.cpu(), distances=distances.cpu(),
                followers=task._templates[env, follower].cpu(), near_pairs=task._before_near_pairs.cpu())
    destination = Path(task.cfg['args'].output_path)
    torch.save(data, destination / 'new_conditions.pt')
    (destination / 'new_conditions.json').write_text(json.dumps(dict(
        protocol='new_protocol_pilot', seed=task.cfg['args'].seed, environments=task.num_envs,
        enable_pairs=len(env), follower_template_counts={str(k):int((task._templates[env,follower]==k).sum()) for k in (1,2,4)},
        near50=False, common_size_bounds=[[.4,.6],[.4,.6],[.35,.5]],
        solo_skill_probabilities=[.2]*5, follower_probabilities=[1/3]*3,
        group_sampler='sequential uniform feasible I/E/C; C size uniform 2..4',
        distractor_physics='retained', conditions_verified=True), indent=2))


def checked_eval(player):
    reset = player.env_reset
    def reset_and_capture(*args, **kwargs):
        obs = reset(*args, **kwargs)
        capture(player.env.task)
        return obs
    player.env_reset = reset_and_capture
    return original_eval(player)


formal_eval.run_formal_eval = checked_eval
