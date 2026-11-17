"""Search Bridge rollouts for qualitative scenes: three pushers, a crosser that carries a box across
(AT), and a follower on the destination platform that walks near the goal, waits, then SIT/CLIMBs.

Evaluation-only: all scene changes are runtime patches installed by this script, so training code
and checkpoints are untouched. Candidates that pass the filters are saved to <out>/pool.
"""
import argparse
import json
import math
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import isaacgym  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

FOLLOWER_TASKS = dict(sit=1, climb=2)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num_envs', type=int, default=512)
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--batch_offset', type=int, default=0)
    p.add_argument('--follower_task', choices=tuple(FOLLOWER_TASKS), default='climb')
    p.add_argument('--follower_distance', type=float, nargs=2, default=[4.0, 6.0])
    p.add_argument('--follower_heading_deg', type=float, default=10.)
    p.add_argument('--near_radius', type=float, default=1.5)
    p.add_argument('--min_walk', type=float, default=2.0)
    p.add_argument('--min_wait_frames', type=int, default=30)
    p.add_argument('--hold_frames', type=int, default=90)
    p.add_argument('--tail_frames', type=int, default=90)
    p.add_argument('--episode_length', type=int, default=900)
    p.add_argument('--dump', action='store_true')
    p.add_argument('--agents', type=int, choices=(4, 5), default=5)
    p.add_argument('--pushers', type=int, choices=(1, 2, 3), default=3)
    opts = p.parse_args()
    pushers = (0, 2, 3)[:opts.pushers]
    rest = [c for c in range(opts.agents) if c != 1 and c not in pushers]
    if len(rest) > 1:
        raise ValueError('agents must equal pushers+1 (no follower) or pushers+2')
    opts.follower_col = rest[0] if rest else None
    return opts


def install(opts):
    from utils import sensor_stage2_spec as spec
    from utils.sensor_stage2_spec import BRIDGE, FLOOR, SURFACE_A, bridge_roles
    from env.tasks.multi_agent import square_sensor_scene as scene
    from env.tasks.multi_agent.humanoid_ma_sensor_compact import HumanoidMASensorCompact as T

    validate = spec.validate_sensor
    spec.validate_sensor = lambda env: validate(dict(env, numAgents=4, numObjects=8))

    init = T._init_bridge_curriculum

    def init_curriculum(self):
        init(self)
        for key in ('_compact_kind', '_compact_requested', '_cliff_exited', '_cliff_dropped'):
            old = getattr(self, key)
            setattr(self, key, torch.zeros(self.num_envs, self.num_agents, device=self.device, dtype=old.dtype))
        self._qual_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._qual_tries = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def draw(self, ids):
        env, m, n = self.cfg['env'], self.num_agents, len(ids)
        rows = torch.arange(n, device=self.device)
        sizes = self._box_size[ids, self._leader_bank:self._leader_bank+m]
        groups, pre, tasks, roles, push = bridge_roles(sizes, env['box']['build']['taskSizeRanges'],
            self._relation_graph_spec['template_probabilities'], self._bridge_count[ids])
        crosser = roles[:, 1]
        tasks[rows, crosser] = 3
        pre[:] = -1
        if opts.follower_col is not None:
            follower = roles[:, opts.follower_col]
            tasks[rows, follower] = FOLLOWER_TASKS[opts.follower_task]
            pre[rows, follower] = crosser
        groups = torch.where(tasks == 5, groups, torch.arange(m, device=self.device))
        groups = torch.where(pre >= 0, pre, groups)
        self._roles[ids], self._bridge_pushers[ids] = roles, push
        self._controlled_forces[self._controlled_bodies[ids].flatten()] = 0.
        first = torch.randint(1, 3, (n,), device=self.device)
        surface = torch.where(push, FLOOR, SURFACE_A)
        start = surface.clone()
        start[rows, crosser], surface[rows, crosser] = first, 3-first
        if opts.follower_col is not None:
            start[rows, follower], surface[rows, follower] = 3-first, 3-first
        self._surface[ids], self._start_surface[ids] = surface, start
        self._unreachable[ids] = False
        self._bridge_start_mode[ids] = 2
        self._controlled_present[ids] = False
        self._bridge_near[ids], self._bridge_aligned[ids] = False, False
        self._bridge_side[ids], self._bridge_carry_from_start[ids] = True, True
        self._curriculum_surfaces(ids)
        return (self._bridge_count[ids] > 1).long(), groups, pre, tasks

    sample = T._sample_episode_graph

    def sample_graph(self, ids):
        sample(self, ids)
        self._before_near_pairs[ids] = False
        self._qual_tries[ids] = 0

    place = T._place_scene

    def place_scene(self, ids):
        place(self, ids)
        if opts.follower_col is None:
            self._qual_ok[ids] = True
            return
        n = len(ids)
        rows = torch.arange(n, device=self.device)
        crosser, follower = self._roles[ids, 1], self._roles[ids, opts.follower_col]
        dest = self._surface[ids, follower]-1
        center = self._surface_center[ids, BRIDGE][rows, dest]
        half = self._surface_half[ids, BRIDGE][rows, dest]
        yaw, shift, origin = self._structure_yaw[ids, BRIDGE], self._structure_shift[ids, BRIDGE], self._env_origins[ids]
        goal = self._tar_pos[ids, self._scenario_goal(ids, crosser), :2]
        local_goal = scene.rotate_xy(goal-origin[:, :2]-shift, -yaw)
        trial = center[:, None]+(torch.rand(n, 512, 2, device=self.device)*2-1)*(half[:, None]-.6)
        distance = (trial-local_goal[:, None]).norm(dim=-1)
        lo, hi = opts.follower_distance
        ok = (distance >= lo) & (distance <= hi) & ((trial-self._bridge_landing[ids, None]).norm(dim=-1) > 1.5)
        found = ok.any(-1)
        xy = scene.to_world(trial[rows, ok.float().argmax(-1)], yaw, shift, origin)
        jitter = math.radians(opts.follower_heading_deg)*(torch.rand(n, device=self.device)*2-1)
        heading = torch.atan2(goal[:, 1]-xy[:, 1], goal[:, 0]-xy[:, 0])+jitter
        zero = torch.zeros(n, device=self.device)
        moved = self._place_person(ids, follower, xy, heading, zero)
        self._move_neutral_goal(ids, follower, moved, zero)
        self._qual_tries[ids] += 1
        self._qual_ok[ids] = found
        self._place_failed[ids[~found & (self._qual_tries[ids] < 24)]] = True

    T._init_bridge_curriculum = init_curriculum
    T._draw_episode_graph = draw
    T._sample_episode_graph = sample_graph
    T._place_scene = place_scene


def first_true(x):
    hit = x.any(0)
    return torch.where(hit, x.float().argmax(0), torch.full_like(hit, -1, dtype=torch.long))


def longest_run(x):
    run = torch.zeros(x.shape[1:], dtype=torch.long)
    best = run.clone()
    for row in x:
        run = (run+1)*row
        best = torch.maximum(best, run)
    return best


class Recorder:
    def __init__(self, task, frames):
        n, m = task.num_envs, task.num_agents
        p, d = task._box_states.shape[1], task._dof_pos.reshape(n, m, -1).shape[-1]
        self.body = torch.zeros(frames, n, m, 15, 13)
        self.roots = torch.zeros(frames, n, m, 13)
        self.dof = torch.zeros(frames, n, m, 2, d)
        self.boxes = torch.zeros(frames, n, p, 13)
        self.actions = torch.zeros(frames, n, m, d)
        self.flags = torch.zeros(frames, n, 2, dtype=torch.bool)
        self.done = torch.zeros(frames, n, m, dtype=torch.bool)
        self.fall = torch.zeros(frames, n, m, dtype=torch.bool)

    def __call__(self, t, task, action=None, terminate=None):
        n, m = task.num_envs, task.num_agents
        if t == 0:
            body = task._kinematic_humanoid_rigid_body_states.reshape(n, m, -1, 13)
        else:
            body = torch.cat((task._rigid_body_pos, task._rigid_body_rot, task._rigid_body_vel,
                              task._rigid_body_ang_vel), -1)
        self.body[t] = body.cpu()
        self.roots[t] = task._humanoid_root_states.cpu()
        self.dof[t] = torch.stack((task._dof_pos.reshape(n, m, -1), task._dof_vel.reshape(n, m, -1)), 2).cpu()
        self.boxes[t] = task._box_states.cpu()
        crossed = task._cross_completed[torch.arange(n, device=task.device), task._roles[:, 1]]
        self.flags[t] = torch.stack((task._bridge_arrived, crossed), -1).cpu()
        self.done[t] = task.relation_runtime.done.reshape(n, m).bool().cpu()
        if action is not None:
            self.actions[t] = action.view(n, m, -1).cpu()
            self.fall[t] = terminate.view(n, m).bool().cpu()


def capture(task, column):
    ids = torch.arange(task.num_envs, device=task.device)
    crosser = task._roles[:, 1]
    follower = crosser if column is None else task._roles[:, column]
    goal = task._tar_pos[ids, task._scenario_goal(ids, crosser)]
    payload = task._scenario_physical_object(ids, crosser)
    values = dict(roles=task._roles, templates=task._templates, predecessor=task._before_predecessor,
        logical_box_order=task._logical_box_order, payload=payload, bridge_slot=task._bridge_slot,
        box_size=task._box_size, goals=task._tar_pos, at_goal=goal, origin=task._env_origins,
        yaw=task._structure_yaw[:, 1], shift=task._structure_shift[:, 1],
        surface_center=task._surface_center[:, 1], surface_half=task._surface_half[:, 1],
        surface_height=task._surface_height[:, 1], start_surface=task._start_surface,
        surface=task._surface, ok=task._qual_ok, count=task._bridge_count,
        insert=task._bridge_start_mode == 2, crosser=crosser, follower=follower)
    return {k: v.detach().clone().cpu() for k, v in values.items()}


def evaluate(rec, s, opts, feet, ended):
    t_all, n = rec.roots.shape[:2]
    rows = torch.arange(n)
    c, f, p = s['crosser'], s['follower'], s['payload']
    stage_z = s['origin'][:, 2]+s['surface_height']
    fell = rec.fall.any(0).any(-1)
    arrived, crossed = first_true(rec.flags[..., 0]), first_true(rec.flags[..., 1])
    done_c, done_f = rec.done[:, rows, c], rec.done[:, rows, f]
    t_at = first_true(done_c & rec.flags[..., 1])
    frames = torch.arange(t_all)[:, None]
    after = frames > t_at[None]
    t_climb = first_true(done_f & after & (t_at >= 0)[None])
    follower = rec.roots[:, rows, f]
    goal = s['at_goal'][:, :2]
    dist = (follower[..., :2]-goal).norm(dim=-1)
    speed = follower[..., 7:9].norm(dim=-1)
    step = torch.cat((torch.zeros(1, n), (follower[1:, :, :2]-follower[:-1, :, :2]).norm(dim=-1)))
    before = (frames < t_at[None]) & (t_at >= 0)[None]
    walked = (step*before).sum(0)
    d_at = dist[t_at.clamp_min(0), rows]
    min_before = torch.where(before, dist, torch.full_like(dist, 1e3)).amin(0)
    wait = longest_run(before & (speed < .3) & (dist <= opts.near_radius+.5))
    premature = (done_f & before).any(0)
    held = longest_run(done_f & after)
    box = rec.boxes[:, rows, p]
    top = box[..., 2]+s['box_size'][rows, p, 2]/2
    feet_z = rec.body[:, rows, f][:, :, feet, 2]
    on_top = (feet_z.amin(-1) > top-.08) & (rec.body[:, rows, f, 0, 2] > top+.6)
    climb_hold = longest_run(on_top & after & done_f)
    low = (follower[..., 2] < stage_z+.6).any(0)
    end = (t_climb+opts.hold_frames+opts.tail_frames).clamp_max(t_all-1)
    checks = dict(init=s['ok'] & s['insert'] & (s['count'] == opts.pushers), no_fall=~fell, connected=arrived >= 0,
        crossed=crossed >= 0, at=t_at >= 0, climb=t_climb > 0, full_episode=ended >= t_all-1,
        held=held >= opts.hold_frames, walked=walked >= opts.min_walk, near=d_at <= opts.near_radius,
        clear=min_before >= .4, waited=wait >= opts.min_wait_frames, no_premature=~premature,
        stayed_on_platform=~low)
    if opts.follower_task == 'climb':
        checks['on_top'] = climb_hold >= opts.hold_frames
    good = torch.stack(list(checks.values())).all(0)
    score = wait/30.+.3*(t_all-1-end)/30.-2.*(d_at-1.).clamp_min(0)-.5*(walked < 3.).float()
    metrics = dict(arrived=arrived, crossed=crossed, at=t_at, climb=t_climb, end=end, walked=walked,
        d_at=d_at, min_before=min_before, wait=wait, held=held, climb_hold=climb_hold,
        initial_distance=dist[0], score=score)
    return checks, good, metrics


def rotz(v, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    out = v.copy()
    out[..., 0], out[..., 1] = c*v[..., 0]-s*v[..., 1], s*v[..., 0]+c*v[..., 1]
    return out


def quat_yaw(q, yaw):
    z, w = np.sin(yaw/2), np.cos(yaw/2)
    x, y, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack((w*x-z*y, w*y+z*x, w*qz+z*qw, w*qw-z*qz), -1)


def to_local(states, origin, yaw):
    out = states.copy()
    out[..., :3] = rotz(states[..., :3]-origin, -yaw)
    out[..., 3:7] = quat_yaw(states[..., 3:7], -yaw)
    out[..., 7:10] = rotz(states[..., 7:10], -yaw)
    out[..., 10:13] = rotz(states[..., 10:13], -yaw)
    return out


def save(path, rec, s, i, metrics, checks, meta, names):
    os.makedirs(path, exist_ok=True)
    end = int(metrics['end'][i])+1
    origin = np.concatenate((s['origin'][i, :2].numpy()+s['shift'][i].numpy(), s['origin'][i, 2:3].numpy()))
    yaw = float(s['yaw'][i])
    body = to_local(rec.body[:end, i].numpy(), origin, yaw)
    roles = s['roles'][i].tolist()
    pushers = [a for a, task in enumerate(s['templates'][i].tolist()) if task == 5]
    used = sorted({int(v) for v in s['logical_box_order'][i]})
    half, center, height = s['surface_half'][i].numpy(), s['surface_center'][i].numpy(), float(s['surface_height'][i])
    goals = rotz(s['goals'][i].numpy()-origin, -yaw)
    frames = {k: int(metrics[k][i]) for k in ('arrived', 'crossed', 'at', 'climb', 'end')}
    np.savez_compressed(os.path.join(path, 'trajectory.npz'),
        body=body, roots=to_local(rec.roots[:end, i].numpy(), origin, yaw),
        dof_pos=rec.dof[:end, i, :, 0].numpy(), dof_vel=rec.dof[:end, i, :, 1].numpy(),
        boxes=to_local(rec.boxes[:end, i].numpy(), origin, yaw), positions=body[..., :3],
        quaternions=body[..., 3:7], body_names=np.array(names['body']), dof_names=np.array(names['dof']),
        object_sizes=s['box_size'][i].numpy(), object_slots=np.array(used),
        platform_centers=np.array([[*center[k], height/2] for k in range(2)]),
        platform_quaternions=np.array([[0., 0., 0., 1.]]*2),
        platform_sizes=np.array([[*(2*half[k]), height] for k in range(2)]),
        roles=np.array(roles), payload_slot=np.array(int(s['payload'][i])), bridge_slot=np.array(int(s['bridge_slot'][i])),
        goals=goals, actions=rec.actions[1:end, i].numpy(), fps=np.array(30), world_origin=origin,
        world_yaw=np.array(yaw), logical_box_order=s['logical_box_order'][i].numpy(),
        templates=s['templates'][i].numpy(), predecessor=s['predecessor'][i].numpy(),
        event_done=rec.done[:end, i].numpy(), event_fallen=rec.fall[:end, i].numpy(),
        event_connected=rec.flags[:end, i, 0].numpy(), event_crossed=rec.flags[:end, i, 1].numpy())
    data = dict(meta, good=True, steps=end, original_steps=rec.roots.shape[0],
        role_agents=dict(pushers=pushers, AT=int(s['crosser'][i]), **{meta['follower_task'].upper(): int(s['follower'][i])}),
        first=frames, event_times_seconds={k: v/30. for k, v in frames.items()},
        metrics={k: float(v[i]) for k, v in metrics.items()}, checks={k: bool(v[i]) for k, v in checks.items()},
        pose_units='meters, radians, quaternion xyzw',
        coordinate_frame='Z-up bridge-local; inverse: world_origin + Rz(world_yaw) @ local_position')
    with open(os.path.join(path, 'metadata.json'), 'w') as f:
        json.dump(data, f, indent=2)


def make_search(opts):
    @torch.no_grad()
    def search(player):
        task = player.env.task
        n, m = task.num_envs, task.num_agents
        names = dict(body=task.gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles_all[0][0]),
                     dof=task.gym.get_actor_dof_names(task.envs[0], task.humanoid_handles_all[0][0]))
        feet = [names['body'].index('right_foot'), names['body'].index('left_foot')]
        pool = os.path.join(opts.out, 'pool')
        summary = []
        for r in range(opts.rounds):
            batch = opts.batch_offset+r
            obs = player.env_reset()
            player.get_batch_size(obs['obs'], 1)
            s = capture(task, opts.follower_col)
            rec = Recorder(task, task.max_episode_length+1)
            rec(0, task)
            alive = torch.ones(n, dtype=torch.bool, device=player.device)
            ended = torch.full((n,), task.max_episode_length, dtype=torch.long)
            resets = []
            for t in range(task.max_episode_length):
                obs = player.env_reset(resets)
                action = player.get_action(obs, player.is_determenistic)
                obs, _, done, info = player.env_step(player.env, action)
                terminate = info['terminate'].view(n, m).bool() & alive[:, None]
                rec(t+1, task, action, terminate)
                ending = done.view(n, m).bool().any(-1)
                ended[(ending & alive).cpu()] = t+1
                alive &= ~ending
                resets = ending.nonzero().flatten()*m
            checks, good, metrics = evaluate(rec, s, opts, feet, ended)
            passed = {k: int(v.sum()) for k, v in checks.items()}
            rows = torch.arange(n)
            push = s['templates'] == 5
            lift = rec.boxes[:, rows, s['payload'], 2]-s['origin'][:, 2]-s['surface_height']-s['box_size'][rows, s['payload'], 2]/2
            passed.update(pusher_fall_envs=int((rec.fall.any(0) & push).any(-1).sum()),
                crosser_fall_envs=int(rec.fall.any(0)[rows, s['crosser']].sum()),
                payload_lifted=int((lift > .15).any(0).sum()))
            if opts.dump:
                torch.save(dict(static=s, roots=rec.roots, done=rec.done, fall=rec.fall, flags=rec.flags,
                    payload=rec.boxes[:, rows, s['payload']], bridge=rec.boxes[:, rows, s['bridge_slot']],
                    metrics=metrics, checks=checks), os.path.join(opts.out, f'dump_batch{batch:04d}.pt'))
            meta = dict(batch=batch, seed=opts.seed, checkpoint=os.path.abspath(opts.checkpoint),
                follower_task=opts.follower_task, inference_only=True, pose_postprocessing=False,
                num_envs=n, agents=m, pushers=opts.pushers, bridge_start='insert', follower_distance=opts.follower_distance)
            for i in good.nonzero().flatten().tolist():
                save(os.path.join(pool, f'batch{batch:04d}_env{i:03d}'), rec, s, i, metrics, checks,
                     dict(meta, env=i), names)
            summary.append(dict(batch=batch, envs=n, passed_each=passed, good=int(good.sum())))
            print('QUAL', json.dumps(summary[-1]), flush=True)
            del rec
        os.makedirs(opts.out, exist_ok=True)
        with open(os.path.join(opts.out, f'summary_seed{opts.seed}.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print('QUAL DONE', flush=True)
        os._exit(0)
    return search


def main():
    opts = parse()
    if not os.path.abspath(opts.out).startswith(os.path.abspath('output_etc/check')):
        raise ValueError('--out must be under output_etc/check')
    os.environ.update(SCENARIO='bridge', BRIDGE_MODE='cooperative', BRIDGE_START='insert', BRIDGE_PUSHERS=str(opts.pushers))
    install(opts)
    import learning.multi_agent.viewer_diagnostics as viewer
    viewer.run_viewer_diagnostics = make_search(opts)
    sys.argv = ['tokenhsi/run.py', '--task', 'HumanoidMACoupledPush',
        '--cfg_train', 'tokenhsi/data/cfg/train/rlg/stage2_sensor_900_cliff_bridge_compact.yaml', '--cfg_env', '',
        '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml',
        '--output_path', os.path.join(opts.out, f'run_seed{opts.seed}'), '--seed', str(opts.seed), '--no_video',
        '--checkpoint', opts.checkpoint, '--num_agents', str(opts.agents), '--num_objects', str(2*opts.agents),
        '--num_envs', str(opts.num_envs), '--eval_repeats', '1', '--task_graph', 'random_scenario',
        '--task_camera', 'stack', '--eval_skills', 'loco', '--eval_skill_probs', '1.0', '--test', '--eval',
        '--headless', '--coupled_assignment', 'ordered', '--coupled_peers', 'all',
        '--episode_length', str(opts.episode_length)]+(['--group_spawn_over4'] if opts.agents > 4 else [])
    runpy.run_path('tokenhsi/run.py', run_name='__main__')


if __name__ == '__main__':
    main()
