"""Group-complete case search and original-task live policy viewing (evaluation only)."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from ksh_jointpush.ksh_success_cases import first_hits

CATEGORIES = ('1+3', '2+2', '4')

def category(groups):
    _, counts = np.unique(np.asarray(groups), return_counts=True)
    return '+'.join(map(str, sorted(counts.tolist())))


def complete_success(flags, groups):
    hits = first_hits(flags)
    return category(groups) in CATEGORIES and bool((hits >= 0).all())


def center_scene(saved):
    """Move world positions to env0, preserving velocities and local geometry."""
    tensors = saved['tensors']
    offset = tensors['_env_origins'][0].clone()
    for name in ('_humanoid_root_states', '_initial_humanoid_root_states',
                 '_initial_humanoid_rigid_body_states', '_rigid_body_pos',
                 '_prev_root_pos', '_prev_box_pos', '_tar_pos', '_box_states',
                 '_initial_box_states', '_platform_states', '_platform_pos',
                 '_platform_default_pos', '_tar_platform_states', '_tar_platform_pos',
                 '_tar_platform_default_pos', '_kinematic_humanoid_rigid_body_states'):
        if name in tensors:
            tensors[name] = tensors[name].clone()
            tensors[name][..., :3] -= offset
    tensors['_env_origins'] = torch.zeros_like(tensors['_env_origins'])
    saved['roots'] = saved['roots'].clone()
    saved['roots'][..., :3] -= offset
    return saved


def snapshot(task):
    """Reset-time scene rows, excluding simulator indices and static body index tables."""
    n = task.num_envs
    skip = ('actor_ids', 'body_ids', 'handles')
    tensors = {}
    for name, value in vars(task).items():
        if (torch.is_tensor(value) and value.ndim and value.shape[0] == n
                and not any(s in name for s in skip)):
            tensors[name] = value.detach().cpu().clone()
    nested = {}
    for prefix, obj in (('runtime', task.relation_runtime), ('graph', task.relation_runtime.graph)):
        nested[prefix] = {k: v.detach().cpu().clone() for k, v in vars(obj).items()
                          if torch.is_tensor(v) and v.ndim and v.shape[0] == n}
    return dict(tensors=tensors, nested=nested,
                roots=task._root_states.view(n, -1, 13).cpu().clone(),
                dofs=task._dof_state.view(n, -1, 2).cpu().clone(),
                density=task._box_density.view(n, -1).cpu().clone())


def scene_snapshot(state, env_id):
    return dict(tensors={k: v[env_id:env_id+1].clone() for k, v in state['tensors'].items()},
                nested={p: {k: v[env_id:env_id+1].clone() for k, v in values.items()}
                        for p, values in state['nested'].items()},
                roots=state['roots'][env_id].clone(), dofs=state['dofs'][env_id].clone(),
                density=state['density'][env_id].clone())


def install_collect():
    from learning.multi_agent.ma_players import MAPlayerContinuous
    from utils import stage2_jointpush as spec
    original_groups = spec.sample_groups
    def groups(count, people, probability, device, preset='random_scenario'):
        # Only requested group compositions; retain random member identities and leaders.
        order = torch.rand(count, people, device=device).argsort(-1)
        choices = torch.arange(count, device=device) % 3
        result = original_groups(count, people, probability, device, preset)
        for index, sizes in enumerate(((1, 3), (2, 2), (4,))):
            rows = (choices == index).nonzero().flatten()
            start = 0
            for size in sizes:
                members = order[rows, start:start+size]
                result[rows[:, None], members] = members[:, :1]
                start += size
        return result
    spec.sample_groups = groups

    @torch.no_grad()
    def run(player):
        task = player.env.task
        if task.viewer is not None:
            raise ValueError('Group search must be headless')
        output = Path(task.cfg['args'].output_path).resolve()
        output.mkdir(parents=True, exist_ok=True)
        quota = int(os.environ.get('KSH_GROUP_QUOTA', '2'))
        rounds = int(os.environ.get('KSH_SUCCESS_ROUNDS', '64'))
        n, m = task.num_envs, task.num_agents
        cases, trials = [], {k: 0 for k in CATEGORIES}
        counts = {k: 0 for k in CATEGORIES}
        report_path = output / 'cases.json'
        def report():
            result = dict(cases=cases, completed_trials=trials, found=counts,
                          quota_per_category=quota, deterministic=True, init_mode='loco',
                          selection='Every group reaches its own official goal after initialization',
                          training_started=False)
            report_path.write_text(json.dumps(result, indent=2) + '\n')
            return result
        for repeat in range(rounds):
            obs = player.env_reset()
            player.get_batch_size(obs['obs'], n)
            initial = snapshot(task)
            grouped = task._groups.cpu().numpy().copy()
            flags = np.zeros((task.max_episode_length + 1, n, m), bool)
            flags[0] = task.relation_runtime.done.cpu().numpy()
            alive = np.ones(n, bool)
            resets = []
            for frame in range(1, task.max_episode_length + 1):
                obs = player.env_reset(resets)
                action = player.get_action(obs, True)
                obs, _, done, _ = player.env_step(player.env, action)
                flags[frame] = task.relation_runtime.done.cpu().numpy()
                ending = done.reshape(n, m).bool().any(-1).cpu().numpy()
                finished = np.flatnonzero(ending & alive)
                for env_id in finished:
                    kind = category(grouped[env_id])
                    trials[kind] += 1
                    episode = flags[:frame+1, env_id]
                    if counts[kind] < quota and complete_success(episode, grouped[env_id]):
                        name = 'case_' + str(len(cases)+1).zfill(3)
                        path = output / (name + '.pt')
                        saved = scene_snapshot(initial, env_id)
                        saved['success'] = torch.from_numpy(episode.copy())
                        torch.save(saved, path)
                        meta = dict(file=str(path), category=kind, groups=grouped[env_id].tolist(),
                                    repeat=repeat, env_id=int(env_id), frames=frame+1,
                                    first_hit_frames=first_hits(episode).tolist(),
                                    simultaneous=bool(episode.all(-1).any()),
                                    final_all=bool(episode[-1].all()),
                                    checkpoint=str(task.cfg['args'].checkpoint))
                        path.with_suffix('.json').write_text(json.dumps(meta, indent=2)+'\n')
                        cases.append(meta)
                        counts[kind] += 1
                        print('KSH_GROUP_SUCCESS', json.dumps(meta), flush=True)
                        report()
                alive[finished] = False
                resets = torch.as_tensor(np.flatnonzero(ending), device=player.device, dtype=torch.long)*m
                if frame % 300 == 0:
                    print('KSH_GROUP_PROGRESS', repeat, frame, counts, trials, flush=True)
                if not alive.any():
                    break
            report()
            if all(counts[k] >= quota for k in CATEGORIES):
                break
        print('KSH_GROUP_SEARCH_DONE', json.dumps(report()), flush=True)
        return report()
    MAPlayerContinuous.run = MAPlayerContinuous.run_eval = run


def install_live():
    """Recreate original physical assets, restore initial scene, then infer policy every step."""
    from learning.multi_agent.ma_players import MAPlayerContinuous
    from env.tasks.multi_agent.humanoid_ma_stage2_jointpush_rsi import HumanoidMAStage2JointPushRsi as Task
    from utils import stage2_jointpush as spec
    from isaacgym import gymapi, gymtorch
    from ksh_jointpush.ksh_live_control import read_json, write_json
    control = Path(os.environ['KSH_LIVE_CONTROL']) if os.environ.get('KSH_LIVE_CONTROL') else None
    if control:
        write_json(control/'boot.json', dict(pid=os.getpid()))
    case_path = Path(os.environ['KSH_LIVE_CASE']).resolve()
    saved = center_scene(torch.load(case_path, map_location='cpu'))
    meta = json.loads(case_path.with_suffix('.json').read_text())
    camera_mode = os.environ.get('KSH_VIEW_CAMERA', 'behind' if 'hand_quality' in meta else 'legacy')
    if camera_mode not in ('behind', 'legacy'):
        raise ValueError('KSH_VIEW_CAMERA must be behind or legacy')
    assigned_box = int(saved['tensors']['_agent_box_assignment'][0, 0])
    q = saved['roots'][4+assigned_box, 3:7].numpy()
    # Preserve the initial push heading even after the box passes the goal.
    forward = np.array([2*(q[0]*q[1]-q[2]*q[3]), 1-2*(q[0]*q[0]+q[2]*q[2])])
    original_update_camera = Task._update_camera
    def update_camera(self):
        if camera_mode != 'behind' or not self._camera_follow_flag:
            return original_update_camera(self)
        from ksh_jointpush.ksh_view_camera import behind_camera
        people = self._humanoid_root_states[0, :, :3].cpu().numpy()
        box = self._box_states[0, assigned_box, :3].cpu().numpy()
        width = float(self._box_size[0, assigned_box, 0])
        eye, target = behind_camera(people, box, width, forward)
        self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(*eye), gymapi.Vec3(*target))
        self._cam_prev_char_pos[:] = people[0]
    Task._update_camera = update_camera
    source_batch = int(os.environ.get('KSH_LIVE_POLICY_BATCH', '256'))
    # This viewer always restores LOCO-start PUSH scenes; unused skill libraries
    # do not participate in policy inference or physical simulation.
    Task._motion_skills_to_load = lambda self, available: [s for s in available if s in ('loco', 'push_loco', 'push')]
    def bank(count, env, device):
        if count != 1:
            raise ValueError('Live success viewer requires exactly one environment')
        sizes = saved['tensors']['_box_size'].to(device)
        mass = saved['density'].to(device)[None] * sizes[..., 0]*sizes[..., 1]*sizes[..., 2]
        return sizes.clone(), mass
    spec.asset_bank = bank
    def reset(self, ids):
        if not len(ids):
            return
        for name, source in saved['tensors'].items():
            target = getattr(self, name, None)
            if torch.is_tensor(target) and target.shape == source.shape:
                target.copy_(source.to(target.device))
        for prefix, obj in (('runtime', self.relation_runtime), ('graph', self.relation_runtime.graph)):
            for name, source in saved['nested'][prefix].items():
                target = getattr(obj, name)
                target.copy_(source.to(target.device))
        roots = self._root_states.view(1, -1, 13)
        roots[0, :len(saved['roots'])] = saved['roots'].to(self.device)
        self._dof_state.copy_(saved['dofs'].to(self.device))
        self._reset_env_tensors(ids)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))
        self._refresh_sim_tensors()
        self._compute_observations(ids)
        reference_obs = saved['tensors']['obs_buf'].to(self.device)
        error = (self.obs_buf-reference_obs).abs().max().item()
        print('KSH_LIVE_INITIAL_OBSERVATION_MAX_ERROR', error, flush=True)
        if error > 1e-3:
            raise RuntimeError('Restored scene observation differs from the selected episode')
        self._update_box_assignment_colors(ids)
        if self.viewer:
            pos = self._humanoid_root_states[0, :, :3].mean(0).cpu().numpy()
            self._cam_prev_char_pos[:] = self._humanoid_root_states[0, 0, :3].cpu().numpy()
            self._camera_follow_flag = camera_mode == 'behind'
            if camera_mode == 'behind':
                update_camera(self)
            else:
                self.gym.viewer_camera_look_at(self.viewer, None,
                    gymapi.Vec3(float(pos[0]+7), float(pos[1]-10), 7),
                    gymapi.Vec3(float(pos[0]), float(pos[1]+1), .8))
        print('KSH_LIVE_RESET', meta['category'], meta['groups'], flush=True)
    Task._reset_envs = reset
    original_render = Task.render
    class Events:
        def __init__(self, gym, task):
            self.gym, self.task = gym, task
        def __getattr__(self, name):
            return getattr(self.gym, name)
        def query_viewer_action_events(self, viewer):
            result = []
            for event in self.gym.query_viewer_action_events(viewer):
                if event.value > 0:
                    if event.action == 'ksh_next_episode':
                        self.task._live_exit_code = 44
                    elif event.action.startswith('live_category_'):
                        self.task._live_exit_code = 41 + int(event.action[-1])
                if not event.action.startswith('live_category_'):
                    if not (control and event.action == 'ksh_next_episode'):
                        result.append(event)
            return result
    def render(self, *args, **kwargs):
        if not self.viewer:
            return original_render(self, *args, **kwargs)
        if not getattr(self, '_live_keys', False):
            for index, key in enumerate((gymapi.KEY_1, gymapi.KEY_2, gymapi.KEY_3)):
                self.gym.subscribe_viewer_keyboard_event(self.viewer, key, 'live_category_'+str(index))
            self._live_keys = True
            print('KSH_LIVE_KEYS 1=1+3, 2=2+2, 3=4, N=next case, Esc=quit', flush=True)
        gym = self.gym
        self.gym = Events(gym, self)
        try:
            return original_render(self, *args, **kwargs)
        finally:
            self.gym = gym
    Task.render = render

    @torch.no_grad()
    def run(player):
        task = player.env.task
        output = Path(task.cfg['args'].output_path).resolve()
        output.mkdir(parents=True, exist_ok=True)
        if task.viewer and not control:
            active_path = Path(__file__).resolve().parents[1] / 'runs/jointpush/ksh_jointpush_active_view.json'
            active = dict(kind='live_success_policy', state='running', pid=os.getpid(),
                          epoch=5500, case=meta, checkpoint=str(task.cfg['args'].checkpoint),
                          run=str(output.parent), floor_centered=True, display=os.environ.get('DISPLAY'),
                          training_state='stopped', selection='Every original group reached its goal',
                          live_result_may_differ=True)
            active_path.write_text(json.dumps(active, indent=2)+'\n')
        repeats = int(os.environ.get('KSH_LIVE_REPEATS', '100000'))
        def managed_request(code):
            sequence = time.time_ns()
            torch.cuda.synchronize()
            write_json(control/'request.json', dict(sequence=sequence, code=code, pid=os.getpid()))
            task._live_exit_code = 0
            print('KSH_LIVE_SWITCH_WAIT', code, flush=True)
            while True:
                state = read_json(control/'state.json', {})
                if state.get('mode') == 'quit':
                    raise SystemExit(0)
                if state.get('acknowledged', 0) >= sequence:
                    task._live_exit_code = 0
                    print('KSH_LIVE_SWITCH_RESUME', code, flush=True)
                    return
                task.render()
                time.sleep(.03)
        for repeat in range(repeats):
            obs = player.env_reset()
            player.get_batch_size(obs['obs'], 1)
            flags = [task.relation_runtime.done.cpu().numpy()[0].copy()]
            for frame in range(task.max_episode_length):
                if control and read_json(control/'state.json', {}).get('mode') == 'quit':
                    torch.cuda.synchronize()
                    raise SystemExit(0)
                obs = player.env_reset([])
                # Preserve the collection GEMM batch shape to reduce numerical
                # trajectory drift while simulating only the selected scene.
                policy_obs = dict(obs)
                policy_obs['obs'] = obs['obs'].expand(source_batch, -1).contiguous()
                actions = player.get_action(policy_obs, True)
                action = actions.reshape(source_batch, task.num_agents, -1)[0]
                obs, _, done, _ = player.env_step(player.env, action)
                if control and not (control/'ready.json').exists():
                    torch.cuda.synchronize()
                    write_json(control/'ready.json', dict(pid=os.getpid(), case=meta))
                if getattr(task, '_live_exit_code', 0):
                    if control:
                        managed_request(task._live_exit_code)
                    else:
                        raise SystemExit(task._live_exit_code)
                flags.append(task.relation_runtime.done.cpu().numpy()[0].copy())
                if done.any():
                    break
            flags = np.asarray(flags)
            result = dict(category=meta['category'], case=str(case_path), repeat=repeat,
                          first_hit_frames=first_hits(flags).tolist(),
                          all_groups_succeeded=complete_success(flags, meta['groups']),
                          simultaneous=bool(flags.all(-1).any()), final_all=bool(flags[-1].all()),
                          frames=len(flags), live_policy=True, physics=True)
            (output / ('verification_' + str(repeat) + '.json')).write_text(json.dumps(result, indent=2)+'\n')
            print('KSH_LIVE_RESULT', json.dumps(result), flush=True)
            if task.viewer and getattr(task, '_viewer_manual_skip', False):
                break
        return result
    MAPlayerContinuous.run = MAPlayerContinuous.run_eval = run
