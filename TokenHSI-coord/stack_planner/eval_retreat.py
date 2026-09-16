"""Deterministic, episode-aware retreat evaluation; no training updates."""
from .train_closed_loop import (
    torch, os, json, Path, get_args, load_cfg, set_seed, _make_player,
    _refresh_obs, load_stack_checkpoint,
)
from .retreat_metrics import summarize_retreat


@torch.no_grad()
def main():
    args = get_args()
    cfg, cfg_train, _ = load_cfg(args)
    seed = set_seed(cfg_train['params'].get('seed', 0), False)
    cfg_train['params']['seed'] = seed
    cfg_train['params']['config']['seed'] = seed
    cfg_train['params']['config']['train_dir'] = args.output_path
    if args.motion_file:
        cfg['env']['motion_file'] = args.motion_file
    player = _make_player(args, cfg, cfg_train)
    task = player.env.task
    planner, payload = load_stack_checkpoint(os.environ['STACK_PLANNER_EVAL_CKPT'], player.device)
    out = Path(os.environ['STACK_PLANNER_EVAL_OUTPUT'])
    out.mkdir(parents=True, exist_ok=False)
    steps = int(os.environ.get('STACK_PLANNER_EVAL_STEPS', '3600'))
    period = int(os.environ.get('STACK_PLANNER_EVAL_REPLAN', '30'))
    if steps <= 0 or period <= 0:
        raise ValueError('steps/replan must be positive')
    episodes = [dict(env=e, episode=0, retreat_steps=0, endpoint_changes=[],
                     path_errors=[], min_box_gap=None, min_agent_gap=None,
                     collision_steps=0, fall=False, reached=False, stopped=False,
                     approaching_steps=0, stationary_steps=0) for e in range(task.num_envs)]
    previous_endpoint = [None] * task.num_envs
    previous_phase = None
    obs = _refresh_obs(player)
    player.get_batch_size(obs, 1)
    records = []
    with (out / 'episodes.jsonl').open('x') as stream:
        for tick in range(steps):
            phase = task._stack_phase.clone()
            retreat = (phase == task.A1_RETREAT) & ~task._carry_rehearsal
            if tick % period == 0 or previous_phase is None or bool((phase != previous_phase).any()):
                output = planner(task.planner_state())
                task.install_external_plan(output)
                endpoints = output['path_world'][:, 0, 0, -1].cpu().tolist()
                for e in range(task.num_envs):
                    if retreat[e]:
                        old = previous_endpoint[e]
                        if old is not None:
                            episodes[e]['endpoint_changes'].append(sum((a-b)**2 for a,b in zip(endpoints[e], old))**.5)
                        previous_endpoint[e] = endpoints[e]
                obs = _refresh_obs(player)
            previous_phase = phase
            action = player.get_action({'obs': obs}, is_determenistic=True)
            obs, _, done, _ = player.env.step(action)
            state = task.planner_state()
            roots = task.humanoid_rows(task._humanoid_root_states).reshape(task.num_envs, 2, -1)
            collision = task.planner_collision_cost()
            fall = task.planner_fall()
            distance = (state.root_xy[:, 0] - task._planner_virtual_retreat_pos[:, :2]).norm(dim=-1)
            speed = roots[:, 0, 7:9].norm(dim=-1)
            relative = state.root_xy[:, 1] - state.root_xy[:, 0]
            closing = -(relative * roots[:, 1, 7:9]).sum(-1) / relative.norm(dim=-1).clamp(min=1e-6)
            box_gap = (state.root_xy[:, 0] - state.box_xyz[:, 0, :2]).norm(dim=-1) - .5 * state.box_size_xy[:, 0].norm(dim=-1) - .35
            for e in range(task.num_envs):
                row = episodes[e]
                if retreat[e]:
                    row['fall'] |= bool(fall[e])
                    row['retreat_steps'] += 1
                    row['path_errors'].append(float(task._lat_root.reshape(task.num_envs, 2)[e, 0].abs()))
                    for name, value in [('min_box_gap', float(box_gap[e])), ('min_agent_gap', float(relative[e].norm()) - 1.)]:
                        row[name] = value if row[name] is None else min(row[name], value)
                    row['collision_steps'] += int(collision[e] > 0)
                    row['reached'] |= bool(distance[e] <= .25)
                    row['stopped'] |= bool((distance[e] <= .25) & (speed[e] <= .15))
                    row['approaching_steps'] += int(closing[e] > .1)
                    row['stationary_steps'] += int(roots[e, 1, 7:9].norm() <= .1)
            done_env = done.reshape(task.num_envs, 2).any(-1)
            for e in range(task.num_envs):
                if done_env[e] or tick == steps-1:
                    row = episodes[e]
                    row['completed'] = bool(done_env[e])
                    stream.write(json.dumps(row) + '\n')
                    records.append(row.copy())
                    if done_env[e]:
                        number = row['episode'] + 1
                        episodes[e] = dict(env=e, episode=number, retreat_steps=0, endpoint_changes=[], path_errors=[], min_box_gap=None, min_agent_gap=None, collision_steps=0, fall=False, reached=False, stopped=False, approaching_steps=0, stationary_steps=0)
                        previous_endpoint[e] = None
            if done_env.any():
                ids = torch.nonzero(done_env.repeat_interleave(2), as_tuple=False).squeeze(-1)
                obs = player.env.reset(ids)
                previous_phase = None
    summary = dict(checkpoint=os.environ['STACK_PLANNER_EVAL_CKPT'], checkpoint_step=payload['step'],
                   seed=seed, steps=steps, envs=task.num_envs, replan_steps=period,
                   **summarize_retreat(records))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('[stack-planner-retreat-eval] ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
