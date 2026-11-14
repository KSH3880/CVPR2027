from check_before_pickup_carry_rsi import main
from check_before_virtual_object import check_virtual
import torch


def inspect(task):
    check_virtual(task, task._compute_clean_scene_obs())
    state = task._before_wait
    graph = task.relation_runtime.graph
    ids = torch.arange(task.num_envs, device=task.device)
    roots, goals = task._humanoid_root_states.clone(), task._tar_pos.clone()
    buffers = task._terminate_buf.clone(), task.reset_buf.clone()
    previous_result = getattr(task.relation_runtime, 'last_result', None)
    raw = torch.zeros_like(graph.edge_valid)
    try:
        state.reset(ids, graph, raw)
        rows, edges = state.rows, state.edges
        boxes = task._logical_box_values(task._box_states)
        xy = torch.where(state.top[:, None],
            boxes[rows, (state.source-task.num_agents).clamp(0, task.num_objects-1), :2],
            task._rigid_body_pos[rows, state.source.clamp(0, task.num_agents-1), 0, :2])
        task._tar_pos[rows, state.goal, :2] = xy
        task._humanoid_root_states[..., 7:10] = 0.
        if state.speed_dims == 3:
            task._humanoid_root_states[..., 9] = 1.
        else:
            task._humanoid_root_states[..., 7:9] = 1.
        task.relation_runtime.last_result = {'raw_own_success': raw}
        task._before_wait = None
        task._compute_reset()
        baseline = task._terminate_buf.clone()
        baseline_reset = task.reset_buf.clone()
        task._before_wait = state
        for step in range(state.grace_steps+state.violation_steps-1):
            task._compute_reset()
            assert torch.equal(task._terminate_buf, baseline)
            if state.cumulative and step == 0:
                count = state.moving_steps.clone()
                task._humanoid_root_states[..., 7:10] = 0.
                for _ in range(5):
                    task._compute_reset()
                    assert torch.equal(state.moving_steps, count)
                    assert torch.equal(task._terminate_buf, baseline)
                task._humanoid_root_states[..., 9] = 1.
        task._compute_reset()
        expected = torch.zeros(task.num_envs, task.num_agents, dtype=torch.long, device=task.device)
        expected[rows, state.owners] = 1
        assert torch.equal(task._terminate_buf, baseline | expected.flatten())
        assert torch.equal(task.reset_buf, baseline_reset | expected.any(-1).long())
        raw[rows, state.predecessor] = True
        task._compute_reset()
        assert torch.equal(task._terminate_buf, baseline)
        raw.zero_()
        task._compute_reset()
        assert torch.equal(task._terminate_buf, baseline)
        saved = [v[1:].clone() for v in (state.armed, state.released, state.age, state.moving_steps)]
        state.reset(ids[:1], graph, raw)
        assert all(torch.equal(v[1:], before) for v, before in
            zip((state.armed, state.released, state.age, state.moving_steps), saved))
        task.cfg['args'].eval_protocol = 'general'
        age = state.age.clone()
        task._compute_reset()
        assert torch.equal(state.age, age)
    finally:
        task.cfg['args'].eval_protocol = ''
        task._before_wait = state
        task._humanoid_root_states.copy_(roots)
        task._tar_pos.copy_(goals)
        task._terminate_buf.copy_(buffers[0])
        task.reset_buf.copy_(buffers[1])
        if previous_result is None:
            del task.relation_runtime.last_result
        else:
            task.relation_runtime.last_result = previous_result
        state.reset(ids, graph, task.relation_runtime.own_success)
    return dict(waiting_reset_integration=True, speed_dimensions=state.speed_dims,
        cumulative=state.cumulative, grace_steps=state.grace_steps, same_frame_release=True,
        latched_release=True, partial_wait_reset=True, formal_protocol_preserved=True)


def step_inspector(task, obs):
    check_virtual(task, obs)
    assert torch.isfinite(task.rew_buf).all()


if __name__ == '__main__':
    import sys
    from pathlib import Path
    Path(sys.argv[sys.argv.index('--output_path')+1]).mkdir(parents=True, exist_ok=True)
    preset = sys.argv[sys.argv.index('--task_graph')+1]
    cases = (preset,) if preset in ('before_sit', 'before_climb') else (
        'random_scenario', 'before_stack', 'all_solo', 'push')
    main(inspector=inspect, presets=[(p, 'loco') for p in cases],
        step_inspector=step_inspector)
