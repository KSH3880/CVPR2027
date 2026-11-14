from check_before_push import inspect as inspect_push
from check_before_pickup_carry_rsi import main
import torch

from utils.before_object_alias import alias_layout, goal_aligned_scene


def check_virtual(task, obs):
    m, o = task.num_agents, task.num_objects
    scene = obs[:, :-m]
    packet = scene[:, -5*m:]
    dependent, source, alias, _ = alias_layout(packet, m, o)
    row, record = dependent.nonzero(as_tuple=True)
    tasks = packet.reshape(-1, m, 5).long()
    followers = tasks[row, record, 2]
    leaders = task._before_predecessor[row, followers]
    assert (leaders >= 0).all()
    goals = task._scenario_goal(row, leaders)
    saved = [value.clone() for value in (obs, task._box_states, task._tar_pos,
        task.relation_runtime.graph.edge_dst, task.relation_runtime.graph.prereq_mask)]
    virtual = goal_aligned_scene(scene, m, o)
    start = m*223+o*30+m
    poses = virtual[:, start:start+7*(2*m+o)].reshape(-1, 2*m+o, 7)
    original = scene[:, start:start+7*(2*m+o)].reshape_as(poses)
    expected = task._tar_pos[row, goals]-task._env_origins[row]
    assert torch.equal(poses[row, alias[row, record], :3], expected)
    assert torch.equal(poses[row, alias[row, record], 3:7], original[row, source[row, record], 3:7])
    changed = torch.zeros_like(scene, dtype=torch.bool)
    changed[:, start:start+7*(2*m+o)].reshape_as(poses)[row, alias[row, record], :3] = True
    assert torch.equal(virtual[~changed], scene[~changed])
    for before, after in zip(saved, (obs, task._box_states, task._tar_pos,
            task.relation_runtime.graph.edge_dst, task.relation_runtime.graph.prereq_mask)):
        assert torch.equal(before, after)


def inspect(task):
    result = inspect_push(task)
    check_virtual(task, task._compute_clean_scene_obs())
    result['virtual_xyz_real_quaternion'] = True
    result['physical_state_original_graph_unchanged'] = True
    return result


if __name__ == '__main__':
    import sys
    from pathlib import Path
    Path(sys.argv[sys.argv.index('--output_path')+1]).mkdir(parents=True, exist_ok=True)
    preset = sys.argv[sys.argv.index('--task_graph')+1]
    cases = (preset,) if preset in ('before_sit', 'before_climb') else (
        'random_scenario', 'before_stack', 'one_pair_stack', 'all_solo', 'push')
    main(inspector=inspect, presets=[(p, 'loco') for p in cases], step_inspector=check_virtual)
