"""Live physics/observation checks for the fixed-budget BEFORE role tokens."""
from check_before_pickup_carry_rsi import main, inspect
import torch

from utils.before_object_alias import MODE, alias_layout, encoder_packet
from utils.task_role_spec import task_edges
from learning.multi_agent.human_coordination import human_relations


def check_observations(task, obs):
    m, o = task.num_agents, task.num_objects
    packet = obs[:, -5*m:]
    dependent, source, alias, count = alias_layout(packet, m, o)
    row, record = dependent.nonzero(as_tuple=True)
    mapping = task._logical_box_order
    assert torch.equal(mapping[row, alias[row, record]-m], mapping[row, source[row, record]-m])
    objects = obs[:, m*223:m*223+o*30].reshape(-1, o, 30)
    start = m*223+o*30+m
    poses = obs[:, start:start+(2*m+o)*7].reshape(-1, 2*m+o, 7)
    assert torch.equal(objects[row, alias[row, record]-m], objects[row, source[row, record]-m])
    assert torch.equal(poses[row, alias[row, record]], poses[row, source[row, record]])
    converted = encoder_packet(packet, m, o)
    valid, src, dst, _ = task_edges(converted)
    tasks = packet.reshape(-1, m, 5).long()
    # Original relation packet and all reward edges still point at the source.
    relations = human_relations(packet, m, before=True)
    expected = torch.eye(m, device=task.device, dtype=torch.long).expand(task.num_envs, -1, -1).clone()
    es, owner = (task._before_predecessor >= 0).nonzero(as_tuple=True)
    expected[es, owner, task._before_predecessor[es, owner]] = 2
    assert torch.equal(relations, expected)
    actors = tasks[row, record, 2]
    sources, aliases = source[row, record], alias[row, record]
    links = valid[row] & (src[row] == actors[:, None]) & (dst[row] == aliases[:, None])
    assert links.any(-1).all()
    assert not (valid[row] & (src[row] == actors[:, None]) & (dst[row] == sources[:, None])).any()
    return dependent.sum(-1), count


def inspect_alias(task):
    assert task._relation_graph_spec['before_object_alias'] == MODE
    result = inspect(task)
    pairs, count = check_observations(task, task._compute_clean_scene_obs())
    ids = torch.arange(task.num_envs, device=task.device)
    active = task._active_physical(ids)
    assert torch.equal(active.sum(-1), task.num_objects-pairs)
    assert torch.equal(task._task_token_count, count+pairs)
    far = (task._box_states[..., :2]-task._env_origins[:, None, :2]).norm(dim=-1)
    assert (far[~active] > 100).all()
    from env.tasks.multi_agent.shared_support_scene import scene_infeasible
    assert not scene_infeasible(task, ids).any()
    if task.cfg['env']['before']['distance_mixture'].get('mode') == 'relative_sum':
        from env.tasks.multi_agent.before_relative_sampling import invalid
        assert not invalid(task, ids).any()
    else:
        # Verify the mix226 reset geometry directly, without imposing near50's sum inequality.
        from utils.before_sampling import MIX226_DISTANCE_MIXTURE
        assert task.cfg['env']['before']['distance_mixture'] == MIX226_DISTANCE_MIXTURE
        es, follower = (task._before_predecessor >= 0).nonzero(as_tuple=True)
        leader = task._before_predecessor[es, follower]
        mode = task._before_distance_modes[es, follower]
        template = task._templates[es, follower]
        mix = MIX226_DISTANCE_MIXTURE
        box = task._box_states[es, leader, :2]
        root = task._humanoid_root_states[es, follower, :2]
        leader_root = task._humanoid_root_states[es, leader, :2]
        bounds = root.new_tensor(mix['follower_distance'])[mode]
        bounds = torch.where((template == 4)[:, None], root.new_tensor(mix['ontop_follower_distance'])[mode], bounds)
        bounds = torch.where(((template == 1) & (mode == 0))[:, None], root.new_tensor(mix['sit_original_distance']), bounds)
        def within(distance, limits):
            assert ((distance >= limits[:, 0]-3e-5) & (distance <= limits[:, 1]+3e-5)).all()
        within((root-box).norm(dim=-1), bounds)
        within((leader_root-box).norm(dim=-1), root.new_tensor(mix['leader_distance'])[mode])
        # RSI keeps the native human/payload offset; only loco samples this radius.
        top = (template == 4) & ~task._before_carry_mask[es, follower]
        payload = task._box_states[es[top], follower[top], :2]
        within((payload-root[top]).norm(dim=-1), root.new_tensor(mix['payload_distance'])[mode[top]])
        goal = task._scenario_goal(es, leader)
        within((task._tar_pos[es, goal, :2]-box).norm(dim=-1), root.new_tensor(mix['leader_goal_distance'])[mode])
        if not task.cfg['env']['before'].get('follower_pickup_carry_rsi'):
            assert not task._before_rsi_skill.any()
        result['mix226_distances_verified'] = True
    physical = task._logical_box_order
    for row in range(task.num_envs):
        assert len(physical[row].unique()) == task.num_objects-int(pairs[row])
        # Every duplicate lies in the reserved task prefix; distractors are unique.
        used = physical[row, :task._task_token_count[row]]
        other = physical[row, task._task_token_count[row]:]
        assert not (other[:, None] == used[None]).any()
    result.update(alias_tokens=int(pairs.sum()), object_tokens=task.num_objects,
        physical_count_range=[int(active.sum(-1).min()), int(active.sum(-1).max())],
        shared_obs_pose_exact=True, before_relations_preserved=True, hidden_distractors=False)
    return result


if __name__ == '__main__':
    import sys
    from pathlib import Path
    import yaml
    preset = sys.argv[sys.argv.index('--task_graph')+1] if '--task_graph' in sys.argv else 'random_scenario'
    cfg = yaml.safe_load(Path(sys.argv[sys.argv.index('--cfg_env')+1]).read_text())
    cases = [(preset, 'loco')] if preset in ('before_sit', 'before_climb') else [
        ('random_scenario', 'loco'), ('random_scenario', 'mix'),
        ('before_stack', 'pickup'), ('before_stack', 'carry'), ('all_solo', 'mix')]
    if preset not in ('before_sit', 'before_climb') and not cfg['env']['before'].get('follower_pickup_carry_rsi'):
        cases = [(name, 'loco') for name in ('random_scenario', 'before_stack', 'one_pair_stack', 'all_solo')]
    main(inspector=inspect_alias, step_inspector=check_observations, presets=cases)
