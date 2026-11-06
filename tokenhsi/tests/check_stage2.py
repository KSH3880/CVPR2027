import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
from torch import nn

from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from utils.before_object_alias import alias_layout, goal_aligned_scene
from utils.task_role_spec import task_graph_packet
from utils.stage2_spec import before_members, evaluation_groups
from utils.coupled_push_spec import check_checkpoint, group_sizes
from env.tasks.multi_agent.stage2_layout import invalid_before_layout
from learning.multi_agent.stage2_push_network import Stage2PushBuilder
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from learning.multi_agent.human_coordination import stage2_human_relations


def inspect(task, obs):
    n, m = task.num_envs, task.num_agents
    ids = torch.arange(n, device=task.device)
    packet = task_graph_packet(task.relation_runtime.graph, n)
    records = packet.reshape(n, m, 5).long()
    relation = stage2_human_relations(packet, task._groups)
    before = before_members(task._before_predecessor)
    shared = (relation == 3).any(-1)
    assert not (before & shared).any()
    assert torch.equal(task._members > 1, shared)
    assert task._regions[~shared].eq(0).all()
    assert not invalid_before_layout(task, ids).any()
    assert torch.equal(obs[:, -m:].long(), task._groups)
    dependent, target, alias, _ = alias_layout(packet, m, task.num_objects)
    rows, rec = dependent.nonzero(as_tuple=True)
    mapping = task._logical_box_order
    assert torch.equal(mapping[rows, alias[rows, rec]-m], mapping[rows, target[rows, rec]-m])
    scene = obs[:, :-m]
    virtual = goal_aligned_scene(scene, m, task.num_objects)
    assert torch.isfinite(virtual).all()
    groups = evaluation_groups(packet, task._groups)
    assert torch.equal(group_sizes(groups) > 1, shared)
    if task._task_graph_preset == 'mixed':
        assert (before.sum(-1) >= 2).all() and (shared.sum(-1) >= 2).all()
    if m > 4:
        assert task._members.max() <= 4
        for row in range(n):
            centers = torch.stack([task._humanoid_root_states[row, task._groups[row] == group, :2].mean(0)
                                   for group in task._groups[row].unique()])
            if len(centers) > 1:
                assert float(torch.pdist(centers).min()) >= 5.-.0005
        retained = torch.zeros(n, task.num_physical_objects, dtype=torch.long, device=task.device)
        retained.scatter_add_(1, mapping, task._stage2_task_slots.long())
        distance = (task._box_states[..., :2]-task._env_origins[:, None, :2]).norm(dim=-1)
        assert (distance[retained == 0] >= 1000.).all()
    return dict(preset=task._task_graph_preset, before_people=int(before.sum()),
                coupled_people=int(shared.sum()), domains=torch.bincount(task._scene_domains, minlength=3).tolist(),
                aliases=int(dependent.sum()), finite=True)


def main():
    torch.set_num_threads(1)
    args = get_args(); args.test = args.eval = True
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'] = args.motion_file
    set_seed(train['params']['seed'])
    task, env = parse_task(args, cfg, train['params']['config'], parse_sim_params(args, cfg, train))
    builder = Stage2PushBuilder(); builder.load(train['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(task.get_obs_size(),),
        amp_input_shape=(task.get_num_amp_obs(),), value_size=1, num_agents=task.num_agents, num_objects=task.num_objects,
        observation_mode='clean_scene', scene_entity_sizes=task.get_scene_entity_sizes(),
        scene_kinematic_size=7, scene_arena_scale=task.get_scene_arena_scale(),
        relation_reward_mode=task._relation_cfg['mode'], relation_graph_spec=task._relation_graph_spec,
        device=task.device).to(task.device)
    teachers = bool(args.teacher_before)
    if teachers:
        from learning.multi_agent.stage2_teachers import Stage2Teachers
        model = Stage2Teachers(task, args.teacher_before, args.teacher_coupled)
    else:
        model = nn.Module(); model.a2c_network = net
        checkpoint = torch.load(args.checkpoint, map_location=task.device, weights_only=False)
        check_checkpoint(checkpoint, cfg['env'], inference=True)
        model.load_state_dict(checkpoint['model'], strict=True); model.eval()
        rms = SceneRunningMeanStd([223, 30, 1], [task.num_agents, task.num_objects, task.num_agents],
            normalized_sizes=[223, 30, 0], kinematic_size=7,
            extra_passthrough_size=6*task.num_agents).to(task.device).eval()
        rms.load_state_dict(checkpoint['running_mean_std'], strict=True)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    cases = []
    with torch.no_grad():
        for preset in ('random_scenario', 'before', 'coupled', 'mixed', 'all_solo', 'all_coupled'):
            task._task_graph_preset = preset
            obs = env.reset(); case = inspect(task, obs)
            assert torch.isfinite(task.fetch_amp_obs_demo(512)).all()
            for _ in range(8):
                valid = task._stage2_task_slots if task.num_agents > 4 else None
                if teachers:
                    mu = model.means(obs, task._members > 1, valid)
                    value = model.before.a2c_network.eval_critic(model.before.normalizer(obs))
                else:
                    net.object_valid = valid
                    mu, _ = net.eval_actor(rms(obs)); value = net.eval_critic(rms(obs))
                obs, reward, _, _ = env.step(mu)
                assert all(torch.isfinite(x).all() for x in (obs, reward, mu, value))
            roots = task._root_states.view(task.num_envs, -1, 13)[1:].clone()
            pre = task._before_predecessor[1:].clone()
            wait = task._before_wait.armed[1:].clone()
            task._reset_envs(torch.tensor([0], device=task.device))
            assert torch.equal(roots, task._root_states.view(task.num_envs, -1, 13)[1:])
            assert torch.equal(pre, task._before_predecessor[1:])
            assert torch.equal(wait, task._before_wait.armed[1:])
            cases.append(dict(partial_reset=True, **case))
            print('STAGE2_CHECK', cases[-1], flush=True)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in weights.items())
    out = Path(args.output_path); out.mkdir(parents=True, exist_ok=True)
    (out/'physics_validation.json').write_text(json.dumps(dict(weights_unchanged=True, cases=cases), indent=2))
    print('STAGE2 CHECK COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
