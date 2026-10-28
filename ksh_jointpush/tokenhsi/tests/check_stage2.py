import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaacgym
import torch
from torch import nn

import run as entry
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from utils.task_role_spec import task_graph_packet
from learning.multi_agent.stage2_network_builder import Stage2Builder, transfer_stage1
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd


def main():
    torch.set_num_threads(1)
    args = get_args()
    args.test = True
    args.eval = os.environ.get("STAGE2_TEMPLATE_RSI") != "1"
    if not args.task_graph:
        args.task_graph = "random_scenario"
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'] = args.motion_file
    cfg['args'] = args
    cfg["env"]["eval"]["skill"] = list(cfg["env"]["skill"])
    cfg["env"]["eval"]["skillInitProb"] = [1.] + [0.]*7
    set_seed(42)
    if os.environ.get('STAGE2_SMALL_SOLO_FIXTURE') == '1':
        import env.tasks.multi_agent.humanoid_ma_stage2 as stage2_env
        original_sizes = stage2_env.sample_sizes
        def fixture_sizes(*values, **options):
            sizes = original_sizes(*values, **options)
            sizes[:, :4] = .2
            return sizes
        stage2_env.sample_sizes = fixture_sizes
    task, env = parse_task(args, cfg, train['params']['config'], parse_sim_params(args, cfg, train))
    builder = Stage2Builder()
    builder.load(train['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(task.get_obs_size(),),
        amp_input_shape=(task.get_num_amp_obs(),), value_size=1, num_agents=4, num_objects=8,
        observation_mode='clean_scene', scene_entity_sizes=task.get_scene_entity_sizes(),
        scene_kinematic_size=7, scene_arena_scale=task.get_scene_arena_scale(),
        relation_reward_mode=task._relation_cfg['mode'], relation_graph_spec=task._relation_graph_spec,
        device=task.device).to(task.device)
    model = nn.Module()
    model.a2c_network = net
    checkpoint = torch.load(cfg['experiment']['transfer']['checkpoint'], map_location=task.device, weights_only=False)
    report = transfer_stage1(model, checkpoint)
    rms = SceneRunningMeanStd([223, 30, 1], [4, 8, 4], normalized_sizes=[223, 30, 0],
                              kinematic_size=7, extra_passthrough_size=20, num_views=4).to(task.device)
    rms.load_state_dict(checkpoint['running_mean_std'], strict=True)
    rms.eval()
    results = []
    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    presets = os.environ.get('STAGE2_PRESETS', 'all_solo,sit,climb,two_pairs_at,two_pairs_ontop,one_pair_at,one_pair_ontop,random_scenario').split(',')
    skills = os.environ.get('STAGE2_SKILL', 'loco').split(',')
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for skill in skills:
            task._skill_init_prob.zero_()
            task._skill_init_prob[task._skill.index(skill)] = 1.
            for preset in presets:
                task._task_graph_preset = preset
                task.reset_buf[:] = 1
                obs = env.reset()
                packet = task_graph_packet(task.relation_runtime.graph, task.num_envs)
                demo = task.fetch_amp_obs_demo(256)
                if demo.shape != (256, task.get_num_amp_obs()) or not torch.isfinite(demo).all():
                    raise RuntimeError("Invalid conditioned AMP reference batch")
                amp_sources = task._last_amp_source_counts.cpu().tolist()
                initial_root = task._humanoid_root_states.clone()
                initial_frames = task._joint_frames.clone()
                initial_skills = task._joint_skills.clone()
                tasks = packet.reshape(task.num_envs, 4, 5)
                if 'putDown' in task._skill:
                    forbidden = task._joint & (tasks[..., 1] == 3) & (initial_skills == task._skill.index('putDown'))
                    if forbidden.any():
                        raise RuntimeError('ON_TOP initialized with joint putDown RSI')
                    solo_putdown = task._reset_ref_slots.get('putDown')
                    if solo_putdown is not None and (tasks[solo_putdown[0], solo_putdown[1], 1] == 3).any():
                        raise RuntimeError('ON_TOP initialized with solo putDown RSI')
                initial_phi, initial_diag = task._evaluate_ontop_context(torch.arange(task.num_envs, device=task.device))
                graph = task.relation_runtime.graph
                carry_edges = (graph.edge_relation == 6) & task._joint.gather(1, graph.edge_owner)
                if 'carryWith' in task._skill:
                    carry_edges &= initial_skills.gather(1, graph.edge_owner) == task._skill.index('carryWith')
                else:
                    carry_edges[:] = False
                minimum_carry_phi = float(initial_phi[carry_edges].min()) if carry_edges.any() else None
                if minimum_carry_phi is not None and minimum_carry_phi < .9:
                    raise RuntimeError('Joint carry reset does not match its assigned grasp anchors')
                if skill != "loco":
                    for e in range(task.num_envs):
                        for leader in task._groups[e].unique():
                            members = (task._groups[e] == leader).nonzero().flatten()
                            if len(members) == 2 and not (initial_frames[e, members] == initial_frames[e, members[0]]).all():
                                raise RuntimeError("Paired RSI frame mismatch")
                for _ in range(int(os.environ.get('STAGE2_STEPS', '16'))):
                    mu, _ = net.eval_actor(rms(obs))
                    value = net.eval_critic(rms(obs))
                    if not (torch.isfinite(mu).all() and torch.isfinite(value).all() and torch.isfinite(obs).all()):
                        raise RuntimeError('Non-finite Stage2 output')
                    obs, reward, done, info = env.step(mu)
                untouched = task._root_states.view(task.num_envs, -1, 13)[1:].clone()
                task._reset_envs(torch.tensor([0], device=task.device))
                if not torch.equal(untouched, task._root_states.view(task.num_envs, -1, 13)[1:]):
                    raise RuntimeError('Partial reset changed another environment')
                results.append(dict(preset=preset, skill=skill if args.eval else "template", amp_sources=amp_sources, packet=packet.cpu().tolist(),
                                    reward_mean=float(reward.mean()), finite=True, initial_joint_frames=initial_frames.cpu().tolist(),
                                    initial_joint_skills=initial_skills.cpu().tolist(), initial_phi=initial_phi.cpu().tolist(),
                                    minimum_initial_joint_carry_phi=minimum_carry_phi))
    if any(not torch.equal(value, before[name]) for name, value in model.state_dict().items()):
        raise RuntimeError('Validation changed model weights')
    report['physics'] = results
    report['physical_sizes'] = task._box_size.cpu().tolist()
    report['training_executed'] = False
    report['template_rsi'] = not args.eval
    report['small_solo_fixture'] = os.environ.get('STAGE2_SMALL_SOLO_FIXTURE') == '1'
    import copy, yaml
    from utils.relation_task_spec import checkpoint_metadata
    from utils.edge_context_spec import task_instance
    from env.tasks.multi_agent.collision_reward import agent_collision_config
    from joint_carry.artifacts import contract_digest
    weights = {name: checkpoint[name] for name in ("reward_mean_std", "amp_input_mean_std")}
    weights.update(model=model.state_dict(), running_mean_std=rms.state_dict(),
        relation_metadata=checkpoint_metadata(task._relation_cfg),
        agent_collision_config=agent_collision_config(task.cfg["env"]),
        relation_task_instance=task_instance(task._relation_graph_spec, 4, 8),
        stage2_contract=copy.deepcopy(task.cfg["env"]["stage2"]), stage2_rsi_contract=contract_digest(),
        validation_only=True, epoch=0, frame=0)
    snapshot = dict(env=copy.deepcopy(task.cfg["env"]), train=train, sim=task.cfg["sim"],
                    experiment=task.cfg["experiment"], num_agents=4, num_objects=8, num_envs=task.num_envs)
    weights["relation_experiment_config"] = snapshot
    (output/"nn").mkdir(exist_ok=True)
    torch.save(weights, output/"nn/Stage2Transfer.pth")
    (output/"relation_config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False))
    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    (output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Stage2 verification complete:', output/'validation.json')


if __name__ == '__main__':
    main()
