import copy
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaacgym
import torch
from torch import nn
import yaml

from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from utils.before_spec import checkpoint_contract
from learning.multi_agent.before_network import BeforeBuilder
from learning.multi_agent.stage2_network_builder import transfer_stage1
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd


def check_at_progress(task, env_ids=None):
    if task.cfg['env']['before'].get('progress_target') != 'predecessor_at_goal':
        return {}
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    from utils.edge_ontop_spec import select_graph
    ids = torch.arange(task.num_envs, device=task.device) if env_ids is None else env_ids
    phi, diag = task._evaluate_ontop_context(env_ids)
    native_phi, native = HumanoidMACarry._evaluate_ontop_context(task, env_ids)
    torch.testing.assert_close(phi, native_phi, rtol=0, atol=0)
    for key in native:
        if key != 'progress':
            torch.testing.assert_close(diag[key], native[key], rtol=0, atol=0)
    graph = select_graph(task.relation_runtime.graph, ids)
    dependent = graph.prereq_mask.any(-1)
    assert torch.equal(diag['progress'][~dependent], native['progress'][~dependent])
    rows, edges = dependent.nonzero(as_tuple=True)
    es, follower = ids[rows], graph.edge_owner[rows, edges]
    leader = task._before_predecessor[es, follower]
    assert (leader >= 0).all()
    goal = task._scenario_goal(es, leader)
    bodies = (task._rigid_body_pos if env_ids is None
              else task._kinematic_humanoid_rigid_body_states[ids, ..., :3])
    root = bodies[rows, follower, 0, :2]
    source = task._scenario_physical_object(es, follower)
    top = task._templates[es, follower] == 4
    point = torch.where(top[:, None], task._box_states[es, source, :2], root)
    distance = (point-task._tar_pos[es, goal, :2]).norm(dim=-1)
    support = task._scenario_physical_object(es, leader)
    radius = task._box_size[es, support, :2].norm(dim=-1)/2 + torch.where(top, .1, .3)
    expected = 1/(1+(distance-radius).clamp_min(0))
    torch.testing.assert_close(diag['progress'][rows, edges], expected)
    return dict(dependent_edges=len(rows), progress_goal_verified=True,
                state_success_geometry_unchanged=True, other_progress_unchanged=True)


def check_layout(task):
    from utils.task_role_spec import task_graph_packet
    from learning.multi_agent.human_coordination import human_relations
    m, o = task.num_agents, task.num_objects
    mapping = task._logical_box_order
    packet = task_graph_packet(task.relation_runtime.graph, task.num_envs).reshape(-1, m, 5).long()
    relations = human_relations(packet, m, before=True)
    if task._ground_only:
        active = task._active_physical(torch.arange(task.num_envs, device=task.device))
        torch.testing.assert_close(task._box_states[..., 2][active], task._box_size[..., 2][active]/2)
        assert (task._platform_pos[..., 2] < -10).all()
        assert (task._tar_platform_pos[..., 2] < -10).all()
        rows, owner = (task._templates == 3).nonzero(as_tuple=True)
        goal = task._scenario_goal(rows, owner)
        payload = task._scenario_physical_object(rows, owner)
        torch.testing.assert_close(task._tar_pos[rows, goal, 2], task._box_size[rows, payload, 2]/2)
    pairs = 0
    distances = []
    for env in range(task.num_envs):
        assert len(mapping[env].unique()) == o
        count = int(task._task_token_count[env])
        xy = task._box_states[env,mapping[env,count:],:2]-task._env_origins[env,:2]
        assert (xy.norm(dim=-1) <= 4.5001).all()
        for human in range(m):
            pre = int(task._before_predecessor[env,human])
            template = int(task._templates[env,human])
            target = int(packet[env,human,4])
            if template >= 3:
                assert mapping[env,packet[env,human,3]-m] == human
            if pre >= 0:
                pairs += 1
                assert mapping[env,target-m] == pre
                assert relations[env,human,pre] == 2 and relations[env,pre,human] == 0
                if (task.cfg['args'].before_follower_near
                        and task.cfg['env']['before'].get('distance_mixture') is None):
                    distance = (task._humanoid_root_states[env,[pre,human],:2]-task._box_states[env,pre,:2]).norm(dim=-1)
                    assert 1.-1e-5 <= distance[1] < distance[0] <= 5.+1e-5
                    if template == 4:
                        payload_distance = (task._box_states[env,human,:2]
                                            - task._humanoid_root_states[env,human,:2]).norm()
                        assert 1.-1e-5 <= payload_distance <= 3.+1e-5
                elif task.cfg['env']['before'].get('distance_mixture') is not None:
                    near = bool(task._before_near_pairs[env,human])
                    mode = int(task._before_distance_modes[env,human])
                    if getattr(task.cfg['args'], 'before_follower_short', False):
                        assert mode == 2
                    if task.cfg['args'].before_follower_near:
                        assert near
                    maximum = 5. if near or template == 1 else 10.
                    if mode == 2:
                        maximum = 2.5
                    mixture = task.cfg['env']['before']['distance_mixture']
                    if template == 4 and 'ontop_follower_distance' in mixture:
                        maximum = mixture['ontop_follower_distance'][mode][1]
                    distance = (task._humanoid_root_states[env,[pre,human],:2]
                                - task._box_states[env,pre,:2]).norm(dim=-1)
                    leader_max = 5. if mode == 2 else 10.
                    assert 1.-1e-5 <= distance[0] <= leader_max+1e-5
                    assert 1.-1e-5 <= distance[1] <= maximum+1e-5
                    goal = task._scenario_goal(torch.tensor([env], device=task.device),
                                               torch.tensor([pre], device=task.device))[0]
                    transport = (task._tar_pos[env, goal, :2]-task._box_states[env, pre, :2]).norm()
                    if 'leader_goal_distance' in mixture:
                        assert 1.-1e-5 <= transport <= mixture['leader_goal_distance'][mode][1]+1e-5
                        assert (task._tar_pos[env, goal, :2]-task._humanoid_root_states[env, pre, :2]).norm() >= 1.-1e-5
                    record = dict(template=template, mode=mode, leader=float(distance[0]),
                                  follower=float(distance[1]), transport=float(transport))
                    if template == 4:
                        own_distance = (task._box_states[env,human,:2]
                                        - task._humanoid_root_states[env,human,:2]).norm()
                        assert 1.-1e-5 <= own_distance <= maximum+1e-5
                        record['payload'] = float(own_distance)
                    distances.append(record)
            elif template < 3:
                assert mapping[env,target-m] == human+template*m
            elif template == 4:
                assert mapping[env,target-m] == 3*m+human
            if pre < 0 and not (task._before_predecessor[env] == human).any():
                # Solo goals retain the environment-centered square sampler.
                if template == 3:
                    goal = task._scenario_goal(torch.tensor([env], device=task.device),
                                               torch.tensor([human], device=task.device))[0]
                    xy = task._tar_pos[env, goal, :2]
                    assert ((xy-task._env_origins[env, :2]).abs() <= 4.50001).all()
                elif template == 4:
                    xy = task._box_states[env, mapping[env,target-m], :2]
                    assert ((xy-task._env_origins[env, :2]).abs() <= 4.50001).all()
    assert int((relations == 2).sum()) == pairs
    assert not task._before_near_pairs[task._before_predecessor < 0].any()
    return dict(pairs=pairs, near_pairs=int(task._before_near_pairs.sum()), distances=distances,
                distance_modes=task._before_distance_modes.tolist(),
                task_tokens=task._task_token_count.tolist(),
                distractors=(o-task._task_token_count).tolist())


def main():
    torch.set_num_threads(1)
    args = get_args()
    args.test, args.eval = True, os.environ.get('BEFORE_TRAIN_INIT') != '1'
    args.task_graph = 'random_scenario'
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'], cfg['args'] = args.motion_file, args
    set_seed(42)
    task, env = parse_task(args, cfg, train['params']['config'], parse_sim_params(args,cfg,train))
    builder = BeforeBuilder(); builder.load(train['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(task.get_obs_size(),),
        amp_input_shape=(task.get_num_amp_obs(),), value_size=1, num_agents=task.num_agents,
        num_objects=task.num_objects, observation_mode='clean_scene',
        scene_entity_sizes=task.get_scene_entity_sizes(), scene_kinematic_size=7,
        scene_arena_scale=task.get_scene_arena_scale(), relation_reward_mode=task._relation_cfg['mode'],
        relation_graph_spec=task._relation_graph_spec, device=task.device).to(task.device)
    model = nn.Module(); model.a2c_network = net
    checkpoint = torch.load(cfg['experiment']['transfer']['checkpoint'], map_location=task.device, weights_only=False)
    report = transfer_stage1(model, checkpoint)
    rms = SceneRunningMeanStd([223,30,1], [task.num_agents,task.num_objects,task.num_agents],
        normalized_sizes=[223,30,0], kinematic_size=7,
        extra_passthrough_size=task.get_relation_suffix_size()).to(task.device)
    rms.load_state_dict(checkpoint['running_mean_std'], strict=True); rms.eval()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    results = []
    presets = os.environ.get('BEFORE_PRESETS', 'all_solo,sit,climb,holding_at,holding_ontop,before_stack,one_pair_stack,random_scenario').split(',')
    with torch.no_grad():
        for preset in presets:
            task._task_graph_preset = preset
            task.reset_buf[:] = 1
            obs = env.reset()
            layout = check_layout(task)
            progress_check = check_at_progress(task)
            phi, diag = task._evaluate_relations()
            result = task.relation_runtime.step(phi, diag['progress'], diag['z_error'], diag['feet_height_error'], diag['region_error'])
            dependent = task.relation_runtime.graph.prereq_mask.any(-1)
            if result['prerequisite_gate'][dependent].any():
                raise RuntimeError('BEFORE gate starts open')
            demo = task.fetch_amp_obs_demo(128)
            if not torch.isfinite(demo).all():
                raise RuntimeError('Non-finite AMP demos')
            groups = task._before_predecessor.cpu().tolist()
            templates = task._templates.cpu().tolist()
            for _ in range(int(os.environ.get('BEFORE_STEPS','16'))):
                mu,_ = net.eval_actor(rms(obs)); value = net.eval_critic(rms(obs))
                obs, reward, done, info = env.step(mu)
                if not all(torch.isfinite(x).all() for x in (mu,value,obs,reward)):
                    raise RuntimeError('Non-finite BEFORE rollout')
            check_at_progress(task)
            untouched = task._root_states.view(task.num_envs,-1,13)[1:].clone()
            untouched_modes = task._before_distance_modes[1:].clone()
            task._reset_envs(torch.tensor([0],device=task.device))
            assert torch.equal(untouched_modes, task._before_distance_modes[1:])
            if getattr(args, 'before_follower_short', False):
                assert (task._before_distance_modes[task._before_predecessor >= 0] == 2).all()
            if not torch.equal(untouched,task._root_states.view(task.num_envs,-1,13)[1:]):
                raise RuntimeError('Partial reset changed another environment')
            check_at_progress(task, torch.tensor([0],device=task.device))
            results.append(dict(preset=preset,groups=groups,templates=templates,reward=float(reward.mean()),finite=True,
                                progress_check=progress_check,**layout))
    if any(not torch.equal(value,before[key]) for key,value in model.state_dict().items()):
        raise RuntimeError('Validation changed weights')
    from utils.relation_task_spec import checkpoint_metadata
    from utils.edge_context_spec import task_instance
    from env.tasks.multi_agent.collision_reward import agent_collision_config
    output = Path(args.output_path); (output/'nn').mkdir(parents=True,exist_ok=True)
    snapshot = dict(env=copy.deepcopy(task.cfg['env']),train=train,sim=task.cfg['sim'],experiment=task.cfg['experiment'])
    weights = {k:checkpoint[k] for k in ('reward_mean_std','amp_input_mean_std')}
    weights.update(model=model.state_dict(),running_mean_std=rms.state_dict(),
        relation_metadata=checkpoint_metadata(task._relation_cfg),agent_collision_config=agent_collision_config(task.cfg['env']),
        relation_task_instance=task_instance(task._relation_graph_spec,task.num_agents,task.num_objects),
        before_contract=checkpoint_contract(task.cfg['env']),validation_only=True,epoch=0,frame=0,
        relation_experiment_config=snapshot)
    torch.save(weights,output/'nn/BeforeTransfer.pth')
    (output/'relation_config.yaml').write_text(yaml.safe_dump(snapshot,sort_keys=False))
    report.update(physics=results,training_executed=False,train_init=not args.eval,
                  agents=task.num_agents,physical_objects=task.num_physical_objects,token_capacity=task.num_objects)
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('BEFORE verification complete:',output/'validation.json')


if __name__ == '__main__':
    main()
