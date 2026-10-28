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
from utils.shared_support import checkpoint_contract
from learning.multi_agent.shared_support_network import SharedSupportBuilder
from learning.multi_agent.stage2_network_builder import transfer_stage1
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd


def check_layout(task):
    mapping = task._logical_box_order
    graph = task.relation_runtime.graph
    m, o = task.num_agents, task.num_objects
    sizes = task._logical_box_values(task._box_size)
    for env in range(task.num_envs):
        count = int(task._task_token_count[env])
        distractors = mapping[env, count:]
        assert len(distractors.unique()) == o-count
        assert not torch.isin(distractors, mapping[env,:count]).any()
        assert (task._token_region[env,count:] == -1).all()
        xy = task._box_states[env,distractors,:2]-task._env_origins[env,:2]
        assert (xy.norm(dim=-1) <= 4.5001).all()
        for physical in task._support_physical[env,task._shared[env]].unique():
            tokens = mapping[env] == physical
            area = sizes[env,tokens,:2].prod(-1).sum()
            torch.testing.assert_close(area, task._box_size[env,physical,:2].prod())
        for edge in graph.edge_valid[env].nonzero().flatten():
            owner = int(graph.edge_owner[env,edge])
            relation = int(graph.edge_relation[env,edge])
            src, dst = int(graph.edge_src[env,edge]), int(graph.edge_dst[env,edge])
            if relation == 6:
                assert src == owner and mapping[env,dst-m] == owner
            elif relation == 7:
                assert mapping[env,src-m] == owner and dst == m+o+owner
            else:
                assert mapping[env,dst-m] == task._support_physical[env,owner]
                if relation == 8:
                    assert mapping[env,src-m] == owner
                else:
                    assert src == owner
                if task._shared[env,owner]:
                    assert task._token_region[env,dst-m] >= 0
    return dict(task_tokens=task._task_token_count.tolist(),
                distractors=(o-task._task_token_count).tolist(),
                physical_active=task._active_physical(torch.arange(task.num_envs,device=task.device)).sum(-1).tolist())


def main():
    torch.set_num_threads(1)
    args = get_args()
    args.test, args.eval = True, os.environ.get('SHARED_TRAIN_INIT') != '1'
    args.task_graph = 'random_scenario'
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'], cfg['args'] = args.motion_file, args
    set_seed(42)
    task, env = parse_task(args, cfg, train['params']['config'], parse_sim_params(args,cfg,train))
    builder = SharedSupportBuilder(); builder.load(train['params']['network'])
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
    presets = os.environ.get('SHARED_PRESETS', 'all_solo,sit,climb,holding_at,holding_ontop,shared_sit,shared_climb,shared_ontop,all_shared,random_scenario').split(',')
    with torch.no_grad():
        for preset in presets:
            task._task_graph_preset = preset
            task.reset_buf[:] = 1
            obs = env.reset()
            layout = check_layout(task)
            demo = task.fetch_amp_obs_demo(128)
            if not torch.isfinite(demo).all():
                raise RuntimeError('Non-finite AMP demos')
            groups = task._groups.cpu().tolist()
            templates = task._templates.cpu().tolist()
            for _ in range(int(os.environ.get('SHARED_STEPS','16'))):
                mu,_ = net.eval_actor(rms(obs)); value = net.eval_critic(rms(obs))
                obs, reward, done, info = env.step(mu)
                if not all(torch.isfinite(x).all() for x in (mu,value,obs,reward)):
                    raise RuntimeError('Non-finite shared support rollout')
            untouched = task._root_states.view(task.num_envs,-1,13)[1:].clone()
            task._reset_envs(torch.tensor([0],device=task.device))
            if not torch.equal(untouched,task._root_states.view(task.num_envs,-1,13)[1:]):
                raise RuntimeError('Partial reset changed another environment')
            results.append(dict(preset=preset,groups=groups,templates=templates,reward=float(reward.mean()),finite=True,**layout))
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
        shared_support_contract=checkpoint_contract(task.cfg['env']),validation_only=True,epoch=0,frame=0,
        relation_experiment_config=snapshot)
    torch.save(weights,output/'nn/SharedSupportTransfer.pth')
    (output/'relation_config.yaml').write_text(yaml.safe_dump(snapshot,sort_keys=False))
    report.update(physics=results,training_executed=False,train_init=not args.eval,
                  agents=task.num_agents,physical_objects=task.num_physical_objects,token_capacity=task.num_objects)
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Shared support verification complete:',output/'validation.json')


if __name__ == '__main__':
    main()
