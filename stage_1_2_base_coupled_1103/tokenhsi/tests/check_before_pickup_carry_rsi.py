import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
from torch import nn
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from utils.before_spec import check_checkpoint
from learning.multi_agent.before_network import BeforeBuilder
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from env.tasks.multi_agent.edge_ontop_reward import vertical_extent
from check_before import check_at_progress


def inspect(t):
    skills = t._before_rsi_skill
    eligible = (t._before_predecessor >= 0) & (t._templates == 4)
    assert not (skills[~eligible] > 0).any()
    assert torch.equal(t._before_carry_mask, skills > 0)
    for j, name in enumerate(('pickUp', 'carryWith'), 1):
        rows, owners = (skills == j).nonzero(as_tuple=True)
        if not len(rows):
            continue
        es, hs = t._reset_ref_slots[name]
        clips, times = t._reset_ref_motion_ids[name], t._reset_ref_motion_times[name]
        lib = t._motion_lib[name]
        body = t._kinematic_humanoid_rigid_body_states[es, hs, :, :3]
        box = t._box_states[es, hs]
        ref_body = lib.get_motion_state_max(clips, times)[0]
        ref_box = lib.get_obj_motion_state(clips, times)[0]
        torch.testing.assert_close((body-box[:, None, :3]).norm(dim=-1),
            (ref_body-ref_box[:, None]).norm(dim=-1), rtol=1e-4, atol=3e-5)
        if name == 'carryWith':
            assert (((body[:, t._key_body_ids[:2]].mean(1)-box[:, :3]).square().sum(-1)*-10).exp() >= .9-1e-5).all()
            assert (box[:, 2]-vertical_extent(box[:, 3:7], t._box_size[es, hs]/2) > .08-1e-5).all()
    distances = {}
    if t.cfg['env']['before'].get('distance_mixture', {}).get('mode') == 'relative_sum':
        from utils.before_relative_sampling import pair_distances
        es, followers = (t._before_predecessor >= 0).nonzero(as_tuple=True)
        leaders = t._before_predecessor[es, followers]
        first, second = pair_distances(t._humanoid_root_states[es, leaders, :2],
            t._box_states[es, leaders, :2], t._humanoid_root_states[es, followers, :2],
            t._box_states[es, followers, :2], t._templates[es, followers])
        near = t._before_near_pairs[es, followers]
        assert (second[near] < first[near]).all()
        assert ((first >= 1.-2e-5) & (first <= 10.+2e-5)).all()
        assert not t._before_near_pairs[t._before_predecessor < 0].any()
        distances = dict(near_pairs=int(near.sum()), original_pairs=int((~near).sum()),
            near_min_margin=float((first-second)[near].min()) if near.any() else None)
    return dict(eligible=int(eligible.sum()), distances=distances,
        counts=torch.bincount(skills[eligible], minlength=3).tolist(),
        progress=check_at_progress(t))


def main(inspector=inspect, presets=None, step_inspector=None):
    torch.set_num_threads(1)
    args = get_args(); args.test = args.eval = True
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'], cfg['args'] = args.motion_file, args
    set_seed(42)
    task, env = parse_task(args, cfg, train['params']['config'], parse_sim_params(args, cfg, train))
    builder_class = BeforeBuilder
    if cfg["env"].get("jointPush") is not None:
        from learning.multi_agent.before_push_network import BeforePushBuilder
        builder_class = BeforePushBuilder
    builder = builder_class(); builder.load(train['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(task.get_obs_size(),),
        amp_input_shape=(task.get_num_amp_obs(),), value_size=1, num_agents=task.num_agents,
        num_objects=task.num_objects, observation_mode='clean_scene',
        scene_entity_sizes=task.get_scene_entity_sizes(), scene_kinematic_size=7,
        scene_arena_scale=task.get_scene_arena_scale(), relation_reward_mode=task._relation_cfg['mode'],
        relation_graph_spec=task._relation_graph_spec, device=task.device).to(task.device)
    model = nn.Module(); model.a2c_network = net
    checkpoint = torch.load(args.checkpoint, map_location=task.device, weights_only=False)
    check_checkpoint(checkpoint, task.cfg['env'])
    model.load_state_dict(checkpoint['model'], strict=True); model.eval()
    rms = SceneRunningMeanStd([223, 30, 1], [task.num_agents, task.num_objects, task.num_agents],
        normalized_sizes=[223, 30, 0], kinematic_size=7,
        extra_passthrough_size=task.get_relation_suffix_size()).to(task.device)
    rms.load_state_dict(checkpoint['running_mean_std'], strict=True); rms.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    results = []
    with torch.no_grad():
        for preset, mode in (presets if presets is not None else [
                ('random_scenario', 'loco'), ('random_scenario', 'mix'),
                ('before_stack', 'pickup'), ('before_stack', 'carry'), ('all_solo', 'mix')]):
            task._task_graph_preset = preset; args.before_pickup_carry_rsi = mode
            obs = env.reset(); layout = inspector(task)
            if mode == 'loco' or preset == 'all_solo':
                assert not task._before_rsi_skill.any()
            assert torch.isfinite(task.fetch_amp_obs_demo(128)).all()
            for _ in range(16):
                mu, _ = net.eval_actor(rms(obs)); value = net.eval_critic(rms(obs))
                obs, reward, done, info = env.step(mu)
                assert all(torch.isfinite(x).all() for x in (mu, value, obs, reward))
                if step_inspector is not None:
                    step_inspector(task, obs)
            untouched = task._root_states.view(task.num_envs, -1, 13)[1:].clone()
            saved = task._before_rsi_skill[1:].clone()
            saved_mapping = task._logical_box_order[1:].clone()
            task._reset_envs(torch.tensor([0], device=task.device))
            assert torch.equal(untouched, task._root_states.view(task.num_envs, -1, 13)[1:])
            assert torch.equal(saved, task._before_rsi_skill[1:])
            assert torch.equal(saved_mapping, task._logical_box_order[1:])
            results.append(dict(preset=preset, mode=mode, finite=True, partial_reset=True, **layout))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    p = Path(args.output_path)/'physics_validation.json'
    p.write_text(json.dumps(dict(weights_unchanged=True, cases=results), indent=2)+'\n')
    print('PICKUP/CARRY RSI CHECK COMPLETE', p, flush=True)


if __name__ == '__main__':
    main()
