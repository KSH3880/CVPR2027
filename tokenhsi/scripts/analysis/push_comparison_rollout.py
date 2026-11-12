"""Collect actual policy trajectories for matched two-person PUSH captures."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tokenhsi'))
import isaacgym
import numpy as np
import torch
from learning.multi_agent import viewer_diagnostics

SOURCE = Path(os.environ.get('COMPARE_SOURCE_ROOT', str(ROOT/'output_etc/check/push_wait_v4_seed0_5_20261112')))


@torch.no_grad()
def collect(player):
    task = player.env.task
    args = task.cfg['args']
    method = os.environ['COMPARE_METHOD']
    out = Path(args.output_path).resolve()
    out.relative_to(ROOT/'output_etc/check')
    out.mkdir(parents=True, exist_ok=True)
    assert task.num_agents == 2 and task.viewer is None
    obs = player.env_reset()
    player.get_batch_size(obs['obs'], 1)
    initial = torch.load(SOURCE/f'seed{args.seed}/initial.pt', map_location='cpu', weights_only=False)
    assert task.num_envs == initial['num_envs']
    assert initial.get('fixed_layout')==getattr(task,'_push_fixed_metadata',None), 'Controlled layout mismatch'
    checked = {}
    for name, value in dict(observation=obs['obs'], roots=task._humanoid_root_states,
        boxes=task._box_states, sizes=task._box_size, goals=task._tar_pos,
        center_goals=task._center_tar_pos, groups=task._groups, regions=task._regions,
        logical_box_order=task._logical_box_order, assignment=task._agent_box_assignment).items():
        torch.testing.assert_close(value.cpu(), initial[name], atol=0, rtol=0)
        checked[name] = True
    torch.testing.assert_close(task._dof_state.cpu(), initial['dof'], atol=0, rtol=0)
    checked['dof'] = True
    selection_file = os.environ.get('COMPARE_SELECTION_FILE', '')
    if selection_file:
        selection = json.loads(Path(selection_file).read_text())
        assert selection['seed'] == args.seed and selection['num_envs'] == task.num_envs
        env_ids = selection['env_ids']
        assert env_ids, 'No scene passed the preceding stage'
    elif os.environ.get('COMPARE_INCLUDE_ALL_ENVS') == '1':
        env_ids = list(range(task.num_envs))
    else:
        candidates = json.loads((SOURCE/f'seed{args.seed}/candidates.json').read_text())['candidates']
        env_ids = sorted(c['env_id'] for c in candidates if c['qualifies'] or os.environ.get('COMPARE_INCLUDE_ALL_SUCCESS')=='1')
    extra = os.environ.get('COMPARE_ENV_IDS', '')
    if extra:
        env_ids = sorted(set(env_ids) | {int(v) for v in extra.split(',')})
    (out/'initial_verification.json').write_text(json.dumps(dict(method=method, seed=args.seed,
        initial_exact=checked, num_agents=2, num_envs=task.num_envs), indent=2)+'\n')
    selected = torch.tensor(env_ids, device=task.device)
    policy = controller = None
    common = Path('/home/hwanhee/ksh/stage1_2_base_teacher')
    sys.path[:0] = [str(common/'eval'), str(common/'FSM')]
    from table2_metrics import surface_distance
    if method == 'tokenhsi':
        baseline = Path('/home/hwanhee/ksh/tokenhsi_push')
        sys.path[:0] = [str(baseline/'scripts'), str(baseline)]
        from tokenhsi_table_policy import TokenHSIPolicy, project_scene
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            policy = TokenHSIPolicy(baseline/'checkpoints/TokenHSIPushKD_00040500.pth', task.device)
    elif method == 'fsm':
        from stage1_protocol_policy import Stage1Policy
        from controller import StateFreeze
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            policy = Stage1Policy(player, common/'pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth', obs['obs'])
        controller = StateFreeze(task)
    else:
        assert method == 'ours'
    # Do not reset or hide physical distractors: retain the exact search scene.
    task._relation_cfg['diagnostics']['enabled'] = False
    task._reward_debug = False
    player._amp_debug = lambda info: None
    alive = torch.ones(task.num_envs, 2, dtype=torch.bool, device=task.device)
    physical = task._agent_box_assignment[selected, 0]
    trace = {key: [] for key in ('roots', 'dof', 'boxes', 'on_face', 'surface_distance',
        'force_contact', 'hand_force_xy_norm', 'box_contact_xy_norm', 'alive', 'frozen', 'released', 'arrived')}
    if os.environ.get('COMPARE_SAVE_DOF', '1') == '0':
        del trace['dof']
    ids = torch.arange(task.num_envs, device=task.device)

    def capture():
        *_, face = task._contact_geometry(ids, task._rigid_body_pos)
        box = task._box_states[selected, physical]
        hands = task._rigid_body_pos[selected][:, :, task._key_body_ids[:2]]
        distance = surface_distance(hands.double(), box[:, :7].double(),
                                    task._box_size[selected, physical].double())
        false = torch.zeros(len(selected), 2, dtype=torch.bool, device=task.device)
        values = dict(roots=task._humanoid_root_states[selected],
            dof=task._dof_state.view(task.num_envs, -1, 2)[selected], boxes=box,
            on_face=face['on_face'][selected], surface_distance=distance,
            force_contact=task._push_hits[selected], alive=alive[selected],
            hand_force_xy_norm=task._contact_forces[selected][:,:,task._key_body_ids[:2],:2].norm(dim=-1),
            box_contact_xy_norm=task._assigned_box_values(task._ontop_box_contact)[selected,0,:2].norm(dim=-1),
            frozen=controller.frozen[selected] if controller else false,
            released=controller.state.released[selected] if controller else false,
            arrived=controller.state.arrived[selected] if controller else false)
        for key, value in values.items():
            if key in trace:
                trace[key].append(value.cpu().clone())

    capture()
    start = time.time()
    for step in range(1, 901):
        if controller is not None and controller.update():
            task._compute_observations()
            obs = player.env_reset(torch.empty(0, device=task.device, dtype=torch.long))
        if method == 'tokenhsi':
            action = policy.action(project_scene(obs['obs'], 2, task.num_objects))
        else:
            action = player.get_action(obs, True)
        if controller is not None:
            action = action.reshape(task.num_envs, 2, -1).masked_fill(controller.frozen[..., None], 0).flatten(0, 1)
        obs, _, _, _ = player.env_step(player.env, action)
        if not isinstance(obs, dict):
            obs = {'obs': obs}
        alive &= ~task._terminate_buf.reshape(task.num_envs, 2).bool()
        if controller is not None:
            controller.frame += 1
            controller.score_alive = controller.alive.clone()
            controller.alive &= ~task._terminate_buf.reshape(task.num_envs, 2).bool()
        capture()
        if step % 100 == 0:
            report = dict(method=method, seed=args.seed, step=step, horizon=900,
                selected_envs=len(selected), selected_alive=int(alive[selected].all(-1).sum()), elapsed=time.time()-start)
            (out/'progress.json').write_text(json.dumps(report, indent=2))
            print('[comparison]', report, flush=True)
    saved = {key: torch.stack(value) for key, value in trace.items()}
    for key in ('roots', 'boxes', 'surface_distance'):
        assert torch.isfinite(saved[key]).all(), key
    proof = policy.finish() if policy else dict(checkpoint=str(args.checkpoint),
        sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
    frozen_proof = controller.finish() if controller else None
    if frozen_proof:
        assert frozen_proof['max_frozen_root_error'] == frozen_proof['max_frozen_dof_error'] == 0
    verification = dict(method=method, seed=args.seed, env_ids=env_ids, horizon=900,
        initial_exact=checked, policy=proof, fsm=frozen_proof, source_initial=str(SOURCE/f'seed{args.seed}/initial.pt'),
        num_agents=2, num_envs=task.num_envs, distractors='Original physics retained; hidden only in snapshot rendering',
        dof_saved='dof' in saved,
        fixed_layout=initial.get('fixed_layout'),
        elapsed_seconds=time.time()-start)
    torch.save(dict(trajectory=saved, env_ids=env_ids, verification=verification), out/'rollout.pt')
    (out/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print('[comparison complete]', method, args.seed, flush=True)


viewer_diagnostics.run_viewer_diagnostics = collect
if __name__ == '__main__':
    runpy.run_path(str(ROOT/'tokenhsi/run.py'), run_name='__main__')
