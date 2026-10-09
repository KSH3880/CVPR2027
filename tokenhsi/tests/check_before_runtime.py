import json
import math
from pathlib import Path

from isaacgym import gymtorch
import torch

from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from utils.edge_stage2_spec import classify_family
from utils.task_role_spec import task_graph_packet
from utils.unified_training import amp_family_ids, sample_family_matched
from utils.size_rsi import screen_states


args = get_args()
cfg, train, _ = load_cfg(args)
set_seed(42, False)
cfg['env']['motion_file'] = args.motion_file
task, env = parse_task(args, cfg, train, parse_sim_params(args, cfg, train))
env.reset()
n = task.num_envs
graph = task.relation_runtime.graph
family = classify_family(graph)
report = {'environment_families': torch.bincount(family, minlength=4).tolist()}
report['initial_agent_success_by_family'] = {str(f): task.relation_runtime.done[family == f].sum(0).tolist() for f in range(4)}
assert torch.equal(family, task._before_families)
packet = task_graph_packet(graph, n).reshape(n, 2, 5)
assert (packet[family > 0, 1, 4] == packet[family > 0, 0, 3]).all()
assert (graph.prereq_mask.any(-1).sum(-1) == (family > 0)).all()
report['rsi_requested'] = task._size_rsi_requested.tolist()
report['rsi_executed'] = task._size_rsi_executed.tolist()
report['rsi_fallbacks'] = task._size_rsi_fallbacks.tolist()
skills = torch.full((n, 2), -1, device=task.device, dtype=torch.long)
for name, slots in task._reset_ref_slots.items():
    skills[slots] = task._skill.index(name)
assert (skills >= 0).all()
assert torch.equal(skills, task._before_reset_skills)
report['rsi_by_family'] = {str(f): {str(a): torch.bincount(skills[family == f, a], minlength=len(task._skill)).tolist()
    for a in (0, 1)} for f in range(4)}
labels = task._amp_obs_buf[..., -3:].argmax(-1)
expected = task._amp_family().reshape(n, 2)
assert torch.equal(labels, expected[:, :, None].expand_as(labels))
demo = task.fetch_amp_obs_demo(4096)
rollout = task._amp_obs_buf.flatten(2).reshape(2*n, -1)
matched = sample_family_matched(demo, rollout, task._num_amp_obs_steps)
assert torch.equal(amp_family_ids(matched, task._num_amp_obs_steps), expected.flatten())
report['amp_label_matching'] = True
report['amp_reference_history_varies'] = {
    name: int((task._hist_amp_obs_buf[slots][:, 1:] != task._hist_amp_obs_buf[slots][:, :-1]).any(-1).any(-1).sum())
    for name, slots in task._reset_ref_slots.items()}

initial_root = task._humanoid_root_states.clone()
initial_box = task._box_states.clone()
targets = task._dof_pos.contiguous().clone()
ok = torch.ones(n, 2, dtype=torch.bool, device=task.device)
maxima = torch.zeros(5, device=task.device)
for _ in range(math.ceil(.1/task.sim_params.dt)):
    task.gym.set_dof_position_target_tensor(task.sim, gymtorch.unwrap_tensor(targets))
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    task._refresh_sim_tensors()
    for a in (0, 1):
        ok[:, a] &= screen_states(task._humanoid_root_states[:, a], task._box_states,
                                  initial_root[:, a], initial_box)
    metrics = torch.stack(((task._humanoid_root_states[..., 7:10]-initial_root[..., 7:10]).norm(dim=-1).max(),
        (task._humanoid_root_states[..., :3]-initial_root[..., :3]).norm(dim=-1).max(),
        task._box_states[..., 7:10].norm(dim=-1).max(),
        (task._box_states[..., :3]-initial_box[..., :3]).norm(dim=-1).max(),
        task._humanoid_root_states[..., 7:10].norm(dim=-1).max()))
    maxima = torch.maximum(maxima, metrics)
report['physics_passed'] = int(ok.all(-1).sum())
report['physics_all_passed'] = bool(ok.all())
report['physics_maxima_root_dv_root_dx_box_v_box_dx_root_v'] = maxima.tolist()
report['physics_failures_by_family_skill_pair'] = {}
failed = (~ok.all(-1)).nonzero().flatten()
for row in failed.tolist():
    key = '{}:{}/{}'.format(int(family[row]), task._skill[int(skills[row, 0])], task._skill[int(skills[row, 1])])
    report['physics_failures_by_family_skill_pair'][key] = report['physics_failures_by_family_skill_pair'].get(key, 0)+1
report['physics_failed_envs'] = failed.tolist()

checks = []
for ids in [torch.arange(0, n, 2, device=task.device)] + [(family == f).nonzero().flatten()[:32] for f in range(4)]:
    unchanged = torch.ones(n, device=task.device, dtype=torch.bool)
    unchanged[ids] = False
    arrays = [task._root_states.view(n, -1, 13), task._dof_state.view(n, -1, 2),
              task.obs_buf, task._amp_obs_buf, task._before_amp_rotation, task._before_amp_translation]
    old = [a[unchanged].clone() for a in arrays]
    task._reset_envs(ids)
    for before, after in zip(old, arrays):
        torch.testing.assert_close(before, after[unchanged], rtol=0, atol=0)
    assert torch.equal(classify_family(task.relation_runtime.graph), family)
    assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
    checks.append(len(ids))
    task.gym.set_dof_position_target_tensor(task.sim, gymtorch.unwrap_tensor(task._dof_pos.contiguous()))
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    task._refresh_sim_tensors()
report['partial_reset_preserved_counts'] = checks
for _ in range(8):
    task.step(torch.zeros(n*2, task.num_actions, device=task.device))
    assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
report['finite_control_steps'] = 8
output = Path(args.output_path)/'reset_check.json'
output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
