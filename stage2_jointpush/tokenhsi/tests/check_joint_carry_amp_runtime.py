import json
from pathlib import Path

from isaacgym import gymapi
import torch
import yaml

from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from utils.unified_training import amp_family_ids


args = get_args()
cfg, train, _ = load_cfg(args)
set_seed(42, False)
cfg['env']['motion_file'] = args.motion_file
task, env = parse_task(args, cfg, train, parse_sim_params(args, cfg, train))
env.reset()
n = task.num_envs
report = {'environments': n, 'joint':int(task._joint_env_mask.sum()),
          'independent':int((~task._joint_env_mask).sum())}
assert report['joint'] == n-round(n*.2)
assert set(task._motion_lib) == set(yaml.safe_load(Path(args.motion_file).read_text())['motions'])
assert set(task._amp_locomotion_libs) == {'backward','sideways'}
assert task._amp_locomotion_libs['backward'].num_motions() == 9
assert task._amp_locomotion_libs['sideways'].num_motions() == 3
for lib in task._motion_lib.values():
    assert all('teamhoi_retarget' not in path for path in lib._motion_files)
report['rsi_library_isolation'] = True
libraries = {id(lib):name for name,lib in dict(task._motion_lib, **task._amp_locomotion_libs).items()}
sampled = {}
original_builder = task.build_amp_obs_demo


def capture(ids, times, library):
    name = libraries[id(library)]
    sampled[name] = {'samples':len(ids), 'per_clip':torch.bincount(ids, minlength=library.num_motions()).tolist()}
    assert (times >= task.dt*(task._num_amp_obs_steps-1)-1e-6).all()
    assert (times <= library._motion_lengths[ids]+1e-6).all()
    return original_builder(ids, times, library)


task.build_amp_obs_demo = capture
demo = task.fetch_amp_obs_demo(30000)
task.build_amp_obs_demo = original_builder
assert demo.shape == (30000,1320) and torch.isfinite(demo).all()
assert amp_family_ids(demo, 10).count_nonzero() == 0
source_fraction = task._last_amp_source_counts.float()/30000
torch.testing.assert_close(source_fraction, torch.tensor([.8,.1,.1], device=task.device), atol=.01, rtol=0)
report['amp_source_fraction'] = source_fraction.tolist()
report['amp_samples'] = sampled
report['amp_shapes_and_labels'] = True
assert torch.equal(task._hist_amp_obs_buf[task._joint_env_mask],
                   task._curr_amp_obs_buf[task._joint_env_mask,:,None].expand_as(task._hist_amp_obs_buf[task._joint_env_mask]))
report['joint_reset_amp_history_repeats_state'] = True
report['independent_reference_history_varies'] = bool((task._hist_amp_obs_buf[~task._joint_env_mask,:,1:]
    != task._hist_amp_obs_buf[~task._joint_env_mask,:,:-1]).any())
checks = []
for ids in (torch.arange(0,n,2,device=task.device),
            task._joint_env_mask.nonzero().flatten()[:32], (~task._joint_env_mask).nonzero().flatten()[:32]):
    remaining = torch.ones(n,device=task.device,dtype=torch.bool);remaining[ids] = False
    arrays = [task._root_states.view(n,-1,13),task._dof_state.view(n,-1,2),task.obs_buf,task._amp_obs_buf]
    old = [values[remaining].clone() for values in arrays]
    task._reset_envs(ids)
    for before, values in zip(old,arrays):
        torch.testing.assert_close(before,values[remaining],rtol=0,atol=0)
    assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
    checks.append(len(ids))
report['partial_reset_preserved_counts'] = checks
for _ in range(8):
    task.step(torch.zeros(n*2,task.num_actions,device=task.device))
    assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
report['finite_steps'] = 8
out = Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
(out/'reset_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
