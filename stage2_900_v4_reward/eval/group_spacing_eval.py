import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ['SPACING_MODEL']
GAP = None if os.environ.get('SPACING_METERS', 'native') == 'native' else float(os.environ['SPACING_METERS'])
RADIUS = float(os.environ['GROUP_SPAWN_RADIUS']) if 'GROUP_SPAWN_RADIUS' in os.environ else None
PLACEMENT = ('fixed_radius_group_center_ring' if RADIUS is not None else
             'native_agent_offsets_without_group_translation' if GAP is None else 'group_center_ring')
SOURCE = ROOT.parent/'sct_coopft' if MODEL == 'coopft' else ROOT
sys.path[:0] = [str(SOURCE/'tokenhsi'), str(SOURCE/'eval'), str(ROOT/'FSM'),
               str(ROOT.parent/'tokenhsi_push/scripts')]
import isaacgym
import torch
from env.tasks.multi_agent import stage2_group_layout as layout
from learning.multi_agent import formal_eval
from utils import output_paths


torch.set_num_threads(1)
resolve = output_paths.resolve_output_path


def local_output(*args, **kwargs):
    return str((ROOT/resolve(*args, **dict(kwargs, root=ROOT))).resolve())


output_paths.resolve_output_path = local_output
original_translate = layout.translate_people


def natural_centers(task, ids, radius=None):
    task.cfg['env']['stage2JointPush']['layout_attempts'] = 1024
    roots = task._humanoid_root_states[ids].clone()
    bodies = task._kinematic_humanoid_rigid_body_states[ids].clone()
    groups = task._groups[ids]
    same = groups[:, :, None] == groups[:, None, :]
    centers = torch.bmm(same.float(), roots[..., :2])/same.sum(-1, keepdim=True)
    original_translate(task, ids)
    targets = (task._env_origins[ids, None, :2]+GAP/5*(task._group_spawn_targets[ids]-task._env_origins[ids, None, :2])
               if GAP is not None else centers)
    if RADIUS is not None:
        direction = task._group_spawn_targets[ids]-task._env_origins[ids, None, :2]
        targets = task._env_origins[ids, None, :2]+RADIUS*direction/direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    task._humanoid_root_states[ids] = roots
    task._kinematic_humanoid_rigid_body_states[ids] = bodies
    task._group_spawn_targets[ids] = targets
    if GAP is not None or RADIUS is not None:
        layout.move_people(task, ids, targets-centers)
    task._group_spawn_radius = RADIUS
    task._group_spawn_gap = GAP
    task._natural_group_center_checks = getattr(task, '_natural_group_center_checks', 0)+len(ids)
    if GAP is None and RADIUS is None and not torch.equal(task._humanoid_root_states[ids], roots):
        raise ValueError('Natural humanoid initialization changed')


layout.translate_people = natural_centers
base_eval = formal_eval.run_formal_eval


def digest(state):
    value = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value.update(key.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


@torch.no_grad()
def checked_eval(player):
    task = player.env.task
    path = Path(task.cfg['args'].output_path)
    reference_only = os.environ.get('REFERENCE_ONLY') == '1'
    if reference_only:
        from stage2_protocol_eval import capture_initial
        return capture_initial(player)
    action = player.get_action
    checked = False
    preservation = {}
    if MODEL in ('ours', 'coopft'):
        checkpoint = torch.load(task.cfg['args'].checkpoint, map_location='cpu', weights_only=False)
        live = player.model.state_dict()
        if any(not torch.equal(tensor, live[key].detach().cpu()) for key, tensor in checkpoint['model'].items()):
            raise ValueError('Checkpoint model tensors differ')
        normalizer = player.running_mean_std.state_dict()
        if any(not torch.equal(tensor, normalizer[key].detach().cpu())
               for key, tensor in checkpoint['running_mean_std'].items()):
            raise ValueError('Checkpoint normalization differs')
        preservation = dict(model=digest(live), normalization=digest(normalizer), epoch=checkpoint['epoch'])

    def first_action(observation, *args, **kwargs):
        nonlocal checked
        if not checked:
            expected = torch.load(Path(os.environ['STAGE2_REFERENCE_ROOT'])/
                f'general_seed{task.cfg["args"].seed}'/'observed_scene.pt', map_location='cpu', weights_only=False)
            actual = observation['obs'].detach().cpu()
            exact = torch.equal(actual, expected)
            if not exact and MODEL != 'fsm':
                raise ValueError('Actual first policy input differs from shared reference')
            torch.save(actual, path/'actual_first_input.pt')
            (path/'first_policy_input.json').write_text(json.dumps(dict(exact=exact,
                maximum_absolute_difference=float((actual-expected).abs().max()),
                known_fsm_reset_recomputation=MODEL == 'fsm', model=MODEL), indent=2))
            checked = True
        return action(observation, *args, **kwargs)

    player.get_action = first_action
    try:
        result = base_eval(player)
    finally:
        player.get_action = action
    if not checked:
        raise ValueError('No first policy input captured')
    if preservation:
        if (digest(player.model.state_dict()), digest(player.running_mean_std.state_dict())) != (
                preservation['model'], preservation['normalization']):
            raise ValueError('Policy weights or normalization changed')
        preservation.update(weights_exact_before_after=True, normalization_exact_before_after=True)
        (path/'policy_preservation.json').write_text(json.dumps(preservation, indent=2))
    result['group_spawn_over4']['min_center_gap'] = GAP
    result['group_spawn_gap'] = GAP
    result['group_placement'] = PLACEMENT
    result['group_spawn_radius'] = RADIUS
    return result


formal_eval.run_formal_eval = checked_eval
original_run_path = runpy.run_path


def run_entry(filename, init_globals=None, run_name=None):
    path = Path(filename)
    if path.name not in ('stage2_scalability_eval.py', 'eval_entry.py'):
        return original_run_path(filename, init_globals=init_globals, run_name=run_name)
    source = path.read_text()
    if 'layout_metadata(task)' in source:
        return original_run_path(filename, init_globals=init_globals, run_name=run_name)
    check = ("            if minimum_gap < 4.99:\n"
             "                raise ValueError('Initial group center gap below five meters')\n"
             if path.name == 'stage2_scalability_eval.py' else
             "            if gap<4.99:\n"
             "                raise ValueError('FSM group center gap below five meters')\n")
    if source.count(check) != 1:
        raise ValueError('Group spacing validation source changed: '+str(path))
    source = source.replace(check, '')
    source = source.replace('group_spawn_gap=5. if task.num_agents > 4 else None', 'group_spawn_gap=None')
    source = source.replace('group_spawn_gap=5. if task.num_agents>4 else None', 'group_spawn_gap=None')
    namespace = dict(init_globals or {}, __name__=run_name or '<run_path>', __file__=str(path))
    exec(compile(source, str(path), 'exec'), namespace)
    return namespace


runpy.run_path = run_entry
if MODEL == 'coopft':
    from learning.multi_agent.mean_cooperation import MeanCooperation
    original_init = MeanCooperation.__init__

    def evaluation_init(self, *args, **kwargs):
        with torch.random.fork_rng(devices=[]):
            original_init(self, *args, **kwargs)

    MeanCooperation.__init__ = evaluation_init

entry = (ROOT.parent/'tokenhsi_push/scripts/tokenhsi_table_eval.py' if MODEL == 'tokenhsi' else
         ROOT/'FSM/eval_entry.py' if MODEL == 'fsm' else
         SOURCE/'eval/stage1_protocol_eval.py' if MODEL == 'stage1' else
         SOURCE/'eval/stage2_scalability_eval.py')
run_entry(entry, run_name='__main__')

destination = Path(sys.argv[sys.argv.index('--output_path')+1])
snapshot = torch.load(destination/('initial_before_fsm.pt' if MODEL == 'fsm' else 'initial.pt'),
                      map_location='cpu', weights_only=False)
groups, roots = snapshot['groups'], snapshot['roots'][..., :2]
if RADIUS is not None:
    groups = groups.clone()
    rows, followers = (snapshot['predecessor'] >= 0).nonzero(as_tuple=True)
    groups[rows, followers] = groups[rows, snapshot['predecessor'][rows, followers]]
same = groups[:, :, None] == groups[:, None, :]
centers = torch.bmm(same.float(), roots)/same.sum(-1, keepdim=True)
gaps = (centers[:, :, None]-centers[:, None]).norm(dim=-1).masked_fill(same, float('inf'))
minimum = gaps.amin((1, 2))
torch.save(dict(group_centers=centers, per_scene_minimum_gap=minimum), destination/'natural_spacing.pt')
metadata = dict(group_spawn_gap=GAP, group_placement=PLACEMENT, group_spawn_radius=RADIUS,
    layout_attempt_limit=1024,
    minimum_initial_group_gap=float(minimum.min()), scenes_with_group_gap_below_5m=int((minimum < 5).sum()),
    retained_group_sampler=True, retained_2k_token_budget=True, ring_rng_draws_retained=True,
    fsm_reset_behavior='legacy_preserved' if MODEL == 'fsm' else None)
(destination/'group_spacing.json').write_text(json.dumps(metadata, indent=2))
if (destination/'result.json').exists():
    result = json.loads((destination/'result.json').read_text())
    result.update(metadata)
    result['comparison_protocol'] = 'cvpr2027_table1_group_spacing_env64'
    result['group_spawn_over4']['min_center_gap'] = GAP
    (destination/'result.json').write_text(json.dumps(result, indent=2))
