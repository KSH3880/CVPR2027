"""Isolated reset/Stage-1 rollout audit of proposed shared-object combinations.

Uses current reset/reward code, not a new training sampler. Six proposed graphs
are injected only in this process, including the currently rejected double
ON_TOP target. Per-role assets are fixed at creation and physical/logical IDs
are identical in this probe. No production training configuration is changed.
"""
import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import runpy
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--deps', type=Path, required=True)
parser.add_argument('--output', type=Path, default=ROOT / 'output/shared_scenario_audit/runtime')
parser.add_argument('--envs-per-case', type=int, default=8)
parser.add_argument('--resets', type=int, default=10)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--steps', type=int, default=600)
parser.add_argument('--only-resets', action='store_true')
parser.add_argument('--case-set', choices=('default', 'midrange', 'carryable', 'transport_grid'), default='default')
parser.add_argument('--verify-snapshots', type=Path, help='CPU PhysX verification in a separate process')
parser.add_argument('--shared-loco', action='store_true', help='diagnostic alternative: shared SIT/CLIMB use loco RSI')
parser.add_argument('--all-loco', action='store_true', help='diagnostic alternative: every template uses loco RSI')
parser.add_argument('--cases', nargs='+', help='restrict the selected case set by exact case name')
parser.add_argument('--size-cases', type=Path, help='JSON list of case/support XYZ/payload XYZ for a bounded measurement')
parser.add_argument('--sharing-only', action='store_true', help='evaluate the six shared SIT/CLIMB/ON_TOP pairs')
parser.add_argument('--independent-only', action='store_true', help='separate objects/goals for AT, SIT, CLIMB and ON_TOP individual trials')
parser.add_argument('--snapshot-all', action='store_true', help='save every first-reset scene for unbiased CPU FK verification')
parser.add_argument('--checkpoint', type=Path, default=ROOT / 'stage1/ApproachScenarioStage1RescueAtKLClimb50_00008000.pth')
probe = parser.parse_args()
if probe.independent_only and probe.sharing_only:
    parser.error('independent-only and sharing-only are mutually exclusive')
independent_trials = probe.independent_only or probe.case_set == 'transport_grid'
if probe.envs_per_case < 1 or probe.resets < 0 or probe.repeats < 0 or probe.steps < 1:
    parser.error('envs-per-case/steps must be positive; resets/repeats must be nonnegative')
if not probe.deps.is_dir() or not probe.checkpoint.is_file():
    parser.error('deps directory and source checkpoint must exist')
probe.output.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(probe.deps), str(ROOT / 'tokenhsi'), str(ROOT)]
from isaacgym import gymapi
import numpy as np
np.float = float
np.int = int
import torch
import yaml
import fcl
from scipy.spatial.transform import Rotation

saved_argv = sys.argv
sys.argv = ['measure_shared_support.py', '--deps', str(probe.deps)]
spec = importlib.util.spec_from_file_location('shared_geometry', ROOT / 'tokenhsi/scripts/multi_agent/measure_shared_support.py')
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
sys.argv = saved_argv

from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.edge_context_reward import owner_sum, scene_success
from utils.edge_stage2_spec import _empty_graph, _put, validate_graph
from utils.edge_ontop_spec import select_graph, copy_graph_rows
from learning.multi_agent import edge_context_eval

PAIRS = [('AT', 'ON_TOP'), ('AT', 'CLIMB'), ('AT', 'SIT'),
         ('ON_TOP', 'ON_TOP'), ('ON_TOP', 'CLIMB'), ('ON_TOP', 'SIT'),
         ('CLIMB', 'CLIMB'), ('CLIMB', 'SIT'), ('SIT', 'SIT')]
CASES = [('current_range', None, None), ('support60_payload20', .6, .2),
         ('support60_payload25', .6, .25), ('support60_payload30', .6, .3),
         ('support60_payload40', .6, .4), ('support90_payload40', .9, .4),
         ('support125_payload40', 1.25, .4)]
if probe.case_set == 'midrange':
    CASES = [('current_range', None, None), ('support65_payload30', .65, .3),
             ('support75_payload35', .75, .35), ('support85_payload40', .85, .4),
             ('support75_payload40', .75, .4)]
elif probe.case_set == 'carryable':
    CASES = [('current_range', None, None)]
    for x, y in ((.65,.65), (.7,.7), (.75,.75), (.8,.8), (.6,.7), (.6,.75), (.6,.8)):
        for z in (.45, .5):
            for payload in (.3, .35, .4):
                name = 'support%d_%d_%d_payload%d' % tuple(round(v*100) for v in (x,y,z,payload))
                CASES.append((name, [x,y,z], [payload,payload,.3]))
elif probe.case_set == 'transport_grid':
    # Separate sources/goals remove shared-support prerequisites from this test.
    PAIRS = [('AT', 'AT')]
    CASES = []
    xy = [(i/100, j/100) for i in range(25,81,5) for j in range(i,91,5)] + [(.2,.2)]
    for x,y in xy:
        for z in (.3,.35,.4,.45,.5):
            name = 'transport%d_%d_%d' % tuple(round(v*100) for v in (x,y,z))
            CASES.append((name,[x,y,z],None))
if probe.size_cases:
    if probe.case_set == 'transport_grid':
        parser.error('size-cases is not used with the fixed transport grid')
    definitions = json.loads(probe.size_cases.read_text())
    if not definitions or any(set(d)!={'case','support','payload'} or
            any(len(d[k])!=3 or any(not isinstance(v,(float,int)) or not 0<v<10 for v in d[k])
                for k in ('support','payload')) for d in definitions):
        parser.error('size-cases must contain case, positive XYZ support, positive XYZ payload')
    CASES = [(d['case'],d['support'],d['payload']) for d in definitions]
if probe.sharing_only:
    if probe.case_set == 'transport_grid':
        parser.error('transport-grid uses independent AT pairs')
    PAIRS = PAIRS[3:]
if probe.independent_only:
    PAIRS = [('AT','AT'),('SIT','SIT'),('CLIMB','CLIMB'),('ON_TOP','ON_TOP')]
if probe.cases:
    unknown = set(probe.cases) - {c[0] for c in CASES}
    if unknown:
        parser.error('unknown cases: ' + ', '.join(sorted(unknown)))
    CASES = [c for c in CASES if c[0] in probe.cases]
N = len(PAIRS) * len(CASES) * probe.envs_per_case
rows = []
for case_index, (case, support, payload) in enumerate(CASES):
    for pair_index, pair in enumerate(PAIRS):
        for trial in range(probe.envs_per_case):
            rows.append(dict(case=case, pair='+'.join(pair), support=support,
                             payload=payload, pair_index=pair_index, trial=trial))

generator = np.random.RandomState(103)
asset_sizes = np.empty((N, 4, 3), dtype=np.float32)
for env, row in enumerate(rows):
    asset_sizes[env, :, :2] = generator.choice([.4, .45, .5, .55, .6], (4, 2))
    asset_sizes[env, :, 2] = generator.choice([.3, .35, .4, .45, .5], 4)
    if row['support'] is not None:
        asset_sizes[env] = [.4, .4, .3]
        asset_sizes[env, 0] = row['support'] if isinstance(row['support'], list) else [row['support'], row['support'], .5]
        if probe.case_set == 'transport_grid':
            asset_sizes[env, 1] = asset_sizes[env, 0]
        pair = PAIRS[row['pair_index']]
        for owner, skill in enumerate(pair):
            if skill == 'ON_TOP':
                asset_sizes[env, 1 + owner] = row['payload'] if isinstance(row['payload'], list) else [row['payload'], row['payload'], .3]
        if probe.independent_only:
            support_size = asset_sizes[env,0].copy()
            if pair[0] == 'ON_TOP':
                payload_size = row['payload'] if isinstance(row['payload'],list) else [row['payload'],row['payload'],.3]
                asset_sizes[env] = [support_size,payload_size,support_size,payload_size]
            else:
                asset_sizes[env,1] = support_size

# Reuse the asset builder verbatim except for its size matrix assignment.
# Runtime monkeypatches are confined to this diagnostic process.
asset_source = textwrap.dedent(inspect.getsource(HumanoidMACarry._load_box_asset))
size_expression = 'torch.tensor(self._build_base_size, device=self.device).reshape(1, 3) * self._box_scale'
if asset_source.count(size_expression) != 1:
    raise RuntimeError('Asset builder changed; update the diagnostic size override')
asset_source = asset_source.replace(
    size_expression,
    'torch.tensor(asset_sizes, device=self.device).reshape(num_boxes, 3)')
asset_namespace = dict(HumanoidMACarry._load_box_asset.__globals__, asset_sizes=asset_sizes)
exec(asset_source, asset_namespace)
HumanoidMACarry._load_box_asset = asset_namespace['_load_box_asset']

original_assignments = HumanoidMACarry._sample_box_assignments
def fixed_assignments(self, ids):
    self._randomize_box_assignment = False
    original_assignments(self, ids)
HumanoidMACarry._sample_box_assignments = fixed_assignments

def proposed_graph(device):
    graph = _empty_graph(N, 2, 4, 4, device)
    for env, row in enumerate(rows):
        edge = 0
        for owner, skill in enumerate(PAIRS[row['pair_index']]):
            if skill in ('AT', 'ON_TOP'):
                source = (2 + owner if independent_trials else 2) if skill == 'AT' else (3+2*owner if probe.independent_only else 3+owner)
                _put(graph, env, edge, owner, owner, source, 6, False)
                edge += 1
                _put(graph, env, edge, owner, source, 6 + owner if skill == 'AT' else (2+2*owner if probe.independent_only else 2),
                     7 if skill == 'AT' else 8, True)
            else:
                _put(graph, env, edge, owner, owner, 2+owner if probe.independent_only else 2, 9 if skill == 'SIT' else 10, True)
            edge += 1
    return graph

original_sample = HumanoidMACarry._sample_episode_graph
def injected_sample(self, ids):
    original_sample(self, ids)
    if not hasattr(self, '_audit_graph'):
        self._audit_graph = proposed_graph(self.device)
    graph = select_graph(self._audit_graph, ids)
    copy_graph_rows(self.relation_runtime.graph, ids, graph)
    self._edge_goal_owners[ids] = owner_sum((graph.required_goal & graph.edge_valid).float(), graph) > 0
HumanoidMACarry._sample_episode_graph = injected_sample
original_ref_init = HumanoidMACarry._reset_ref_state_init
def shared_loco_ref_init(self, slot_env, slot_agent):
    if not (probe.shared_loco or probe.all_loco):
        return original_ref_init(self, slot_env, slot_agent)
    previous = {}
    for name in ('_template_rsi_weights', '_independent_template_rsi_weights'):
        value = getattr(self, name, None)
        if value is not None:
            previous[name] = value
            replacement = value.clone()
            chosen = slice(None) if probe.all_loco else slice(1,3)
            replacement[chosen] = 0
            replacement[chosen, self._skill.index('loco')] = 1
            setattr(self, name, replacement)
    try:
        return original_ref_init(self, slot_env, slot_agent)
    finally:
        for name, value in previous.items():
            setattr(self, name, value)
HumanoidMACarry._reset_ref_state_init = shared_loco_ref_init

if independent_trials:
    original_compute_reset = HumanoidMACarry._compute_reset
    def transport_compute_reset(self):
        original_compute_reset(self)
        # A failed independent carrier must not truncate the other's trial.
        # Keep per-human terminate flags and stop recording that human below.
        self.reset_buf[:] = (self.progress_buf >= self.max_episode_length - 1).long()
    HumanoidMACarry._compute_reset = transport_compute_reset

def penetration(a, b):
    data = fcl.CollisionData(fcl.CollisionRequest(num_max_contacts=1000, enable_contact=True))
    a.collide(b, data, fcl.defaultCollisionCallback)
    return max([float(c.penetration_depth) for c in data.result.contacts] or [0.])

def inspect_reset(task, shapes):
    bodies = task._kinematic_humanoid_rigid_body_states.detach().cpu().numpy()
    boxes = task._box_states.detach().cpu().numpy()
    sizes = task._box_size.detach().cpu().numpy()
    result = []
    skills = [['default'] * 2 for _ in range(N)]
    for skill, (envs, agents) in task._reset_ref_slots.items():
        for env, agent in zip(envs.tolist(), agents.tolist()):
            skills[env][agent] = skill
    for env, row in enumerate(rows):
        humans = []
        human_objects = []
        for agent in range(2):
            objects = []
            for shape in shapes:
                state = bodies[env, agent, shape['body']]
                rotation = Rotation.from_quat(state[3:7]).as_matrix()
                objects.append(fcl.CollisionObject(shape['shape'], fcl.Transform(
                    rotation @ shape['rot'], state[:3] + rotation @ shape['local'])))
            humans.append(geometry.manager(objects))
            human_objects.append(objects)
        object_managers = []
        box_objects = []
        for obj in range(4):
            box_object = fcl.CollisionObject(fcl.Box(*sizes[env, obj]),
                fcl.Transform(Rotation.from_quat(boxes[env, obj, 3:7]).as_matrix(), boxes[env, obj, :3]))
            box_objects.append(box_object)
            object_managers.append(geometry.manager([box_object]))
        human_depth = penetration(humans[0], humans[1])
        box_depth = max(penetration(human, obj) for human in humans for obj in object_managers)
        details = []
        graph = task._audit_graph
        if box_depth > .02:
            for agent in range(2):
                held = {int(graph.edge_dst[env,e])-2 for e in range(4)
                        if graph.edge_valid[env,e] and graph.edge_owner[env,e] == agent and graph.edge_relation[env,e] == 6}
                for obj in range(4):
                    if penetration(humans[agent], object_managers[obj]) <= .02:
                        continue
                    for shape, human_object in zip(shapes, human_objects[agent]):
                        collision = fcl.CollisionResult()
                        fcl.collide(human_object, box_objects[obj],
                            fcl.CollisionRequest(num_max_contacts=20, enable_contact=True), collision)
                        depth = max([float(c.penetration_depth) for c in collision.contacts] or [0.])
                        if depth > .02:
                            details.append(dict(agent=agent, object=obj, geom=shape['name'], depth_m=depth,
                                                own_held=obj in held, hand='hand' in shape['name']))
        non_hand = any(not d['hand'] for d in details)
        foreign = any(not d['own_held'] for d in details)
        result.append(dict(env=env, case=row['case'], pair=row['pair'], skills=skills[env],
                           human_human_depth_m=human_depth, human_box_depth_m=box_depth,
                           human_human_penetration=human_depth > .02,
                           human_box_penetration=box_depth > .02,
                           non_hand_box_penetration=non_hand, non_held_box_penetration=foreign,
                           penetrations=details))
    return result

def grouped(records, flag):
    output = []
    for case, _, _ in CASES:
        for a, b in PAIRS:
            group = [r for r in records if r['case'] == case and r['pair'] == a + '+' + b]
            output.append(dict(case=case, pair=a+'+'+b, scenes=len(group),
                               affected=sum(bool(r[flag]) for r in group),
                               rate=sum(bool(r[flag]) for r in group) / max(1, len(group))))
    return output

def verify_reset_physx(snapshots):
    """Check representative accepted reset poses with real PhysX contact shapes."""
    if not snapshots:
        return []
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1e-4
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, 0)
    params.physx.use_gpu = False
    params.use_gpu_pipeline = False
    params.physx.num_threads = 4
    params.physx.contact_offset = .02
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    options = gymapi.AssetOptions()
    options.disable_gravity = True
    options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    human_asset = gym.load_asset(sim, str(geometry.ASSET.parent.parent), 'mjcf/phys_humanoid_v3.xml', options)
    shapes = geometry.shapes_from_asset(gym.get_asset_rigid_body_names(human_asset))
    fixed = gymapi.AssetOptions()
    fixed.fix_base_link = True
    scenes = []
    for index, snapshot in enumerate(snapshots):
        env = gym.create_env(sim, gymapi.Vec3(-4, -4, 0), gymapi.Vec3(4, 4, 4), 10)
        actors = []
        human_body_ids = set()
        for agent in range(2):
            root = snapshot['roots'][agent]
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*root[:3])
            pose.r = gymapi.Quat(*root[3:7])
            actor = gym.create_actor(env, human_asset, pose, 'human'+str(agent), index, 0)
            state = np.zeros(32, dtype=gymapi.DofState.dtype)
            state['pos'] = snapshot['dofs'][agent]
            gym.set_actor_dof_states(env, actor, state, gymapi.STATE_ALL)
            ids = set(gym.get_actor_rigid_body_index(env, actor, i, gymapi.DOMAIN_ENV)
                      for i in range(gym.get_actor_rigid_body_count(env, actor)))
            actors.append((actor, ids))
            human_body_ids |= ids
        box_ids = set()
        box_actors = []
        for obj in range(4):
            box = snapshot['boxes'][obj]
            asset = gym.create_box(sim, *snapshot['sizes'][obj], fixed)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*box[:3])
            pose.r = gymapi.Quat(*box[3:7])
            actor = gym.create_actor(env, asset, pose, 'box'+str(obj), index, 0)
            box_actors.append(actor)
            box_ids.add(gym.get_actor_rigid_body_index(env, actor, 0, gymapi.DOMAIN_ENV))
        scenes.append((env, snapshot, actors, human_body_ids, box_ids, box_actors))
    gym.prepare_sim(sim)
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    results = []
    for env, snapshot, actors, human_ids, box_ids, box_actors in scenes:
        contacts = gym.get_env_rigid_contacts(env)
        field = next(k for k in contacts.dtype.names if k.replace('_', '').lower() == 'initialoverlap')
        cross = [c for c in contacts if (int(c['body0']) in human_ids and int(c['body1']) in box_ids) or
                                      (int(c['body1']) in human_ids and int(c['body0']) in box_ids)]
        hh = [c for c in contacts if (int(c['body0']) in actors[0][1] and int(c['body1']) in actors[1][1]) or
                                   (int(c['body1']) in actors[0][1] and int(c['body0']) in actors[1][1])]
        def pose(state):
            p = state['pose']['p']; q = state['pose']['r']
            return np.array([p['x'],p['y'],p['z']]), Rotation.from_quat([q['x'],q['y'],q['z'],q['w']]).as_matrix()
        humans = []
        for actor, _ in actors:
            state = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
            objects = []
            for shape in shapes:
                p, rotation = pose(state[shape['body']])
                objects.append(fcl.CollisionObject(shape['shape'], fcl.Transform(
                    rotation @ shape['rot'], p + rotation @ shape['local'])))
            humans.append(geometry.manager(objects))
        boxes = []
        for obj, actor in enumerate(box_actors):
            state = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
            p, rotation = pose(state[0])
            boxes.append(geometry.manager([fcl.CollisionObject(fcl.Box(*snapshot['sizes'][obj]), fcl.Transform(rotation,p))]))
        actual_hb = max(penetration(h, b) for h in humans for b in boxes)
        actual_hh = penetration(humans[0], humans[1])
        results.append(dict(case=snapshot['case'], pair=snapshot['pair'], kind=snapshot['kind'],
                            human_box_contact_count=len(cross),
                            human_box_initial_overlap_max=max([float(c[field]) for c in cross] or [0.]),
                            human_human_contact_count=len(hh),
                            human_human_initial_overlap_max=max([float(c[field]) for c in hh] or [0.]),
                            physx_fk_human_box_fcl_depth_m=actual_hb,
                            physx_fk_human_human_fcl_depth_m=actual_hh))
    gym.destroy_sim(sim)
    (probe.output / 'reset_physx_contacts.json').write_text(json.dumps(results, indent=2))
    return results

@torch.no_grad()
def audit(player):
    task = player.env.task
    names = task.gym.get_asset_rigid_body_names(task.humanoid_asset) if hasattr(task, 'humanoid_asset') else task.gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles_all[0][0])
    shapes = geometry.shapes_from_asset(names)
    reset_records = []
    failures = []
    snapshots = {}
    original_skills = task.cfg['env']['skill']
    task._skill = original_skills
    task._skill_init_prob = torch.tensor(task.cfg['env']['skillInitProb'], device=task.device)
    task._stage2_rsi_counts = torch.zeros(len(original_skills), device=task.device)
    task._is_eval = False
    if not hasattr(task, '_audit_graph'):
        task._audit_graph = proposed_graph(task.device)
    validation = []
    for pair_index, (a, b) in enumerate(PAIRS):
        env = pair_index * probe.envs_per_case
        try:
            validate_graph(select_graph(task._audit_graph, env))
            error = None
        except ValueError as exc:
            error = str(exc)
        validation.append(dict(pair=a+'+'+b, current_validator_error=error))
    for trial in range(probe.resets):
        try:
            player.env_reset(torch.arange(N, device=task.device) * 2)
        except RuntimeError as error:
            failures.append(dict(trial=trial, error=str(error)))
            print('[audit reset failure]', trial, str(error), flush=True)
            continue
        inspected = inspect_reset(task, shapes)
        for r in inspected:
            r['reset_trial'] = trial
            kind = ('unbiased_first_reset_%d' % r['env'] if probe.snapshot_all and trial == 0 else
                    'human_human' if r['human_human_penetration'] else
                    'non_hand_box' if r['non_hand_box_penetration'] else None)
            key = (r['case'], r['pair'], kind)
            if kind and key not in snapshots:
                env = r['env']
                snapshots[key] = dict(case=r['case'], pair=r['pair'], kind=kind,
                    roots=task._humanoid_root_states[env].cpu().tolist(),
                    dofs=task._dof_pos[env].cpu().tolist(), boxes=task._box_states[env].cpu().tolist(),
                    sizes=task._box_size[env].cpu().tolist())
        reset_records.extend(inspected)
        print('[audit reset]', trial, 'human-human >2cm', sum(r['human_human_penetration'] for r in inspected),
              'human-box >2cm', sum(r['human_box_penetration'] for r in inspected), flush=True)
        (probe.output / 'reset_records.json').write_text(json.dumps(reset_records, indent=2))
    summary = dict(method='Current training RSI/reset paths; exact MJCF FCL shapes on pre-step reference FK',
                   seed=103, checkpoint=str(probe.checkpoint), scenes_per_reset=N,
                   resets=probe.resets, failures=failures, shared_loco=probe.shared_loco, all_loco=probe.all_loco,
                   proposed_graph_validation=validation,
                   human_human=grouped(reset_records, 'human_human_penetration'),
                   human_box=grouped(reset_records, 'human_box_penetration'),
                   non_hand_box=grouped(reset_records, 'non_hand_box_penetration'),
                   non_held_box=grouped(reset_records, 'non_held_box_penetration'),
                   sampling_retries=float(task._sampling_retries),
                   rsi_rejections=task._rsi_rejected.tolist(),
                   limitations=['Proposed graphs injected only in this process; six families are not implemented in training.',
                                'Depth >2cm is flagged, not expected light hand/foot support contact.',
                                'FCL foot hulls may differ slightly from PhysX cooking; no collision-free guarantee.'])
    (probe.output / 'reset_summary.json').write_text(json.dumps(summary, indent=2))
    (probe.output / 'reset_physx_snapshots.json').write_text(json.dumps(list(snapshots.values()), indent=2))
    if probe.only_resets:
        return summary

    # Stage-1-only rollouts start from loco, as in the existing viewer baseline.
    task._is_eval = True
    task._skill = ['loco']
    task._skill_init_prob = torch.ones(1, device=task.device)
    rollout = []
    for repeat in range(probe.repeats):
        obs = player.env_reset(torch.arange(N, device=task.device) * 2)
        player.get_batch_size(obs['obs'], 1)
        alive = torch.ones(N, dtype=torch.bool, device=task.device)
        ever = torch.zeros(N, 4, dtype=torch.bool, device=task.device)
        final = torch.zeros_like(ever)
        joint_ever = torch.zeros_like(alive)
        terminated = torch.zeros_like(alive)
        ever_supported = torch.zeros_like(ever)
        success_run = torch.zeros(N, 4, dtype=torch.long, device=task.device)
        max_success_run = torch.zeros_like(success_run)
        initial_xy = task._box_states[..., :2].clone()
        max_lift = torch.zeros(N, 4, device=task.device)
        max_motion = torch.zeros_like(max_lift)
        carried = torch.zeros_like(ever)
        delivered_after_transport = torch.zeros_like(ever)
        owner_alive = torch.ones(N,2,dtype=torch.bool,device=task.device)
        joint_run = torch.zeros(N, dtype=torch.long, device=task.device)
        max_joint_run = torch.zeros_like(joint_run)
        for step in range(probe.steps + 1):
            action = player.get_action(obs, True)
            if not torch.isfinite(action).all():
                raise RuntimeError('Non-finite policy action in shared scenario audit')
            obs, _, done, info = player.env_step(player.env, action)
            current = task.relation_runtime.own_success
            graph = task._audit_graph
            edge_alive = alive[:,None] & owner_alive.gather(1,graph.edge_owner)
            current = current & edge_alive
            success_run = torch.where(current & alive[:, None], success_run + 1, 0)
            max_success_run = torch.maximum(max_success_run, success_run)
            from utils.torch_utils import quat_rotate
            graph = task._audit_graph
            batch = torch.arange(N, device=task.device)
            signs = torch.tensor([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)], device=task.device)
            for edge in range(4):
                src = (graph.edge_src[:, edge] - 2).clamp(0,3)
                dst = (graph.edge_dst[:, edge] - 2).clamp(0,3)
                source = task._box_states[batch, src]
                target = task._box_states[batch, dst]
                corners = task._box_size[batch, src, None] * signs[None] / 2
                source_q = source[:,None,3:7].expand(-1,8,-1).reshape(-1,4)
                world = quat_rotate(source_q, corners.reshape(-1,3)).reshape(N,8,3) + source[:,None,:3]
                target_q = target[:,None,3:7].expand(-1,8,-1).reshape(-1,4).clone()
                target_q[:,:3] *= -1
                local = quat_rotate(target_q, (world-target[:,None,:3]).reshape(-1,3)).reshape(N,8,3)
                supported = (local[:,:,:2].abs() <= task._box_size[batch,dst,None,:2]/2 + .002).all(-1).all(-1)
                ever_supported[:,edge] |= supported & current[:,edge] & alive & (graph.edge_relation[:,edge] == 8)
            max_lift = torch.maximum(max_lift, (task._box_states[..., 2] - task._box_size[..., 2] / 2) * alive[:, None])
            max_motion = torch.maximum(max_motion, (task._box_states[..., :2] - initial_xy).norm(dim=-1) * alive[:, None])
            lifted = task._box_states[..., 2] - task._box_size[..., 2]/2 > .15
            moved = (task._box_states[..., :2] - initial_xy).norm(dim=-1) > .5
            for edge in range(4):
                src = (graph.edge_dst[:,edge]-2).clamp(0,3)
                carried[:,edge] |= current[:,edge] & alive & lifted[batch,src] & moved[batch,src] & (graph.edge_relation[:,edge] == 6)
            if independent_trials:
                for owner in range(2):
                    delivered_after_transport[:,2*owner+1] |= current[:,2*owner+1] & carried[:,2*owner] & (graph.edge_relation[:,2*owner+1] == 7)
                owner_alive &= ~info['terminate'].reshape(N,2).bool()
            ever |= current & alive[:, None]
            joint = scene_success(current, task.relation_runtime.graph) & alive
            joint_ever |= joint
            joint_run = torch.where(joint, joint_run + 1, 0)
            max_joint_run = torch.maximum(max_joint_run, joint_run)
            ending = done.reshape(N, 2).bool().any(-1)
            collect = ending & alive
            final[collect] = current[collect]
            terminated[collect] = info['terminate'].reshape(N, 2)[collect].bool().any(-1)
            alive &= ~ending
            if not alive.any():
                break
            obs = player.env_reset(ending.nonzero(as_tuple=False).flatten() * 2)
        for env, row in enumerate(rows):
            graph = task._audit_graph
            relations = graph.edge_relation[env]
            valid = graph.edge_valid[env]
            required = graph.required_goal[env] & valid
            r = dict(case=row['case'], pair=row['pair'], repeat=repeat, env=env,
                     joint_ever=bool(joint_ever[env]), joint_final=bool((final[env] | ~required).all()),
                     terminated=bool(terminated[env]),
                     owner_terminated=(~owner_alive[env]).tolist(),
                     max_joint_success_run_steps=int(max_joint_run[env]),
                     edges=[dict(relation=int(relations[e]), ever=bool(ever[env,e]), final=bool(final[env,e]),
                                 source=int(graph.edge_src[env,e]), target=int(graph.edge_dst[env,e]),
                                 ever_fully_supported=bool(ever_supported[env,e]) if relations[e] == 8 else None,
                                 ever_lifted_and_transported=bool(carried[env,e]) if relations[e] == 6 else None,
                                 delivered_after_transport=bool(delivered_after_transport[env,e]) if relations[e] == 7 else None,
                                 max_success_run_steps=int(max_success_run[env,e]))
                            for e in range(4) if valid[e]])
            r['max_box_lift_m'] = max_lift[env].tolist()
            r['max_box_xy_motion_m'] = max_motion[env].tolist()
            rollout.append(r)
        print('[audit rollout]', repeat, 'joint ever', float(joint_ever.float().mean()), flush=True)
        (probe.output / 'rollout_records.json').write_text(json.dumps(rollout, indent=2))
    output = dict(checkpoint=str(probe.checkpoint), stage2_training_steps=0, steps=probe.steps,
                  envs_per_case=probe.envs_per_case, repeats=probe.repeats,
                  cases=CASES, pairs=PAIRS, asset_sizes_m=asset_sizes.tolist(),
                  transport_grid=probe.case_set == 'transport_grid',
                  independent_only=probe.independent_only,
                  transport_definition='HOLDING + bottom >0.15m + XY displacement >0.5m, then own AT before per-human termination',
                  joint_ever=grouped(rollout, 'joint_ever'), joint_final=grouped(rollout, 'joint_final'),
                  termination=grouped(rollout, 'terminated'))
    (probe.output / 'rollout_summary.json').write_text(json.dumps(output, indent=2))
    return output

edge_context_eval.run_edge_context_eval = audit
if probe.verify_snapshots:
    confirmation = verify_reset_physx(json.loads(probe.verify_snapshots.read_text()))
    print('[audit CPU PhysX]', len(confirmation), 'scenes;',
          sum(r['physx_fk_human_box_fcl_depth_m'] > .02 for r in confirmation), 'human-box shape intersections >2cm;',
          sum(r['physx_fk_human_human_fcl_depth_m'] > .02 for r in confirmation), 'human-human shape intersections >2cm', flush=True)
    sys.exit(0)
sys.argv = [str(ROOT / 'tokenhsi/run.py'), '--task', 'HumanoidMACarry',
    '--cfg_train', 'tokenhsi/data/cfg/train/rlg/amp_ma_stage2_rescue_klclimb50.yaml',
    '--cfg_env', 'tokenhsi/data/cfg/multi_agent/approach_stage2_rescue_klclimb50.yaml',
    '--motion_file', 'tokenhsi/data/dataset_loco_sit_carry_climb.yaml',
    '--checkpoint', str(probe.checkpoint), '--num_agents', '2', '--num_envs', str(N),
    '--num_objects', '4', '--episode_length', str(probe.steps), '--eval_skills', 'loco',
    '--eval_skill_probs', '1.0', '--output_path', str(probe.output), '--seed', '103',
    '--task_graph', 'independent' if independent_trials else 'place_stack', '--headless', '--no_video', '--stage1_only', '--test', '--eval']
runpy.run_path(str(ROOT / 'tokenhsi/run.py'), run_name='__main__')
