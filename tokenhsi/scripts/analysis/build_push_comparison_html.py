"""Export matched recorded policy states as a self-contained interactive 3D HTML."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tokenhsi'))
import isaacgym
from isaacgym import gymapi, gymtorch
import numpy as np
import torch

SOURCE = Path(os.environ.get('COMPARE_OUTPUT_ROOT', str(ROOT/'output_etc/check/push_comparison_capture_20261112'))).resolve()
INITIAL_SOURCE = Path(os.environ.get('COMPARE_SOURCE_ROOT', str(ROOT/'output_etc/check/push_wait_v4_seed0_5_20261112'))).resolve()
OUT = ROOT/'output_etc/check/push_comparison_html_20261112'


def packed(value, dtype=np.float32):
    return base64.b64encode(np.asarray(value, dtype=dtype).tobytes()).decode()


def obj_mesh(path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'v':
            vertices.append([float(v) for v in parts[1:4]])
        elif parts[0] == 'f':
            indices = [int(v.split('/')[0])-1 for v in parts[1:]]
            faces.extend([[indices[0], indices[i], indices[i+1]] for i in range(1, len(indices)-1)])
    return dict(vertices=vertices, faces=faces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', default=['2:95', '2:205'])
    parser.add_argument('--output', type=Path, default=OUT/'seed2_env95_comparison.html')
    parser.add_argument('--ranking-file', type=Path)
    parser.add_argument('--allow-missing-methods', action='store_true')
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT/'output_etc/check')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    requested = [tuple(int(v) for v in key.split(':')) for key in args.cases]
    initials = {seed: torch.load(INITIAL_SOURCE/f'seed{seed}/initial.pt', map_location='cpu', weights_only=False) for seed, _ in requested}
    initial = initials[requested[0][0]]
    ranking = json.loads((args.ranking_file or SOURCE/'comparison_ranking.json').read_text())
    by_key = {(c['seed'], c['env_id']): c for c in ranking['cases']}
    cases = {key: by_key[key] for key in requested}
    methods = ('ours', 'tokenhsi', 'fsm')
    runs = {}
    for seed in initials:
        for method in methods:
            path=SOURCE/f'seed{seed}/{method}/rollout.pt'
            if args.allow_missing_methods and not (path.parent/'verification.json').exists():
                continue
            runs[(seed,method)]=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    for seed, reference in initials.items():
        for method in methods:
            if (seed,method) not in runs:
                continue
            run = runs[(seed, method)]
            proof = run['verification']
            assert proof['num_envs'] == reference['num_envs'] and proof['num_agents'] == 2
            assert all(proof['initial_exact'].values())
            assert Path(proof['source_initial']).resolve() == (INITIAL_SOURCE/f'seed{seed}/initial.pt').resolve()
            chosen = run['env_ids']
            torch.testing.assert_close(run['trajectory']['roots'][0], reference['roots'][chosen], atol=0, rtol=0)
            physical = reference['physical_indices'][chosen]
            torch.testing.assert_close(run['trajectory']['boxes'][0], reference['boxes'][chosen, physical], atol=0, rtol=0)
    strong_path = SOURCE/'genuine_push_ranking.json' if (SOURCE/'genuine_push_ranking.json').exists() else SOURCE/'strong_token_ranking.json'
    strong = {(c['seed'], c['env_id']): c for c in json.loads(strong_path.read_text())['cases']} if strong_path.exists() else {}
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1e-9
    params.substeps = 1
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, 0)
    params.physx.use_gpu = False
    params.use_gpu_pipeline = False
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    assert sim is not None
    options = gymapi.AssetOptions()
    options.disable_gravity = True
    options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    asset_config = initial['config']['env']['asset']
    asset_path = ROOT/asset_config['assetRoot']/asset_config['assetFileName']
    asset = gym.load_asset(sim, str(asset_path.parent), asset_path.name, options)
    body_names = gym.get_asset_rigid_body_names(asset)
    dof_names = gym.get_asset_dof_names(asset)
    scene_keys = [(seed, env_id, method) for seed, env_id in cases for method in methods
        if (seed,method) in runs and env_id in runs[(seed,method)]['env_ids']]
    assert all(any(key[:2]==case for key in scene_keys) for case in cases), 'No recorded method for requested case'
    if not args.allow_missing_methods:
        assert len(scene_keys)==len(cases)*len(methods)
    envs = []
    for scene, key in enumerate(scene_keys):
        env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 3)
        envs.append(env)
        for person in range(2):
            handle = gym.create_actor(env, asset, gymapi.Transform(), f'human{person}', scene*2+person, 1)
            props = gym.get_actor_dof_properties(env, handle)
            props['stiffness'][:] = 0
            props['damping'][:] = 0
            props['driveMode'][:] = gymapi.DOF_MODE_NONE
            gym.set_actor_dof_properties(env, handle, props)
    gym.prepare_sim(sim)
    roots = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dofs = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
    bodies = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
    poses = {key: [] for key in scene_keys}
    root_error = 0.
    for frame in range(901):
        for scene, (seed, env_id, method) in enumerate(scene_keys):
            initial = initials[seed]
            run = runs[(seed, method)]
            index = run['env_ids'].index(env_id)
            trace = run['trajectory']
            roots[scene*2:scene*2+2] = trace['roots'][frame, index]
            roots[scene*2:scene*2+2, :3] -= initial['env_origins'][env_id]
            roots[scene*2:scene*2+2, 7:] = 0
            d = trace['dof'][frame, index]
            dofs[scene*2*len(dof_names):(scene+1)*2*len(dof_names)] = d
            dofs[scene*2*len(dof_names):(scene+1)*2*len(dof_names), 1] = 0
        expected = roots[:, :7].clone()
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(roots))
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dofs))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)
        root_error = max(root_error, float((roots[:, :7]-expected).abs().max()))
        current = bodies.reshape(len(scene_keys), 2, len(body_names), 13)
        for scene, key in enumerate(scene_keys):
            poses[key].append(current[scene, :, :, :7].numpy().copy())
        if frame % 300 == 0:
            print('[HTML body export]', frame, '/900', flush=True)
    gym.destroy_sim(sim)
    tree = ET.parse(asset_path).getroot()
    mesh_paths = {e.attrib['name']: asset_path.parent/e.attrib['file'] for e in tree.findall('./asset/mesh')}
    geometry = []
    for body in tree.findall('.//body'):
        name = body.attrib.get('name')
        if name not in body_names:
            continue
        for geom in body.findall('geom'):
            kind = geom.attrib.get('type', 'capsule')
            item = dict(body=body_names.index(name), kind=kind, pos=[float(v) for v in geom.attrib.get('pos', '0 0 0').split()],
                        size=[float(v) for v in geom.attrib.get('size', '.04').split()])
            if 'fromto' in geom.attrib:
                item['fromto'] = [float(v) for v in geom.attrib['fromto'].split()]
            if kind == 'mesh':
                item['mesh'] = obj_mesh(mesh_paths[geom.attrib['mesh']])
            geometry.append(item)
    scenes = {}
    for (seed, env_id), case in cases.items():
        initial = initials[seed]
        physical = initial['physical_indices'][env_id]
        origin = initial['env_origins'][env_id].numpy()
        goal = initial['center_goals'][env_id, 0].numpy()-origin
        scene = dict(seed=seed, env=env_id, label=case.get('label') or (f"조기 {strong[(seed, env_id)]['tokenhsi']['clear_partner_early_motion_m']:.2f}m · 목표 오차 {strong[(seed, env_id)]['tokenhsi']['final_goal_error_m']:.2f}m" if strong.get((seed, env_id), {}).get("strong_token") else "기존 비교 사례"), early=case.get('early',case['methods'].get('ours',case['methods'].get('tokenhsi',next(iter(case['methods'].values()))))['first_person']),
            boxSize=initial['sizes'][env_id, physical].tolist(), goal=goal.tolist(),
            matches=case['matches'], methods={},pendingMethods=case.get('pending_methods',{}))
        for method in methods:
            if (seed,env_id,method) not in scene_keys:
                continue
            run = runs[(seed, method)]
            index = run['env_ids'].index(env_id)
            trace = run['trajectory']
            boxes = trace['boxes'][:, index, :7].numpy().copy()
            boxes[:, :3] -= origin
            pose = np.stack(poses[(seed, env_id, method)])
            assert np.isfinite(pose).all()
            error = np.linalg.norm(boxes[:, :3]-goal, axis=-1)
            body_root_error = float(np.abs(pose[:, :, body_names.index('pelvis'), :3]-(trace['roots'][:, index, :, :3].numpy()-origin)).max())
            assert body_root_error < 1e-4, body_root_error
            scene['methods'][method] = dict(poses=packed(pose), boxes=packed(boxes), errors=packed(error),
                face=packed(trace['on_face'][:, index].numpy(), np.uint8),
                frozen=packed(trace['frozen'][:, index].numpy(), np.uint8),
                phases=case['methods'][method]['phase_steps'], summary=case['methods'][method],
                source=str(SOURCE/f'seed{seed}/{method}/rollout.pt'),
                checkpoint=run['verification']['policy'].get('checkpoint'), body_root_error=body_root_error)
        candidate = strong.get((seed, env_id))
        if candidate and candidate['strong_token'] and not args.allow_missing_methods:
            token = scene['methods']['tokenhsi']
            token['phases'][1] = candidate['tokenhsi']['early_capture_step']
            token['summary'].update(candidate['tokenhsi'])
            if candidate['tokenhsi'].get('genuine_early_push'):
                token['phases'][0] = candidate['tokenhsi']['first_force_push_step']+1
            error = np.frombuffer(base64.b64decode(token['errors']), dtype=np.float32)
            closest = int(error.argmin())
            valid = runs[(seed, 'tokenhsi')]['trajectory']['alive'][:, runs[(seed, 'tokenhsi')]['env_ids'].index(env_id)].all(-1).numpy()
            after = np.flatnonzero(valid & (np.arange(901) >= max(closest+15, token['phases'][1]+15, token['summary'].get('later_joint_step', -1)+15)))
            if len(after):
                target = float(error[after].max())*.95
                token['phases'][2] = int(after[np.flatnonzero(error[after] >= target)[0]])
            # Show the first clear post-release FSM mismatch, rather than its
            # much later extreme drift, so the comparison remains easy to view.
            fsm_run = runs[(seed, 'fsm')]
            fsm_index = fsm_run['env_ids'].index(env_id)
            fsm_box = fsm_run['trajectory']['boxes'][:, fsm_index, :3].numpy()
            travel = goal[:2]+origin[:2]-fsm_box[0, :2]
            travel /= np.linalg.norm(travel)
            delta = fsm_box[:, :2]-fsm_box[0, :2]
            lateral = np.abs(delta[:, 0]*travel[1]-delta[:, 1]*travel[0])
            fsm_error = np.linalg.norm(fsm_box-(goal+origin), axis=-1)
            release = scene['methods']['fsm']['summary']['release_step']
            eligible = ((np.arange(901) >= release+30) & (lateral >= 1.1) & (fsm_error >= 1.)
                & ~fsm_run['trajectory']['on_face'][:, fsm_index].all(-1).numpy()
                & fsm_run['trajectory']['alive'][:, fsm_index].all(-1).numpy())
            if release >= 0 and eligible.any():
                scene['methods']['fsm']['phases'][2] = int(np.flatnonzero(eligible)[0])
        scenes[f'{seed}:{env_id}'] = scene
    payload = dict(seed=requested[0][0], defaultEnv=f'{requested[0][0]}:{requested[0][1]}', frames=901, dt=initial['dt'], bodyNames=body_names, geometry=geometry,
        reviewOnly=args.allow_missing_methods,startCompare=args.allow_missing_methods,
        reviewNote=ranking.get('review_note'),
        scenes=scenes, provenance='Recorded actual policy executions; body poses reconstructed from original MJCF and saved root/DOF states. This HTML is a replay, not a live policy.',
        thresholds=dict(goalDistance=.2, contact='Both hands on assigned face at native tolerance'), rootExportMaxError=root_error)
    template = (ROOT/'tokenhsi/scripts/analysis/push_comparison_viewer.html').read_text()
    library = (OUT/'three.min.js').read_text()
    output = template.replace('/*__THREE_LIBRARY__*/', library).replace('/*__PAYLOAD__*/', json.dumps(payload, separators=(',', ':'), default=lambda x:x.item()))
    html = args.output
    html.write_text(output)
    verification = dict(seed=requested[0][0], default_env=requested[0][1], included_cases=list(cases), methods=list(methods), frames=901,
        humans=2, geometry_primitives=len(geometry), rigid_bodies=len(body_names),
        body_pose_reconstruction_root_max_error=root_error,
        html=str(html), bytes=html.stat().st_size, sha256=hashlib.sha256(html.read_bytes()).hexdigest(),
        dependency='Three.js r146 embedded; no network required to open HTML', gpu_policy_runs_started=0,
        source_initial_same_for_all_methods=True, replay_not_live_policy=True)
    html.with_suffix('.verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == '__main__':
    main()
