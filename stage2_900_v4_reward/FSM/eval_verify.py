import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tokenhsi'),str(ROOT/'eval')]
import numpy as np
import torch
import stage2_xy100
from rescore import rescore
from state import CoordinationState
from table2_metrics import replay


def verify(path,reference,people,envs,steps):
    stage2_xy100.install()
    result = json.loads((path/'result.json').read_text())
    if (result['num_agents'],result['num_objects'],result['num_envs'],result['horizon'],result['simulated_steps']) != (people,2*people,envs,steps,steps):
        raise ValueError('FSM evaluation dimensions mismatch')
    before = torch.load(reference/'initial.pt',map_location='cpu')
    after = torch.load(path/'initial_before_fsm.pt',map_location='cpu')
    parked = torch.load(path/'physics_only_reset.pt',map_location='cpu')
    for key in before:
        if key=='boxes':
            if not torch.equal(before[key][~parked['mask']],after[key][~parked['mask']]):
                raise ValueError('Required initial physical box mismatch')
        elif key not in after or not torch.equal(before[key],after[key]):
            raise ValueError('Initial scene mismatch: '+key)
    if not torch.equal(before['boxes'],parked['observed']):
        raise ValueError('Observed initial boxes mismatch')
    obs = torch.load(reference/'observed_scene.pt',map_location='cpu')
    if not torch.equal(obs,parked['observed_scene']):
        raise ValueError('Initial full observation mismatch')
    physics = json.loads((path/'physics_only_verification.json').read_text())
    conditions = json.loads((path/'new_conditions.json').read_text())
    if physics['checked_frames']!=steps+1 or not conditions['conditions_verified'] or conditions['near50']:
        raise ValueError('FSM protocol checks missing')
    proof = result['stage1_policy']
    if not all(proof[k] for k in ('native_actor_exact','normalization_exact','weights_unchanged','normalization_unchanged')) or proof['coordination_calls']:
        raise ValueError('Original Stage1 policy changed')
    if result['encoder_attention_checks']!=dict(actor=steps+1,critic=steps):
        raise ValueError('Unexpected FSM encoder call counts')
    fsm = result['fsm']
    if fsm['max_frozen_root_error'] or fsm['max_frozen_dof_error']:
        raise ValueError('FSM freeze projection failed')
    state = CoordinationState(after['predecessor'],after['groups'])
    with np.load(path/'fsm_trace.npz') as archive:
        trace = {key:archive[key] for key in archive.files}
        if len(trace['frame'])!=steps-1:
            raise ValueError('Missing FSM control frames')
        for i in range(len(trace['frame'])):
            expected = state.update(torch.from_numpy(trace['leader_success'][i]),torch.from_numpy(trace['touching'][i]),
                                    torch.from_numpy(trace['alive'][i]))
            if not np.array_equal(expected.numpy(),trace['frozen'][i]) or not np.array_equal(state.released.numpy(),trace['released'][i]):
                raise ValueError('CPU FSM state replay mismatch')
            if not np.array_equal((trace['hand_distance'][i]<.12).all(-1),trace['touching'][i]):
                raise ValueError('FSM hand contact threshold mismatch')
        masks = trace['frozen'].copy()
    with np.load(path/'trace.npz') as trace:
        movement = np.abs(trace['roots'][1:]-trace['roots'][:-1])
        frozen_error = float(movement[masks].max()) if masks.any() else 0.
        first_step_distance = np.linalg.norm(trace['roots'][0]-after['roots'][...,:3].numpy(),axis=-1)
        maximum_initial_motion = float(first_step_distance.max())
        if maximum_initial_motion>.5:
            raise ValueError('First physics frame lost sampled initial state: '+str(maximum_initial_motion))
    if frozen_error:
        raise ValueError('Frozen human moved in independent formal trajectory')
    score = rescore(path)
    if not all(score[k] for k in ('finite','cpu_gpu_exact','json_counts_exact','errors_match_recorded')):
        raise ValueError('CPU formal scoring mismatch')
    checks = dict(initial_states_exact=True,initial_observation_exact=True,physical_removal_checks=steps+1,
        cpu_gpu_sr_exact=True,cpu_gpu_errors_match=True,cpu_fsm_replay_exact=True,native_stage1_unchanged=True,
        frozen_root_movement_m=frozen_error,max_first_step_root_displacement_m=maximum_initial_motion,
        seed=result['seed'],agents=people,envs=envs,steps=steps)
    if (path/'table2.json').exists():
        replayed = replay(path)
        (path/'table2_recomputed.json').write_text(json.dumps(replayed,indent=2))
        checks['cpu_gpu_table2_exact'] = True
    (path/'verification.json').write_text(json.dumps(checks,indent=2))
    return checks
