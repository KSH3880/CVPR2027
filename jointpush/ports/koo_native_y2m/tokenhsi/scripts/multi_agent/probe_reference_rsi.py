"""Short headless physics probe; teacher weights are used only for actions.

No optimizer steps, KL loss, teacher state collection, or training checkpoint.
Uses the target 4-object task and training RSI, not loco-only eval resets.
"""
import os
import sys
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'tokenhsi')]
visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if not visible or ',' in visible:
    raise RuntimeError('Probe must explicitly select physical GPU 6 or 7')
physical = subprocess.check_output(['nvidia-smi','-i',visible,'--query-gpu=index',
                                   '--format=csv,noheader,nounits'],text=True).strip()
if physical not in ('6','7'):
    raise RuntimeError('Resolved GPU is outside allowed GPUs 6 and 7')
import isaacgym
import torch
import numpy as np
import json
import csv
from collections import defaultdict
from learning.multi_agent.ma_agent import MAAgent
from utils.reference_rsi import digest, interpolate_object_velocity


def configure_impact_sweep(task):
    """Probe-only raw reference sampling; no 1cm bank filter or loco fallback.

    Explicitly overrides original skipped intervals to examine otherwise excluded
    late states. Existing scene placement validators still apply.
    """
    lo, hi = map(float, os.environ.get('RSI_PROBE_PHASE', '0.65,0.98').split(','))
    if not 0 <= lo < hi <= 1:
        raise ValueError('RSI_PROBE_PHASE must satisfy 0 <= lo < hi <= 1')
    for name, lib in task._motion_lib.items():
        if name == 'loco':
            continue
        def sample(ids, truncate_time=None, lib=lib):
            return (lo+(hi-lo)*torch.rand(ids.shape, device=task.device))*lib.get_motion_length(ids)
        lib.sample_time_rsi = sample
    for section in ('hard_skill_training', 'independent_training'):
        settings = task._relation_cfg.get(section, {}).get('climb_rsi')
        if settings is not None:
            settings['late_fraction'] = 0.
    original_reset = task._reset_boxes
    def reset_boxes(env_ids):
        original_reset(env_ids)
        for name in ('pickUp', 'carryWith', 'putDown'):
            slots = task._reset_ref_slots.get(name)
            if slots is None:
                continue
            e, a = slots
            box = task._scenario_physical_object(e, a)
            lib = task._motion_lib[name]
            ids = task._reset_ref_motion_ids[name]
            times = task._reset_ref_motion_times[name]
            pos, _ = lib.get_obj_motion_state(ids, times)
            offset = task._agent_spawn_offsets[a]+task._env_origins[e]
            linear, angular = interpolate_object_velocity(lib, ids, times)
            floor = task._box_size[e, box, 2]/2 > pos[:, 2]+offset[:, 2]
            linear[floor, 2] = 0.
            task._box_states[e, box, 7:10] = linear
            task._box_states[e, box, 10:13] = angular
    task._reset_boxes = reset_boxes
    return [lo, hi]


@torch.no_grad()
def probe(self):
    task = self.vec_env.env.task
    student = os.environ.get('RSI_STUDENT_CHECKPOINT')
    teacher = student or os.environ.get('RSI_TEACHER', str(ROOT/'distill_pth/ApproachScenarioStage1RescueAtKLClimb50_00008000.pth'))
    checkpoint = torch.load(teacher, map_location=self.ppo_device, weights_only=False)
    # Explicit probe-only weight load: the teacher reward/3-object metadata is
    # not a contract for this 4-object collision test. Training loaders stay strict.
    if student:
        # Normal task/reward/RSI checkpoint checks before probe-only replay hooks.
        self.set_weights(checkpoint)
    self.model.load_state_dict(checkpoint['model'], strict=True)
    self.set_stats_weights(checkpoint)
    self.set_eval()
    self.rnn_states = None
    mode = os.environ.get('RSI_PROBE_MODE', 'baseline')
    phase_range = configure_impact_sweep(task) if mode == 'impact_sweep' else None
    replay = None
    if mode == 'candidate_replay':
        if not student:
            raise ValueError('Candidate replay requires student checkpoint')
        from utils.impact_candidates import ProbeCandidateReplay
        asset = ROOT/task.cfg['env']['asset']['assetRoot']/task.cfg['env']['asset']['assetFileName']
        replay = ProbeCandidateReplay(os.environ['RSI_CANDIDATES'],task,
                                     ROOT/'tokenhsi/data/dataset_loco_sit_carry_climb.yaml',asset)
        task._reference_rsi_bank = replay
    if mode == 'raw_time_late':
        task._relation_cfg['hard_skill_training']['climb_rsi']['late_fraction'] = .7
    rounds = int(os.environ.get('RSI_PROBE_ROUNDS', '4'))
    steps = int(os.environ.get('RSI_PROBE_STEPS', '10'))
    if rounds < 1 or steps < 1:
        raise ValueError('Probe rounds/steps must be positive')
    ids = torch.arange(task.num_envs, device=task.device)
    rows = []
    agent_rows = []
    partial_reset_checked = False
    for repeat in range(rounds):
        obs = self.env_reset()
        initial_sizes = task._box_size.clone()
        initial_states = task._box_states.clone()
        physical = task._scenario_physical_object(ids.repeat_interleave(2),
                    torch.arange(2,device=task.device).repeat(task.num_envs))
        size = task._box_size[ids.repeat_interleave(2), physical]
        skill = torch.full((task.num_envs,2), -1, device=task.device,dtype=torch.long)
        phase = torch.full_like(skill, -1,dtype=torch.float32)
        motion_id = torch.full_like(skill, -1)
        for name, slots in task._reset_ref_slots.items():
            e,a = slots
            skill[e,a] = task._skill.index(name)
            motion_id[e,a] = task._reset_ref_motion_ids[name]
            lib = task._motion_lib[name]
            phase[e,a] = task._reset_ref_motion_times[name] / lib.get_motion_length(task._reset_ref_motion_ids[name])
        max_box_speed = task._box_states[...,7:10].norm(dim=-1).amax(-1)
        max_root_speed = task._humanoid_root_states[...,7:10].norm(dim=-1).amax(-1)
        max_contact = torch.zeros(task.num_envs,device=task.device)
        max_delta_v = torch.zeros_like(max_contact)
        max_displacement = torch.zeros_like(max_contact)
        done_step = torch.full((task.num_envs,),steps+1,device=task.device,dtype=torch.long)
        active = torch.ones(task.num_envs,device=task.device,dtype=torch.bool)
        previous_v = task._box_states[...,7:10].clone()
        early_delta = torch.zeros_like(task._box_states[..., 0])
        object_delta = torch.zeros_like(early_delta)
        object_speed = previous_v.norm(dim=-1)
        object_angular = task._box_states[...,10:13].norm(dim=-1)
        reference_error = torch.zeros_like(early_delta)
        upward_displacement = torch.zeros_like(early_delta)
        first2_vertical_delta = torch.zeros_like(early_delta)
        initial_rotation = initial_states[...,3:7]
        max_rotation = torch.zeros_like(early_delta)
        initial_root_height = task._humanoid_root_states[...,2].clone()
        max_root_drop = torch.zeros_like(initial_root_height)
        def hand_distances(initial=False):
            # Gym rigid-body tensors can still contain the previous pose immediately
            # after indexed root/DOF reset. Initial reference FK is the valid baseline.
            positions = (task._kinematic_humanoid_rigid_body_states[..., :3] if initial else
                         task._rigid_body_pos.reshape(task.num_envs,task.num_agents,task.num_bodies,3))
            hands = positions[:,:,task._key_body_ids[[0,1]]]
            boxes = task._box_states[ids.repeat_interleave(2),physical,:3].reshape(task.num_envs,2,3)
            return (hands-boxes[:,:,None]).norm(dim=-1).mean(-1)
        initial_hand_distance = hand_distances(initial=True)
        max_hand_distance_increase = torch.zeros_like(initial_hand_distance)
        finite = torch.ones_like(active)
        for step in range(steps):
            action = self.get_action_values(obs,torch.zeros(task.num_envs*task.num_agents,device=self.ppo_device))['mus']
            obs,reward,done,info = self.env_step(action)
            now = task._box_states[...,7:10]
            dv = (now-previous_v).norm(dim=-1)
            object_delta[active] = torch.maximum(object_delta[active], dv[active])
            if step < 2:
                early_delta[active] = torch.maximum(early_delta[active], dv[active])
                first2_vertical_delta[active] = torch.maximum(first2_vertical_delta[active],
                    (now[active,...,2]-previous_v[active,...,2]).abs())
            upward_displacement[active] = torch.maximum(upward_displacement[active],
                task._box_states[active,...,2]-initial_states[active,...,2])
            angle = 2*torch.acos((task._box_states[...,3:7]*initial_rotation).sum(-1).abs().clamp(0,1))
            max_rotation[active] = torch.maximum(max_rotation[active],angle[active])
            max_root_drop[active] = torch.maximum(max_root_drop[active],
                initial_root_height[active]-task._humanoid_root_states[active,...,2])
            max_hand_distance_increase[active] = torch.maximum(max_hand_distance_increase[active],
                hand_distances()[active]-initial_hand_distance[active])
            object_speed[active] = torch.maximum(object_speed[active], now[active].norm(dim=-1))
            object_angular[active] = torch.maximum(object_angular[active], task._box_states[active,...,10:13].norm(dim=-1))
            for name in ('pickUp', 'carryWith', 'putDown'):
                slots = task._reset_ref_slots.get(name)
                if slots is None:
                    continue
                e,a = slots
                box = task._scenario_physical_object(e,a)
                lib = task._motion_lib[name]
                mids = task._reset_ref_motion_ids[name]
                times = torch.minimum(task._reset_ref_motion_times[name]+(step+1)*task.dt,
                                      lib.get_motion_length(mids))
                expected,_ = lib.get_obj_motion_state(mids,times)
                expected = expected+task._agent_spawn_offsets[a]+task._env_origins[e]
                expected[:,2] = torch.maximum(expected[:,2],task._box_size[e,box,2]/2)
                error = (task._box_states[e,box,:3]-expected).norm(dim=-1)
                keep = active[e]
                reference_error[e[keep],box[keep]] = torch.maximum(reference_error[e[keep],box[keep]],error[keep])
            max_box_speed[active] = torch.maximum(max_box_speed[active],now[active].norm(dim=-1).amax(-1))
            max_root_speed[active] = torch.maximum(max_root_speed[active],task._humanoid_root_states[active,...,7:10].norm(dim=-1).amax(-1))
            max_delta_v[active] = torch.maximum(max_delta_v[active],(now[active]-previous_v[active]).norm(dim=-1).amax(-1))
            max_contact[active] = torch.maximum(max_contact[active],task._contact_forces[active].norm(dim=-1).flatten(1).amax(-1))
            max_displacement[active] = torch.maximum(max_displacement[active],
                (task._box_states[active,...,:3]-initial_states[active,...,:3]).norm(dim=-1).amax(-1))
            state_finite = torch.isfinite(task._box_states).flatten(1).all(-1) & torch.isfinite(task._humanoid_root_states).flatten(1).all(-1)
            finite &= state_finite
            ended = done.reshape(task.num_envs,task.num_agents).bool().any(-1)
            done_step[active & ended] = step+1
            active &= ~ended
            previous_v = now.clone()
        for env in range(task.num_envs):
            rows.append({'round':repeat,'env':env,'sizes':size.reshape(task.num_envs,2,3)[env].cpu().tolist(),
                'skill':[task._skill[int(v)] if v>=0 else 'default' for v in skill[env]],
                'phase':phase[env].cpu().tolist(),'max_box_speed':float(max_box_speed[env]),
                'max_root_speed':float(max_root_speed[env]),'max_box_delta_v':float(max_delta_v[env]),
                'max_human_contact_force':float(max_contact[env]),'max_box_displacement':float(max_displacement[env]),
                'done_step':int(done_step[env]),'finite':bool(finite[env])})
            for agent in range(task.num_agents):
                box = int(physical.reshape(task.num_envs,2)[env,agent])
                agent_rows.append({'round':repeat,'env':env,'agent':agent,
                    'size_cm':[round(v*100) for v in size.reshape(task.num_envs,2,3)[env,agent].cpu().tolist()],
                    'skill':rows[-1]['skill'][agent], 'phase':float(phase[env,agent]),
                    'motion_id':int(motion_id[env,agent]),
                    'first_two_steps_delta_v':float(early_delta[env,box]),
                    'max_box_delta_v':float(object_delta[env,box]),
                    'max_box_speed':float(object_speed[env,box]),
                    'max_box_angular_speed':float(object_angular[env,box]),
                    'first2_vertical_delta_v':float(first2_vertical_delta[env,box]),
                    'max_upward_displacement_m':float(upward_displacement[env,box]),
                    'max_rotation_from_initial_rad':float(max_rotation[env,box]),
                    'max_root_height_drop_m':float(max_root_drop[env,agent]),
                    'max_hand_box_center_distance_increase_m':float(max_hand_distance_increase[env,agent]),
                    'max_reference_position_error':float(reference_error[env,box]) if rows[-1]['skill'][agent] in ('pickUp','carryWith','putDown') else None,
                    'scene_done_step':int(done_step[env]),'scene_finite':bool(finite[env])})
                row = agent_rows[-1]
                row['candidate_id'] = (replay.match(row['size_cm'],row['skill'],row['motion_id'],row['phase'])
                                       if replay else None)
        torch.testing.assert_close(task._box_size,initial_sizes)
        # Verify new reset path leaves non-reset scenes and AMP history untouched.
        if not partial_reset_checked:
            retained = torch.arange(1,task.num_envs,2,device=task.device)
            reset = torch.arange(0,task.num_envs,2,device=task.device)
            before = [x[retained].clone() for x in (task._box_states,task._humanoid_root_states,task._amp_obs_buf,task._logical_box_order)]
            task.reset(reset)
            for x,old in zip((task._box_states,task._humanoid_root_states,task._amp_obs_buf,task._logical_box_order),before):
                torch.testing.assert_close(x[retained],old)
            partial_reset_checked = True
    def quantiles(key):
        return {str(q):float(np.quantile([r[key] for r in rows],q)) for q in (.5,.95,.99,1.)}
    summary = {'mode':mode,'phase_range':phase_range,'teacher':teacher,'teacher_sha256':digest(teacher),
        'policy_role':'student' if student else 'distill_teacher',
        'measurement_version':2,
        'hand_distance_initial_basis':'reset_reference_kinematic_state',
        'policy_checkpoint':teacher,'policy_sha256':digest(teacher),
        'candidate_replay_matched_agents':sum(r['candidate_id'] is not None for r in agent_rows),
        'teacher_saved_objects':checkpoint.get('relation_experiment_config',{}).get('num_objects'),
        'target_objects':task.num_objects,'envs':task.num_envs,'rounds':rounds,'steps':steps,
        'reference_bank_sha256':None if task._reference_rsi_bank is None else task._reference_rsi_bank.sha256,
        'n_scenes':len(rows),'partial_reset_checked':partial_reset_checked,
        'nonfinite_scenes':sum(not r['finite'] for r in rows),
        'early_done_fraction':sum(r['done_step']<=steps for r in rows)/len(rows),
        'box_speed_gt_5_fraction':sum(r['max_box_speed']>5 for r in rows)/len(rows),
        'box_delta_v_gt_3_fraction':sum(r['max_box_delta_v']>3 for r in rows)/len(rows),
        'box_speed_m_per_s':quantiles('max_box_speed'),'box_delta_v_m_per_s':quantiles('max_box_delta_v'),
        'human_contact_force_N':quantiles('max_human_contact_force'),
        'sampler_attempt_stats':None if task._reference_rsi_bank is None else task._reference_rsi_bank.stats,
        'note':'Thresholds are diagnostics, not proven explosions or task success. No training or teacher state pool.'}
    target = Path(os.environ['RSI_PROBE_OUTPUT']).resolve()
    target.mkdir(parents=True,exist_ok=True)
    if replay:
        candidate_groups = defaultdict(list)
        for r in agent_rows:
            if r['candidate_id'] is not None:
                candidate_groups[r['candidate_id']].append(r)
        result = []
        for c in replay.payload['candidates']:
            values = candidate_groups[c['id']]
            result.append({**c,'student_n':len(values),
                'student_first2_delta_v_max':max((r['first_two_steps_delta_v'] for r in values),default=None),
                'student_first2_delta_v_gt2_count':sum(r['first_two_steps_delta_v']>2 for r in values),
                'student_first2_vertical_delta_v_max':max((r['first2_vertical_delta_v'] for r in values),default=None),
                'student_upward_displacement_max_m':max((r['max_upward_displacement_m'] for r in values),default=None),
                'student_angular_speed_max':max((r['max_box_angular_speed'] for r in values),default=None),
                'student_hand_box_distance_increase_max_m':max((r['max_hand_box_center_distance_increase_m'] for r in values),default=None),
                'student_root_height_drop_max_m':max((r['max_root_height_drop_m'] for r in values),default=None),
                'status':'unmeasured' if not values else 'observed_not_safety_certified'})
        (target/'candidate_validation.json').write_text(json.dumps(result,indent=2))
    (target/'summary.json').write_text(json.dumps(summary,indent=2))
    with (target/'scenes.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row)+'\n')
    with (target/'agents.jsonl').open('w') as f:
        for row in agent_rows:
            f.write(json.dumps(row)+'\n')
    groups = defaultdict(list)
    for row in agent_rows:
        if row['phase'] < 0:
            continue
        groups[(tuple(row['size_cm']),row['skill'],int(row['phase']*10))].append(row)
    with (target/'by_size_skill_phase.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x_cm','y_cm','z_cm','skill','phase_lo','phase_hi','n',
                         'first2_delta_v_p50','first2_delta_v_p95','first2_delta_v_max',
                         'first2_delta_v_gt3_fraction','speed_gt5_fraction',
                         'scene_early_done_fraction','reference_error_p95_m'])
        for (size,name,bin_id),values in sorted(groups.items()):
            v = [row['first_two_steps_delta_v'] for row in values]
            err = [row['max_reference_position_error'] for row in values if row['max_reference_position_error'] is not None]
            writer.writerow([*size,name,bin_id/10,(bin_id+1)/10,len(values),
                *[float(np.quantile(v,q)) for q in (.5,.95,1)],
                sum(x>3 for x in v)/len(v),sum(row['max_box_speed']>5 for row in values)/len(values),
                sum(row['scene_done_step']<=steps for row in values)/len(values),
                float(np.quantile(err,.95)) if err else ''])
    print('RSI_PROBE',json.dumps(summary),flush=True)


MAAgent.train = probe
import run
run.main()
