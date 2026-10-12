"""Evaluate a fixed policy across box mass/friction overrides at actor construction.

The checkpoint contract is checked normally; audit-only actor properties are explicit.
No external forces or optimizer steps are used. Cases share initial states.
"""
from isaacgym import gymapi
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import runpy
import torch
from learning.multi_agent.ma_players import MAPlayerContinuous
from utils.door_reference import rotate
from utils.open_push_scene import sample_goal_offsets
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush


def numbers(name, default):
    return [float(x) for x in os.environ.get(name, default).split(',')]


MASSES = numbers('CAPACITY_MASSES', '5,10,12,15,20,25,30,40,60,100,219.7')
FRICTIONS = numbers('CAPACITY_FRICTIONS', '.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0')
SCENES = int(os.environ.get('CAPACITY_SCENES', '2'))
CONTACT_REPEATS = int(os.environ.get('CAPACITY_CONTACT_REPEATS', '3'))
WALK_REPEATS = int(os.environ.get('CAPACITY_WALK_REPEATS', '2'))
CONTACT_STEPS = int(os.environ.get('CAPACITY_CONTACT_STEPS', '300'))
WALK_STEPS = int(os.environ.get('CAPACITY_WALK_STEPS', '600'))
CASES = list(itertools.product(FRICTIONS, MASSES))
assert SCENES > 0 and all(0 < x <= 1 for x in FRICTIONS) and all(x > 0 for x in MASSES)


original_build_box = HumanoidMAOpenPush._build_box


def build_capacity_box(self, env_id, object_id, env_ptr):
    original_build_box(self, env_id, object_id, env_ptr)
    if object_id < 2:
        return
    friction, mass = CASES[env_id // SCENES]
    handle = self._box_handles[-1]
    bodies = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
    bodies[0].mass = mass
    self.gym.set_actor_rigid_body_properties(env_ptr, handle, bodies, True)
    shapes = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
    for shape in shapes:
        shape.friction = friction
    self.gym.set_actor_rigid_shape_properties(env_ptr, handle, shapes)


HumanoidMAOpenPush._build_box = build_capacity_box


def run_capacity(self):
    task = self.env.task
    assert task.num_envs == len(CASES) * SCENES and task.num_agents == 2
    assert task.cfg['args'].test and task.cfg['args'].eval
    assert task.cfg['env']['enableTrackInitState'] is False
    self._reward_debug = False
    device = task.device
    ids = torch.arange(task.num_envs, device=device)
    pattern = ids % SCENES
    task._task_graph_preset = 'push'
    masses = torch.tensor([m for _, m in CASES], device=device).repeat_interleave(SCENES)
    frictions = torch.tensor([f for f, _ in CASES], device=device).repeat_interleave(SCENES)
    actual = []
    for e in range(task.num_envs):
        for b in (2, 3):
            handle = task._box_handles[4 * e + b]
            props = task.gym.get_actor_rigid_body_properties(task.envs[e], handle)
            assert len(props) == 1
            shapes = task.gym.get_actor_rigid_shape_properties(task.envs[e], handle)
            measured = task.gym.get_actor_rigid_body_properties(task.envs[e], handle)[0].mass
            assert abs(measured - float(masses[e])) < .001
            assert all(abs(p.friction - float(frictions[e])) < 1e-5 for p in task.gym.get_actor_rigid_shape_properties(task.envs[e], handle))
            actual.append(dict(env=e, box=b, mass_kg=measured, friction=float(frictions[e])))
    out = Path(task.cfg['args'].output_path)
    out.mkdir(parents=True, exist_ok=True)
    bank = task._joint_rsi_bank.rows['push']
    trials = []
    local_corners = masses.new_tensor([[-.65,-.65,-.65],[-.65,.65,-.65],[.65,-.65,-.65],[.65,.65,-.65]])
    for mode, repeats, horizon in [('contact_straight', CONTACT_REPEATS, CONTACT_STEPS), ('walk_random_goal', WALK_REPEATS, WALK_STEPS)]:
        for repeat in range(repeats):
            torch.manual_seed(42 + repeat)
            self.env_reset()
            # Duplicate base scene rows across all mass/friction cells.
            for name in ('_agent_lanes','_push_binding','_box_states','_push_reset_states','_push_directions',
                         '_push_goal_offsets','_humanoid_root_states','_dof_pos','_dof_vel',
                         '_kinematic_humanoid_rigid_body_states','_every_env_init_dof_pos','_tar_pos'):
                value = getattr(task, name)
                value[:] = value[pattern].clone()
            if mode == 'contact_straight':
                task._agent_lanes[:] = torch.tensor([0,1], device=device)
                task._push_binding[:] = task._agent_lanes
                task._box_states[:,2:] = 0.
                task._box_states[:,2:,2] = .65
                task._box_states[:,2:,6] = 1.
                task._box_states[:,2:,0] = -4.
                task._box_states[:,2,1] = -5.
                task._box_states[:,3,1] = 5.
                task._push_reset_states[:] = task._box_states[:,2:]
                task._push_directions[:] = masses.new_tensor([0.,1.,0.])
                task._push_goal_offsets[:] = masses.new_tensor([0.,2.,0.])
            else:
                base_boxes = task._push_reset_states[:SCENES]
                goals = sample_goal_offsets(base_boxes, {'half_extent':1.5,'min_distance':1.})
                for _ in range(64):
                    distance = (base_boxes[...,:2] + goals[...,:2] - task._humanoid_root_states[:SCENES,:,:2]).norm(dim=-1)
                    bad = distance < 1.
                    if not bad.any():
                        break
                    new = sample_goal_offsets(base_boxes, {'half_extent':1.5,'min_distance':1.})
                    goals[bad] = new[bad]
                assert (distance >= 1.).all()
                task._push_goal_offsets[:] = goals[pattern]
            task._sample_box_assignments(ids)
            task.relation_runtime.reset(ids, torch.ones(task.num_envs,2,device=device,dtype=torch.long), task._owner_push_boxes())
            if mode == 'contact_straight':
                for agent in (0,1):
                    selected = [bank[(s*97+repeat*173+agent*37) % len(bank)] for s in range(SCENES)]
                    sample = {k: torch.tensor([r[k] for r in selected],device=device)[pattern] for k in selected[0]}
                    sample['box_progress'] = torch.zeros(task.num_envs,device=device)
                    task._apply_joint_rsi(ids, torch.full_like(ids,agent), 'push', sample)
            task._reset_task(ids)
            task.relation_runtime.reset_push_credit(ids, task._owner_push_boxes(), task._tar_pos)
            task._push_episode_start[:] = task._owner_push_boxes()[...,:3]
            task._reset_env_tensors(ids)
            task._sync_door_reset_states(ids)
            task._rigid_body_state.view(task.num_envs,-1,13)[:,:2*task.num_bodies] = task._kinematic_humanoid_rigid_body_states.reshape(task.num_envs,2*task.num_bodies,13)
            task._compute_observations(ids)
            task._init_amp_obs(ids)
            # Exact matched starts, goal, box yaw and joint state across grid cells.
            for value in (task._owner_push_boxes(), task._humanoid_root_states, task._dof_pos, task._tar_pos):
                torch.testing.assert_close(value, value[pattern], atol=1e-5, rtol=1e-5)
            obs = self.obs_to_torch(self.env._policy_observation())
            self.get_batch_size(obs['obs'], 1)
            start = task._owner_push_boxes().clone()
            initial_corners = start[:,:,:3,None].transpose(-1,-2) + rotate(start[:,:,None,3:7].expand(-1,-1,4,-1), local_corners.expand(task.num_envs,2,-1,-1))
            delta = task._tar_pos - start[...,:3]
            axis = delta / delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            shape = (task.num_envs,2)
            alive = torch.ones(task.num_envs,device=device,dtype=torch.bool)
            tipped = torch.zeros(shape,device=device,dtype=torch.bool)
            max_tilt = torch.zeros(shape,device=device)
            max_slide = torch.zeros_like(max_tilt)
            final_slide = torch.zeros_like(max_tilt)
            peak_reaction = torch.zeros_like(max_tilt)
            contact_steps = torch.zeros_like(max_tilt)
            reaction_sum = torch.zeros_like(max_tilt)
            reached = torch.zeros_like(tipped)
            for step in range(horizon):
                action = self.get_action(obs, self.is_determenistic)
                raw, _, done, _ = self.env.step(action)
                obs = self.obs_to_torch(raw)
                assert torch.isfinite(raw).all()
                boxes = task._owner_push_boxes()
                q = boxes[...,3:7]
                tilt = torch.acos((1-2*(q[...,0].square()+q[...,1].square())).clamp(-1,1))
                valid = alive[:,None]
                max_tilt = torch.maximum(max_tilt, torch.where(valid,tilt,0.))
                tipped |= valid & (tilt > .35)
                corners = boxes[:,:,None,:3] + rotate(q[:,:,None].expand(-1,-1,4,-1),local_corners.expand(task.num_envs,2,-1,-1))
                slide = ((corners-initial_corners)*axis[:,:,None]).sum(-1).amin(-1)
                grounded = (boxes[...,2]-start[...,2]).abs()<=.08
                upright = grounded & (tilt<=.35) & ~tipped & valid
                max_slide = torch.maximum(max_slide,torch.where(upright,slide,0.))
                final_slide = torch.where(valid,slide,final_slide)
                goal_error = (boxes[...,:2]-task._tar_pos[...,:2]).norm(dim=-1)
                reached |= upright & (goal_error<=.15)
                normal = rotate(q,boxes.new_tensor([0.,1.,0.]).expand_as(boxes[...,:3]))
                force = task._contact_forces[:,:,task._key_body_ids[:2]]
                reaction = (-(force*normal[:,:,None]).sum(-1)).clamp_min(0).sum(-1)
                contact = task.relation_runtime.push_contact & valid
                peak_reaction = torch.maximum(peak_reaction,torch.where(contact,reaction,0.))
                contact_steps += contact
                reaction_sum += reaction*contact
                alive &= ~done.reshape(task.num_envs,2).bool().any(-1)
                if not alive.any():
                    break
                if step % 100 == 0:
                    print('CAPACITY_PROGRESS',mode,repeat,step,horizon,flush=True)
            for index,(friction,mass) in enumerate(CASES):
                e=slice(index*SCENES,(index+1)*SCENES)
                for scene in range(SCENES):
                    for agent in (0,1):
                        row=index*SCENES+scene
                        n=float(contact_steps[row,agent])
                        trials.append(dict(mode=mode,repeat=repeat,scene=scene,agent=agent,mass_kg=mass,friction=friction,
                            max_slide_before_tip_m=float(max_slide[row,agent]),final_projected_corner_slide_m=float(final_slide[row,agent]),
                            tipped=bool(tipped[row,agent]),max_tilt_deg=float(max_tilt[row,agent]*180/torch.pi),
                            upright_goal_reached=bool(reached[row,agent]),peak_contact_reaction_N=float(peak_reaction[row,agent]),
                            mean_contact_reaction_N=float(reaction_sum[row,agent])/max(1,n),contact_seconds=n*task.dt))
            (out/'trials.json').write_text(json.dumps(trials,indent=2)+'\n')
            print('CAPACITY_TRIAL_DONE',mode,repeat,'rows',len(trials),flush=True)
    summary=[]
    for mode in ('contact_straight','walk_random_goal'):
        for friction,mass in CASES:
            rows=[r for r in trials if r['mode']==mode and r['friction']==friction and r['mass_kg']==mass]
            if not rows:
                continue
            summary.append(dict(mode=mode,friction=friction,mass_kg=mass,trials=len(rows),
                stable_slide_10cm_rate=sum(r['max_slide_before_tip_m']>=.1 and not r['tipped'] for r in rows)/len(rows),
                stable_slide_50cm_rate=sum(r['max_slide_before_tip_m']>=.5 and not r['tipped'] for r in rows)/len(rows),
                stable_slide_1m_rate=sum(r['max_slide_before_tip_m']>=1. and not r['tipped'] for r in rows)/len(rows),
                slide_50cm_before_tip_rate=sum(r['max_slide_before_tip_m']>=.5 for r in rows)/len(rows),
                tipping_rate=sum(r['tipped'] for r in rows)/len(rows),
                upright_goal_rate=sum(r['upright_goal_reached'] for r in rows)/len(rows),
                mean_max_stable_slide_m=sum(r['max_slide_before_tip_m'] for r in rows)/len(rows),
                max_stable_slide_m=max(r['max_slide_before_tip_m'] for r in rows),
                max_contact_reaction_N=max(r['peak_contact_reaction_N'] for r in rows)))
    args=task.cfg['args']
    report=dict(training=False,gpu=os.environ.get('TOKENHSI_GPU'),checkpoint=args.checkpoint,
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        masses=MASSES,frictions=FRICTIONS,scenes_per_cell=SCENES,contact_repeats=CONTACT_REPEATS,walk_repeats=WALK_REPEATS,
        contact_seconds=CONTACT_STEPS*task.dt,walk_seconds=WALK_STEPS*task.dt,
        original_checkpoint_contract=task.get_open_push_contract(),actor_properties=actual,matched_starts=True,external_force=False,
        notes='Audit-only actor-construction property overrides with normal checkpoint contract validation. GPU PhysX runtime shape-friction setters did not affect measured forces, so properties are set before prepare_sim. Stable slide tracks minimum displacement of four corresponding bottom corners projected toward the initial goal, before any tilt over20deg, with ground height tolerance8cm. Stable rate also requires no tip during the episode. Contact starts use source screened poses at box progress0 and straight2m goal. Walk starts use paired source locomotion states and independent random XY goal. A measured policy benchmark, not maximum actuator capacity or retrained policy.',summary=summary)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print('CAPACITY_COMPLETE',str(out/'audit.json'),flush=True)


MAPlayerContinuous.run_eval = run_capacity
runpy.run_path('tokenhsi/run.py', run_name='__main__')
