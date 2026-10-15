"""Verify explicit evaluation mass constructed before GPU simulation preparation."""
import math
import json
from pathlib import Path


def apply_mass_preview(task,mass):
    mass=float(mass)
    if not math.isfinite(mass) or mass<=0:
        raise ValueError('Preview mass must be finite and positive')
    if not getattr(task,'_joint_push',False) or not task.cfg['args'].test or not task._is_eval or getattr(task,'_three_agent_preview',False):
        raise ValueError('Mass preview requires an ordinary JointPush evaluation player')
    if getattr(task,'_mass_preview_constructed',None)!=mass:
        raise ValueError('Mass preview must be applied during asset creation before GPU simulation preparation')
    after=[]
    for env_id,env in enumerate(task.envs):
        handle=task._box_handles[env_id*task.num_objects+2]
        props=task.gym.get_actor_rigid_body_properties(env,handle)
        if len(props)!=1:raise ValueError('Expected single-body shared payload')
        measured=props[0].mass
        if abs(measured-mass)>.001:raise RuntimeError('Constructed preview mass mismatch')
        after.append(measured)
    inactive=task.gym.get_actor_rigid_body_properties(task.envs[0],task._box_handles[3])[0].mass
    if abs(inactive-task.cfg['env']['pushBox']['mass'])>.001:
        raise RuntimeError('Inactive object mass must remain at checkpoint value')
    report=dict(mode='evaluation_mass_only_at_asset_creation',checkpoint_mass=task.cfg['env']['pushBox']['mass'],
                actual_mass_min=min(after),actual_mass_max=max(after),inactive_box_mass=inactive,inertia_scaled=True,application_stage="before_prepare_sim",
                untouched='checkpoint config, observation, goal, reward, friction, model')
    directory=Path(task.cfg['args'].output_path);directory.mkdir(parents=True,exist_ok=True)
    (directory/'mass_preview.json').write_text(json.dumps(report,indent=2)+'\n')
    print('[JointPush MASS DIAGNOSTIC]',report,flush=True)
    return report
