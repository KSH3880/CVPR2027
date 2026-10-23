"""Full training RSI resets without policy actions or optimizer steps."""
from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tokenhsi')]
import isaacgym
from isaacgym import gymtorch
import torch
import numpy as np
from env.tasks.multi_agent.humanoid_chair_mixed_ft import HumanoidChairMixedFT
from learning.multi_agent.chair_ft_agent import ChairFTPlayer
import utils.parse_task as parser


class RsiValidationTask(HumanoidChairMixedFT):
    def __init__(self,cfg,*args,**kwargs):
        saved=cfg['args'].eval
        try:
            cfg['args'].eval=False
            super().__init__(cfg,*args,**kwargs)
        finally:cfg['args'].eval=saved


class RsiValidationPlayer(ChairFTPlayer):
    @torch.no_grad()
    def run_eval(self):
        self.env_reset();t=self.env.task;out=Path(t.cfg['args'].output_path)
        report=json.loads((out/'rsi_reset_validation.json').read_text());report['optimizer_steps']=0;report['policy_actions']=0
        # Check the same complete accepted reset again, independent of retry counters.
        rows=torch.arange(t.num_envs,device=t.device)
        report['accepted_reset_foot_crossings']=int(t._native_foot_guard.rejected(rows).sum())
        from utils.chair_front import front_allowed
        r=rows.repeat_interleave(t.num_agents);o=torch.arange(t.num_agents,device=t.device).repeat(t.num_envs);p=t._scenario_physical_object(r,o);chair=t._chair_kinds[r,p]==2;r,o,p=r[chair],o[chair],p[chair]
        from isaacgym.torch_utils import quat_rotate
        inv=t._box_states[r,p,3:7].clone();inv[:,:3]*=-1;local=quat_rotate(inv,t._humanoid_root_states[r,o,:3]-t._box_states[r,p,:3])
        report['chair_front_violations']=int((~front_allowed(local,t._chair_seats[r,p],-t._chair_features[r,p,1:3])).sum())
        assert report['accepted_reset_foot_crossings']==0 and report['chair_front_violations']==0
        initial=t._kinematic_humanoid_rigid_body_states.clone();targets=t._dof_pos.contiguous().clone();t.gym.set_dof_position_target_tensor(t.sim,gymtorch.unwrap_tensor(targets))
        t.gym.simulate(t.sim);t.gym.fetch_results(t.sim,True);t._refresh_sim_tensors()
        actual=torch.cat([t._rigid_body_pos,t._rigid_body_rot,t._rigid_body_vel,t._rigid_body_ang_vel],-1)
        # Separate one-step dynamics from relative skeleton/FK disagreement.
        relative_actual=actual[...,:3]-actual[...,:1,:3];relative_initial=initial[...,:3]-initial[...,:1,:3]
        error=(relative_actual-relative_initial).norm(dim=-1)
        report['one_step_relative_body_error_median']=float(error.median());report['one_step_relative_body_error_max']=float(error.max())
        active=t._native_pool.active
        report['native_one_step_relative_body_error_median']=float(error[active].median());report['native_one_step_relative_body_error_max']=float(error[active].max())
        saved=t._kinematic_humanoid_rigid_body_states
        try:
            t._kinematic_humanoid_rigid_body_states=actual
            report['one_step_actual_foot_crossing_envs']=int(t._native_foot_guard.rejected(rows).sum())
        finally:t._kinematic_humanoid_rigid_body_states=saved
        report['post_step_finite']=bool(torch.isfinite(actual).all())
        (out/'validation_report.json').write_text(json.dumps(report,indent=2));(out/'completed.json').write_text(json.dumps(dict(rc=0,optimizer_steps=0)));print('RSI_RESET_VALIDATED',json.dumps(report),flush=True)


parser.RsiValidationTask=RsiValidationTask
spec=importlib.util.spec_from_file_location('rsi_validation_run',ROOT/'tokenhsi/run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run);original=run.build_alg_runner

def build(observer):
    runner=original(observer);runner.player_factory.register_builder('chair_ft',lambda **kw:RsiValidationPlayer(**kw));return runner
run.build_alg_runner=build;run.main()
