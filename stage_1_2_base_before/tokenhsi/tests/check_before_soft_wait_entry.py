"""Headless environment wiring check; no policy training or viewer."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.parse_task import parse_task

def main():
    torch.set_num_threads(1)
    args=get_args();args.test=args.eval=False
    cfg,train,_=load_cfg(args);args.test=args.eval=True
    cfg['env']['motion_file']=args.motion_file;cfg['args']=args;set_seed(42)
    task,env=parse_task(args,cfg,train['params']['config'],parse_sim_params(args,cfg,train))
    try:
        env.reset();s=task._before_wait
        assert s.soft_entry_width==1 and task._approach_timeout is None
        assert task.relation_runtime.continuous_progress is None
        for _ in range(4):
            obs,reward,done,info=env.step(torch.zeros(task.num_envs*task.num_agents,task.num_actions,device=task.device))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            terms=task.extras['reward_terms'];assert terms.shape[-1]==8
            torch.testing.assert_close(terms[:,-1],terms[:,:-1].sum(-1),atol=1e-6,rtol=1e-5)
            speed=task._humanoid_root_states[...,7:10].norm(dim=-1)
            expected=torch.zeros_like(speed);expected[s.rows,s.owners]=s.entry_weight[s.rows,s.edges]*speed[s.rows,s.owners]
            torch.testing.assert_close(terms[:,-2],-expected.flatten())
        saved=s.entry_weight[1:].clone();task._reset_envs(torch.tensor([0],device=task.device));assert torch.equal(saved,s.entry_weight[1:]) and not s.entry_weight[0].any()
        result=dict(finite=True,soft_entry_width=1.,penalty_matches_weight_times_xyz_speed=True,reward_sum=True,partial_reset=True,extra_early_termination=False)
        out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True);(out/'physics_validation.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
    finally:
        torch.cuda.synchronize()
        if task.viewer is not None:task.gym.destroy_viewer(task.viewer)
        task.gym.destroy_sim(task.sim)
if __name__=='__main__':main()
