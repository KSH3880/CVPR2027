"""No training: short physics rollout plus controlled timeout/reset wiring checks."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.parse_task import parse_task

def main():
    torch.set_num_threads(1)
    args=get_args();args.test=args.eval=True
    cfg,train,_=load_cfg(args);cfg['env']['motion_file']=args.motion_file;cfg['args']=args
    set_seed(42)
    task,env=parse_task(args,cfg,train['params']['config'],parse_sim_params(args,cfg,train))
    try:
        env.reset()
        for _ in range(4):
            obs,reward,done,info=env.step(torch.zeros(task.num_envs*task.num_agents,task.num_actions,device=task.device))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        assert not task.relation_runtime.state_rescale
        timer=task._approach_timeout;timer.reset(torch.arange(task.num_envs,device=task.device))
        graph=task.relation_runtime.graph;dep=graph.prereq_mask.any(-1)&graph.edge_valid
        result=task.relation_runtime.last_result
        result['prerequisite_gate'].copy_(dep);result['own_success'].zero_()
        task._compute_reset();assert (timer.idle[dep]==0).all() and timer.limit==60
        for _ in range(59):task._compute_reset()
        assert not task.extras['before_approach_timeout'].any()
        task._compute_reset();assert task.extras['before_approach_timeout'].all()
        assert task.reset_buf.all() and task._terminate_buf.all()
        task._reset_envs(torch.tensor([0],device=task.device))
        assert not timer.started[0].any() and timer.started[1:].any()
        task.cfg['args'].eval_protocol='general'
        before=timer.idle.clone();task._compute_reset();assert torch.equal(before,timer.idle)
        data=dict(finite=True,timeout_steps=timer.limit,control_dt=task.dt,terminal_all_agents=True,partial_reset=True,formal_eval_bypass=True,state_rescale=False)
        out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True);(out/'physics_validation.json').write_text(json.dumps(data,indent=2));print(data,flush=True)
    finally:
        torch.cuda.synchronize()
        if task.viewer is not None:task.gym.destroy_viewer(task.viewer)
        task.gym.destroy_sim(task.sim)
if __name__=='__main__':main()
