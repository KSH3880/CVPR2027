"""Short headless wiring check, no learner and no viewer."""
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
    cfg['env']['motion_file']=args.motion_file;cfg['args']=args
    set_seed(42)
    task,env=parse_task(args,cfg,train['params']['config'],parse_sim_params(args,cfg,train))
    try:
        env.reset();rt=task.relation_runtime;c=rt.continuous_progress
        assert c is not None and task._approach_timeout is None and not rt.state_rescale
        for _ in range(4):
            obs,reward,done,info=env.step(torch.zeros(task.num_envs*task.num_agents,task.num_actions,device=task.device))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        # Read geometry twice without committing anchors.
        seen=c.seen.clone();task._evaluate_ontop_context();q,w=rt.progress_inputs
        task._evaluate_ontop_context();assert torch.equal(seen,c.seen)
        dep=rt.graph.prereq_mask.any(-1)&rt.graph.edge_valid
        assert (dep.sum(-1)==2).all()
        # Controlled opening at unchanged geometry exercises the full runtime.
        phi,diag=task._evaluate_ontop_context();q,w=rt.progress_inputs
        zero=torch.zeros_like(phi);phi=torch.ones_like(phi)
        c.reset(torch.arange(task.num_envs,device=task.device))
        expected=.5*w*task._relation_cfg['progress_reward_weight']
        result=rt.step(phi,diag['progress'],zero,zero,zero)
        torch.testing.assert_close(result['progress_component'][dep],expected[dep])
        assert c.seen[dep].all()
        rt.progress_inputs=(q,w);again=rt.step(phi,diag['progress'],zero,zero,zero)
        assert torch.equal(result['progress_component'][dep],again['progress_component'][dep])
        before=c.q0[1:].clone();task._reset_envs(torch.tensor([0],device=task.device))
        assert not c.seen[0].any() and torch.equal(before,c.q0[1:])
        out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
        data=dict(finite=True,geometry_read_pure=True,gate_open_continuous=True,stationary_unchanged=True,partial_reset=True,approach_timeout=False,state_rescale=False)
        (out/'physics_validation.json').write_text(json.dumps(data,indent=2));print(data,flush=True)
    finally:
        torch.cuda.synchronize()
        if task.viewer is not None:task.gym.destroy_viewer(task.viewer)
        task.gym.destroy_sim(task.sim)
if __name__=='__main__':main()
