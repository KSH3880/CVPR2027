"""Short physics integration check, not training or policy evaluation."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task


def main():
    torch.set_num_threads(1)
    args=get_args();args.test=args.eval=True
    cfg,train,_=load_cfg(args)
    print('Environment-only check: checkpoint argument is not loaded; actions are zero.',flush=True)
    cfg['env']['motion_file'],cfg['args']=args.motion_file,args
    set_seed(42)
    task,env=parse_task(args,cfg,train['params']['config'],parse_sim_params(args,cfg,train))
    try:
        results=[]
        # Each fixed preset needs its own size-compatible asset initialization.
        for preset in (args.task_graph,):
            task._task_graph_preset=preset
            obs=env.reset()
            for _ in range(4):
                actions=torch.zeros(task.num_envs*task.num_agents,task.num_actions,device=task.device)
                obs,reward,done,info=env.step(actions)
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                terms=info['reward_terms']
                assert task.relation_runtime.state_rescale
                assert terms.shape[-1]==len(task.REWARD_TERM_NAMES)==7
                torch.testing.assert_close(terms[:,:-1].sum(-1),reward)
                result=task.relation_runtime.last_result
                assert torch.isfinite(result['state_component']).all()
                assert torch.isfinite(result['state_open_value']).all()
            previous=task._root_states.view(task.num_envs,-1,13)[1:].clone()
            task._reset_envs(torch.tensor([0],device=task.device))
            assert torch.equal(previous,task._root_states.view(task.num_envs,-1,13)[1:])
            assert not task.relation_runtime.state_open_seen[0].any()
            results.append(dict(preset=preset,finite=True,reward_sum=True,partial_reset=True))
        out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
        (out/'physics_validation.json').write_text(json.dumps(results,indent=2)+'\n')
        print('STATE_RESCALE PHYSICS CHECK COMPLETE',flush=True)
    finally:
        torch.cuda.synchronize()
        if task.viewer is not None:task.gym.destroy_viewer(task.viewer)
        task.gym.destroy_sim(task.sim)


if __name__=='__main__':main()
