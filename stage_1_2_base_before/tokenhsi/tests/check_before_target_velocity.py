"""Short physics wiring check; does not train a policy or open a viewer."""
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
        assert s.target_velocity_config['mode'] in ('sit_climb_l2_v1','sit_climb_speed_cap_v1') and s.soft_entry_width==0
        counts={'sit_climb':int((~s.top).sum()),'ontop':int(s.top.sum())}
        assert min(counts.values())>0
        assert task._approach_timeout is None and task.relation_runtime.continuous_progress is None
        cap_mode=s.target_velocity_config['mode']=='sit_climb_speed_cap_v1'
        def error(velocity,target):
            return ((velocity.norm(dim=-1)-target.norm(dim=-1)).clamp_min(0) if cap_mode
                    else (velocity-target).norm(dim=-1))
        for step in range(8):
            if step == 6 and s.target_velocity_config.get('after_release')=='disable':
                s.released[s.rows,s.edges]=True
            if step == 4:
                # Force only the latch for wiring coverage; genuine HOLDING activation is CPU-tested.
                s.picked_up[s.rows[s.top],s.edges[s.top]]=True
            obs,reward,done,info=env.step(torch.zeros(task.num_envs*task.num_agents,task.num_actions,device=task.device))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            terms=task.extras['reward_terms'];assert terms.shape[-1]==8
            torch.testing.assert_close(terms[:,-1],terms[:,:-1].sum(-1),atol=1e-6,rtol=1e-5)
            velocity=task._humanoid_root_states[...,7:10]
            expected=torch.zeros(velocity.shape[:2],device=task.device)
            r,o=s.rows[s.top],s.owners[s.top]
            boxes=task._logical_box_values(task._box_states)
            payload=s.source[s.top]-task.num_agents
            expected[r,o]=s.velocity_weight[r,o]*error(boxes[r,payload,7:9],s.target_velocity[r,o])*s.coefficient
            inactive=~s.picked_up[r,s.edges[s.top]]
            assert not expected[r[inactive],o[inactive]].any()
            r,o=s.rows[~s.top],s.owners[~s.top]
            expected[r,o]=s.velocity_weight[r,o]*error(velocity[r,o,:2],s.target_velocity[r,o])*s.coefficient
            torch.testing.assert_close(terms[:,-2],-expected.flatten())
            assert (s.target_velocity.norm(dim=-1)<=.500001).all()
            if step>=6 and s.target_velocity_config.get('after_release')=='disable':
                assert not terms[:,-2].any()
        saved=s.released[1:].clone();task._reset_envs(torch.tensor([0],device=task.device));assert torch.equal(saved,s.released[1:])
        result=dict(pair_counts=counts,finite=True,velocity_mode=s.target_velocity_config['mode'],reward_sum=True,ontop_box_velocity=True,ontop_latch_injected_for_wiring=True,partial_reset=True,extra_early_termination=False)
        out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True);(out/'physics_validation.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
    finally:
        torch.cuda.synchronize()
        if task.viewer is not None:task.gym.destroy_viewer(task.viewer)
        task.gym.destroy_sim(task.sim)
if __name__=='__main__':main()
