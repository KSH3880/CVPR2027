"""Short non-training GPU audit of the common-face CONTACT runtime."""
from isaacgym import gymapi
import json
from pathlib import Path
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.push_point import common_face_state
from utils.edge_context_spec import HOLDING
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
args=get_args();cfg,train,_=load_cfg(args);cfg['args']=args
cfg['env']['motion_file']=args.motion_file
sim=parse_sim_params(args,cfg,train);set_seed(42);torch.set_num_threads(1);args.eval=False
task=HumanoidMAStage1Push(cfg,sim,args.physics_engine,args.device,args.device_id,True)
try:
    ids=torch.arange(task.num_envs,device=task.device)
    for _ in range(4):
        task._reset_envs(ids)
        assert not task._push_hits.any()
        assert task.obs_buf.shape[1]==634
    preserved=task._box_states[1:].clone();task._reset_envs(ids[:1])
    torch.testing.assert_close(preserved,task._box_states[1:],rtol=0,atol=0)
    for _ in range(12):
        task.pre_physics_step(torch.zeros(task.num_envs*2,task.num_actions,device=task.device))
        task._physics_step();task.post_physics_step()
        phi,diag=task._evaluate_ontop_context()
        score=common_face_state(task._rigid_body_pos[:,:,task._key_body_ids[:2]],task._box_states[:,:2],task._box_size[:,:2],task._tar_pos)
        graph=task.relation_runtime.graph;mask=graph.edge_relation==HOLDING
        torch.testing.assert_close(phi[mask],score.gather(1,graph.edge_owner)[mask])
        assert torch.isfinite(task.rew_buf).all() and torch.isfinite(task.obs_buf).all()
        assert 'midpoint_height_in_band' not in task._push_measurements
    out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
    report=dict(config=args.cfg_env,environments=task.num_envs,steps=12,observation_width=634,partial_reset_preserved=True,state_matches_common_face=True,finite=True)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
finally:
    task.gym.destroy_sim(task.sim)
