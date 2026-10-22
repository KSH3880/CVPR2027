"""Non-training CPU physics check; supply normal task CLI with --num_envs 2."""
from isaacgym import gymapi
from utils.config import get_args, load_cfg, parse_sim_params
from env.tasks.multi_agent.humanoid_ma_door import HumanoidMADoor
import torch
args=get_args()
if not (args.task == 'HumanoidMADoor' and args.num_envs == 2 and args.test and args.eval
        and args.device == 'cpu' and args.rl_device == 'cpu'):
    raise ValueError('This CPU check requires HumanoidMADoor, --num_envs 2 --test --eval and CPU sim/RL devices')
cfg,train,logdir=load_cfg(args)
cfg['args']=args
cfg['env']['motion_file']=args.motion_file
sim_params=parse_sim_params(args,cfg,train)
task=HumanoidMADoor(cfg,sim_params,args.physics_engine,args.device,args.device_id,True)
try:
    env_ids=torch.arange(task.num_envs,device=task.device)
    if getattr(task,'_door_reference_bank',None):
        # Exercise training RSI without constructing an agent or doing optimization.
        task._is_eval=False
        task._skill=list(cfg['env']['skill'])
        task._skill_init_prob=torch.zeros(len(task._skill),device=task.device)
        task._skill_init_prob[task._skill.index('pickUp')]=1.
    task._reset_envs(env_ids)
    if getattr(task,'_door_reference_bank',None):
        assert 'pickUp' in task._reset_ref_slots
        print('DOOR_RSI_POOL_COUNTS',task._door_reference_bank.counts)
    assert task.obs_buf.shape == (2,task.get_obs_size())
    assert torch.isfinite(task.obs_buf).all()
    before=task.door_hinge_angle[1].clone()
    graph=task.relation_runtime.suffix().clone()
    task.reset_door_angles(env_ids[:1],torch.tensor([[.4,.8]],device=task.device))
    assert torch.equal(task.door_hinge_angle[1],before)
    assert torch.equal(graph,task.relation_runtime.suffix())
    # Give the passive hinges initial angular velocity, then observe physical motion.
    task._door_dof_state[0,:,1]=1.
    ids=task._box_actor_ids[:1].flatten().contiguous()
    from isaacgym import gymtorch
    task.gym.set_dof_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._dof_state),gymtorch.unwrap_tensor(ids),len(ids))
    for step in range(8):
        task.pre_physics_step(torch.zeros(task.num_envs*task.num_agents,task.num_dof,device=task.device))
        for _ in range(task.control_freq_inv):
            task.gym.simulate(task.sim)
            task.gym.fetch_results(task.sim,True)
        task.post_physics_step()
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
        if getattr(task,'_door_approach_gate',None) is not None:
            weight=task.extras['door_amp_weight']
            assert weight.shape==(4,1) and torch.isfinite(weight).all()
            assert ((weight>=0)&(weight<=1)).all()
            assert weight[:2].eq(0).all()  # env0 doors already exceed interaction angle
    assert (task.door_hinge_angle[0] > torch.tensor([.4,.8],device=task.device)).all()
    panel=task._door_panel_states
    from isaacgym.torch_utils import quat_rotate
    expected=panel[...,:3]+quat_rotate(panel[...,3:7].reshape(-1,4),panel.new_tensor([.78,-.10,1.]).expand(4,3)).view(2,2,3)
    torch.testing.assert_close(task.door_handle_position,expected,atol=1e-4,rtol=1e-4)
    demo=task.fetch_amp_obs_demo(4)
    assert torch.isfinite(demo).all()
    if not cfg['env']['ampEnabled']:assert not demo.any()
    print('DOOR_CPU_SMOKE_OK', cfg['env']['ampEnabled'],task.obs_buf.shape,task.door_hinge_angle.tolist(),demo.shape)
finally:
    task.gym.destroy_sim(task.sim)
