"""Construct the fixed Stage1 or BEFORE task and scene wrapper."""
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.vec_task_wrapper_ma import VecTaskPythonWrapperMA
import numpy as np


def parse_task(args, cfg, cfg_train, sim_params):
    if args.task == 'HumanoidMACarry':
        task_class = HumanoidMACarry
    elif args.task == 'HumanoidMAPush':
        from env.tasks.multi_agent.humanoid_ma_push import HumanoidMAPush
        task_class = HumanoidMAPush
    elif args.task == 'HumanoidMACoupledPush':
        from env.tasks.multi_agent.humanoid_ma_coupled_push import HumanoidMACoupledPush
        task_class=HumanoidMACoupledPush
    elif args.task == 'HumanoidMABeforePush':
        from env.tasks.multi_agent.humanoid_ma_before_push import HumanoidMABeforePush
        task_class = HumanoidMABeforePush
    elif args.task == 'HumanoidMABefore':
        from env.tasks.multi_agent.humanoid_ma_before import HumanoidMABefore
        task_class = HumanoidMABefore
    else:
        raise ValueError('Supported tasks: HumanoidMACarry, HumanoidMABefore')
    cfg['seed'] = cfg_train.get('seed', -1)
    cfg['env']['seed'] = cfg['seed']
    task = task_class(cfg=cfg, sim_params=sim_params, physics_engine=args.physics_engine,
                      device_type=args.device, device_id=args.device_id, headless=args.headless)
    env = VecTaskPythonWrapperMA(task, args.rl_device,
        cfg_train.get('clip_observations', np.inf), cfg_train.get('clip_actions', 1.0))
    return task, env
