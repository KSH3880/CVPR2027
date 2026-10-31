"""CPU physics integration only; no PPO, no teacher/student optimization."""
import os,sys
from pathlib import Path
__file__ = str(Path(__file__).resolve())
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path[:0]=[str(ROOT),str(ROOT/'tokenhsi')]
import isaacgym
import torch
from utils.config import get_args,load_cfg,parse_sim_params
from env.mixed_reference_env import MixedReferenceEnv
def main():
    mode = sys.argv[1] if len(sys.argv)>1 else 'mixed'
    sys.argv=['cpu_check','--task','HumanoidTokenHSIPushKD','--cfg_env','tokenhsi/data/cfg/multi_task/push_kd.yaml','--cfg_train','tokenhsi/data/cfg/train/rlg/push_kd.yaml','--sim_device','cpu','--rl_device','cpu','--pipeline','cpu','--headless','--num_envs','2']
    if mode=='push':sys.argv+=['--test','--eval','--eval_task','push']
    a=get_args();cfg,train,_=load_cfg(a);params=parse_sim_params(a,cfg,train)
    cfg['seed'] = 42
    torch.manual_seed(42)
    e=MixedReferenceEnv(a,cfg,params)
    o=e.reset(); print('CPU_RESET',mode,list(o.shape),flush=True)
    probe = None
    if mode == 'policy':
        from policy_probe import PolicyProbe
        probe = PolicyProbe()
    for i in range(3):
        action,kl = probe.action(o,e.task.kd_mask) if probe else (torch.zeros(e.num_envs,32),0.)
        o,r,d,info=e.step(action);assert torch.isfinite(o).all();assert torch.isfinite(r).all()
        print('CPU_STEP',i,'finite',True,'teacher_kl',kl,flush=True)
    if e.reference:
        slots=e.reference._push_assets.nonzero()
        if len(slots):
            scene,owner=slots[0].tolist()
            props=e.reference.gym.get_actor_rigid_body_properties(e.reference.envs[scene],e.reference._box_handles[scene*4+owner])
            expected=(e.reference._box_size[scene,owner].prod()*3).item()
            assert abs(props[0].mass-expected)<1e-4
            print('CPU_MASS',props[0].mass,'expected',expected,flush=True)
    if e.native is not None:e.native.close()
    if e.reference is not None:e.reference.gym.destroy_sim(e.reference.sim)
    print('CPU_INTEGRATION_OK',mode,flush=True)

if __name__ == '__main__':
    main()
