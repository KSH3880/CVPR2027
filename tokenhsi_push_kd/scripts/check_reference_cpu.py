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
    device = os.environ.get('CHECK_SIM_DEVICE', 'cpu')
    if device != 'cpu':
        from scripts.local_cuda_compat import install
        install()
        for flag in ('--sim_device', '--rl_device'):sys.argv[sys.argv.index(flag)+1] = device
        sys.argv[sys.argv.index('--pipeline')+1] = 'gpu'
    if mode=='push':sys.argv+=['--test','--eval','--eval_task','push']
    a=get_args();cfg,train,_=load_cfg(a);params=parse_sim_params(a,cfg,train)
    cfg['env']['referenceNumEnvs'] = int(os.environ.get('CHECK_REFERENCE_ENVS', '2'))
    cfg['seed'] = 42
    torch.manual_seed(42)
    e=MixedReferenceEnv(a,cfg,params)
    o=e.reset(); print('CPU_RESET',mode,list(o.shape),flush=True)
    probe = None
    if mode == 'policy':
        from policy_probe import PolicyProbe
        probe = PolicyProbe()
    if e.shared_sim is not None:
        native_roots = e.native._root_states.clone()
        e.reset(torch.tensor([0], device=e.rl_device))
        torch.testing.assert_close(e.native._root_states, native_roots, rtol=0, atol=0)
        reference_roots = e.reference._root_states.clone()
        e.reset(torch.tensor([e.reference_rows], device=e.rl_device))
        torch.testing.assert_close(e.reference._root_states, reference_roots, rtol=0, atol=0)
        assert e.reference.sim is e.native.sim
        print('SHARED_RESET_ISOLATION_OK', flush=True)
    for i in range(3):
        action,kl = probe.action(o,e.task.kd_mask) if probe else (torch.zeros(e.num_envs,32,device=e.rl_device),0.)
        calls_before = e.shared_sim.simulation_calls if e.shared_sim is not None else 0
        o,r,d,info=e.step(action);assert torch.isfinite(o).all();assert torch.isfinite(r).all()
        if e.shared_sim is not None:
            assert e.shared_sim.simulation_calls-calls_before == e.reference.control_freq_inv
            assert e.shared_sim.steps == i+1
            assert (e.reference.progress_buf == i+1).all() and (e.native.progress_buf == i+1).all()
        print('CPU_STEP',i,'finite',True,'teacher_kl',kl,flush=True)
    if e.reference:
        slots=e.reference._push_assets.nonzero()
        if len(slots):
            scene,owner=slots[0].tolist()
            props=e.reference.gym.get_actor_rigid_body_properties(e.reference.envs[scene],e.reference._box_handles[scene*e.reference.num_objects+owner])
            expected=(e.reference._box_size[scene,owner].cpu().prod()*3).item()
            assert abs(props[0].mass-expected)<1e-4
            print('CPU_MASS',props[0].mass,'expected',expected,flush=True)
    e.close()
    print('CPU_INTEGRATION_OK',mode,flush=True)

if __name__ == '__main__':
    main()
