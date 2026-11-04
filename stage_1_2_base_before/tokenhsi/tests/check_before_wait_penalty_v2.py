from check_before_pickup_carry_rsi import main
import torch

def inspect(task):
    assert task._before_wait is not None and task._before_wait.margin==3. and task._before_wait.coefficient==1.
    assert task._approach_timeout is None
    assert task.relation_runtime.continuous_progress is None and not task.relation_runtime.state_rescale
    assert task._relation_cfg['success']['at_z_tolerance']==.2
    assert task.REWARD_TERM_NAMES[-2:] == ('wait_penalty','total')
    return {'waiting_penalty_v2':True,'no_extra_termination':True}

def check_step(task,obs):
    terms=task.extras['reward_terms'];assert terms.shape[-1]==8
    assert (terms[:,-2]<=0).all()
    torch.testing.assert_close(terms[:,-1],terms[:,:-1].sum(-1),atol=1e-6,rtol=1e-5)

if __name__=='__main__':
    main(inspect,[(p,'loco') for p in ['before_mixed','random_scenario','before_stack','all_solo']],check_step)
