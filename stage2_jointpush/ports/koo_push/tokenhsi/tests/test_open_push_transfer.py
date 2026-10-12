"""Warm start copies actor/AMP only and rejects incompatible state."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from isaacgym import gymapi  # before torch
import pytest
import torch
import yaml
from learning.multi_agent.open_push_transfer import transfer_open_push
from utils.relation_task_spec import checkpoint_metadata


class Model:
    def __init__(self):
        self.values={f'a2c_network.{name}.weight':torch.zeros(2,2) for name in (
            'actor_encoder','action_head','_disc_mlp','_disc_logits','critic_encoder','value_head')}
        self.values['a2c_network.sigma']=torch.zeros(2)
    def state_dict(self):return self.values
    def load_state_dict(self,values,strict=True):self.values={k:v.clone() for k,v in values.items()}


def setup(tmp_path):
    env=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/open_push_push180_contact_at.yaml').read_text())['env']
    contract={'share_edge_encoder':False,'contact_at':env['relationReward']['contact_at'],
        'push_reward_contract':'independent_contact_plus_legacy_at_v1','task_probabilities':{'OPEN':.5,'PUSH':.5}}
    task=SimpleNamespace(_relation_cfg=env['relationReward'],get_open_push_contract=lambda:deepcopy(contract),get_obs_size=lambda:684)
    rms=torch.nn.Linear(2,2);amp=torch.nn.Linear(2,2)
    snapshot={'num_agents':2,'num_objects':4,'observation_size':684,'train':{'params':{'network':{'shared':False}}}}
    agent=SimpleNamespace(vec_env=SimpleNamespace(env=SimpleNamespace(task=task)),model=Model(),optimizer=SimpleNamespace(state={}),
        running_mean_std=rms,_amp_input_mean_std=amp,_relation_experiment_config=deepcopy(snapshot),epoch_num=0,frame=0)
    old=deepcopy(contract);old.pop('contact_at');old['push_reward_contract']='rsi_baseline_plus_hand_distance_midpoint_xy_v2'
    reward=deepcopy(env['relationReward']);reward.pop('contact_at')
    weights={'model':{k:torch.ones_like(v)*3 for k,v in agent.model.state_dict().items()},'amp_enabled':True,
        'open_push_contract':old,'relation_metadata':checkpoint_metadata(reward),'relation_experiment_config':snapshot,
        'running_mean_std':{k:torch.ones_like(v)*4 for k,v in rms.state_dict().items()},
        'amp_input_mean_std':{k:torch.ones_like(v)*5 for k,v in amp.state_dict().items()},
        'epoch':4500,'frame':999,'optimizer':{'old':True},'reward_mean_std':{'old':True}}
    path=tmp_path/'source.pth';path.write_bytes(b'source')
    return agent,weights,path


def test_actor_amp_rms_copied_critic_and_optimizer_stay_fresh(tmp_path):
    agent,weights,path=setup(tmp_path)
    report=transfer_open_push(agent,weights,path)
    for k,v in agent.model.state_dict().items():
        assert (v==(0 if k in report['reset_tensors'] else 3)).all()
    assert all((v==4).all() for v in agent.running_mean_std.state_dict().values())
    assert all((v==5).all() for v in agent._amp_input_mean_std.state_dict().values())
    assert not agent.optimizer.state and agent.epoch_num==0 and agent.frame==0
    assert report['source_epoch']==4500 and report['task_probabilities']=={'OPEN':.5,'PUSH':.5}


@pytest.mark.parametrize('bad',['contract','network','shape','normalizer','optimizer','nan'])
def test_incompatible_source_rejected_before_loading(tmp_path,bad):
    agent,weights,path=setup(tmp_path)
    before={k:v.clone() for k,v in agent.model.state_dict().items()}
    if bad=='contract':weights['open_push_contract']['task_probabilities']={'OPEN':0.,'PUSH':1.}
    if bad=='network':weights['relation_experiment_config']['train']['params']['network']['shared']=True
    if bad=='shape':weights['model']['a2c_network.sigma']=torch.ones(3)
    if bad=='normalizer':weights['running_mean_std'].pop('weight')
    if bad=='optimizer':agent.optimizer.state[1]={}
    if bad=='nan':weights['model']['a2c_network.sigma'][0]=float('nan')
    with pytest.raises(ValueError):transfer_open_push(agent,weights,path)
    for k,v in before.items():torch.testing.assert_close(agent.model.state_dict()[k],v)


@pytest.mark.parametrize('bad',[None,'friction','door'])
def test_friction_random_goal_transfer_checks_source_and_resets_critic(tmp_path,bad):
    import hashlib
    agent,weights,path=setup(tmp_path)
    task=agent.vec_env.env.task
    source_path=Path('tokenhsi/data/cfg/multi_agent/open_push_push180_contact_at_warmstart.yaml')
    source=yaml.safe_load(source_path.read_text())['env']
    target=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction1_random_goal.yaml').read_text())
    task.cfg=target
    contract=task.get_open_push_contract()
    contract.update(push_goal=target['env']['pushGoalRandomization'],push_box=target['env']['pushBox'],push_scene=target['env']['pushSceneRandomization'],rsi=target['env']['openPushRSI'],rsi_bank_sha256='new')
    task.get_open_push_contract=lambda:deepcopy(contract)
    expected=deepcopy(contract);expected.pop('push_goal')
    expected.update(push_box=source['pushBox'],push_scene=source['pushSceneRandomization'],rsi=source['openPushRSI'],rsi_bank_sha256=hashlib.sha256(Path(source['openPushRSI']['bank_file']).read_bytes()).hexdigest())
    weights['open_push_contract']=expected
    weights['relation_metadata']=checkpoint_metadata(source['relationReward'])
    weights['relation_experiment_config']['env']=deepcopy(source)
    if bad=='friction':target['env']['pushBox']['friction']=.8
    if bad=='door':weights['relation_experiment_config']['env']['door']['hinge']['friction']=9.
    if bad:
        with pytest.raises(ValueError):transfer_open_push(agent,weights,path)
    else:
        report=transfer_open_push(agent,weights,path)
        assert all((agent.model.state_dict()[k]==0).all() for k in report['reset_tensors'])
        assert not agent.optimizer.state
