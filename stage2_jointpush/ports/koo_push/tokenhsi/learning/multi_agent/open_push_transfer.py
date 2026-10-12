"""Warm start contact+AT from the same OPEN/PUSH scene experiment."""
import copy
import hashlib
from pathlib import Path
import torch
from utils.relation_task_spec import checkpoint_metadata

COPIED=('a2c_network.actor_encoder.', 'a2c_network.action_head.',
        'a2c_network._disc_mlp.', 'a2c_network._disc_logits.')
RESET=('a2c_network.critic_encoder.', 'a2c_network.value_head.')


def transfer_open_push(agent,weights,path):
    task=agent.vec_env.env.task
    target=task.get_open_push_contract()
    if 'contact_at' not in task._relation_cfg or target['share_edge_encoder']:
        raise ValueError('Warm start requires contact+AT and separate actor/critic encoders')
    expected=copy.deepcopy(target)
    source_env=getattr(task,'cfg',{}).get('experiment',{}).get('transfer',{}).get('source_env')
    if source_env:
        import yaml
        source=yaml.safe_load(Path(source_env).read_text())['env']
        current_env=task.cfg['env']
        allowed=copy.deepcopy(source)
        allowed['pushBox']['friction']=1.0
        allowed['pushGoalRandomization']={'half_extent':1.5,'min_distance':1.0}
        allowed['pushSceneRandomization']['min_station_distance']=6.2
        allowed['openPushRSI']['bank_file']=current_env['openPushRSI']['bank_file']
        if any(current_env.get(k)!=v for k,v in allowed.items()):
            raise ValueError('Only friction1/random-goal sampling and screened bank may change')
        snapshot=weights.get('relation_experiment_config',{}).get('env',{})
        if any(snapshot.get(k)!=v for k,v in source.items()):
            raise ValueError('Source saved environment differs')
        expected.pop('push_goal')
        expected['push_box']=source['pushBox']
        expected['push_scene']=source['pushSceneRandomization']
        expected['rsi']=source['openPushRSI']
        expected['rsi_bank_sha256']=hashlib.sha256(Path(source['openPushRSI']['bank_file']).read_bytes()).hexdigest()
    else:
        expected.pop('contact_at')
        expected['push_reward_contract']='rsi_baseline_plus_hand_distance_midpoint_xy_v2'
    if weights.get('open_push_contract')!=expected or weights.get('amp_enabled') is not True:
        raise ValueError('Source OPEN/PUSH physics/AMP/RSI/scene contract differs')
    reward=copy.deepcopy(task._relation_cfg)
    if not source_env:reward.pop('contact_at')
    if weights.get('relation_metadata')!=checkpoint_metadata(reward):
        raise ValueError('Source reward metadata differs')
    snapshot=weights.get('relation_experiment_config',{})
    if (snapshot.get('num_agents'),snapshot.get('num_objects'),snapshot.get('observation_size'))!=(2,4,task.get_obs_size()):
        raise ValueError('Source observation dimensions differ')
    if snapshot['train']['params']['network']!=agent._relation_experiment_config['train']['params']['network']:
        raise ValueError('Source network configuration differs')
    current=agent.model.state_dict();source=weights['model']
    if set(current)!=set(source):
        raise ValueError('Source model keys differ')
    copied=[];reset=[]
    for key,value in current.items():
        if source[key].shape!=value.shape or source[key].dtype!=value.dtype or not torch.isfinite(source[key]).all():
            raise ValueError('Invalid source model tensor: '+key)
        if key.startswith(COPIED) or key=='a2c_network.sigma':copied.append(key)
        elif key.startswith(RESET):reset.append(key)
        else:raise ValueError('Unclassified model tensor: '+key)
    if not copied or not reset or agent.optimizer.state:
        raise ValueError('Warm start requires fresh optimizer and actor/critic tensors')
    modules={'running_mean_std':agent.running_mean_std,
             'amp_input_mean_std':agent._amp_input_mean_std}
    for name,module in modules.items():
        state=module.state_dict();old=weights[name]
        if set(state)!=set(old) or any(state[k].shape!=old[k].shape or state[k].dtype!=old[k].dtype or not torch.isfinite(old[k]).all() for k in state):
            raise ValueError('Source normalizer differs: '+name)
    # Preserve freshly initialized critic; never load optimizer/reward RMS/counters.
    agent.model.load_state_dict({k:source[k] if k in copied else v for k,v in current.items()},strict=True)
    for name,module in modules.items():module.load_state_dict(weights[name],strict=True)
    return dict(source_checkpoint=str(Path(path).resolve()),
        source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        source_epoch=int(weights['epoch']),source_frame=int(weights['frame']),
        copied_tensors=copied,reset_tensors=reset,
        copied_normalizers=list(modules),
        reset_state=['critic','optimizer','reward_mean_std','epoch','frame','AMP replay buffers'],
        freeze=False,task_probabilities=target['task_probabilities'])
