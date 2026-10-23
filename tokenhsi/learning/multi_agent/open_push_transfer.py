"""Warm start contact+AT from the same OPEN/PUSH scene experiment."""
import copy
import hashlib
from pathlib import Path
import torch
from utils.relation_task_spec import checkpoint_metadata

COPIED=('a2c_network.actor_encoder.', 'a2c_network.action_head.',
        'a2c_network._disc_mlp.', 'a2c_network._disc_logits.')
RESET=('a2c_network.critic_encoder.', 'a2c_network.value_head.')


def goal_progress_source(task,weights,target,source_path):
    """Allow only goal25 -> goal1to25 and AT delta .1; retain physics/network."""
    import yaml
    source=yaml.safe_load(Path(source_path).read_text())['env']
    if source.get('pushGoalRandomization')!={'sampling':'radial_mixture','distance_range':[.5,2.5],'near_distance':1.,'near_probability':.5}:
        raise ValueError('Goal/AT transfer requires the original goal25 source')
    if source['relationReward']['contact_at'].get('progress_delta')!=.5 or 'at_progress_delta' in source['relationReward']['contact_at']:
        raise ValueError('Source AT progress must use the original shared .5 delta')
    allowed=copy.deepcopy(source)
    allowed['pushGoalRandomization']={'sampling':'radial_uniform','distance_range':[1.,2.5]}
    allowed['relationReward']['contact_at']['at_progress_delta']=.1
    allowed['openPushRSI']['bank_file']=task.cfg['env']['openPushRSI']['bank_file']
    no_doors=task.cfg['env'].get('pushNoDoorPolicy',False)
    if no_doors:
        allowed['pushNoDoorPolicy']=True
    if any(task.cfg['env'].get(k)!=v for k,v in allowed.items()):
        raise ValueError('Only goal1to25, AT progress .1 and matching screened bank may change')
    snapshot=weights.get('relation_experiment_config',{}).get('env',{})
    if any(snapshot.get(k)!=v for k,v in source.items()):
        raise ValueError('Source saved environment differs')
    expected=copy.deepcopy(target)
    if no_doors:
        expected.pop('policy_observation')
        expected['object_obs_size']=40
    expected['push_goal']=source['pushGoalRandomization']
    expected['contact_at']=source['relationReward']['contact_at']
    expected['rsi']=source['openPushRSI']
    expected['rsi_bank_sha256']=hashlib.sha256(Path(source['openPushRSI']['bank_file']).read_bytes()).hexdigest()
    return source,expected


def transfer_open_push(agent,weights,path):
    task=agent.vec_env.env.task
    target=task.get_open_push_contract()
    if 'contact_at' not in task._relation_cfg or target['share_edge_encoder']:
        raise ValueError('Warm start requires contact+AT and separate actor/critic encoders')
    expected=copy.deepcopy(target)
    transfer=getattr(task,'cfg',{}).get('experiment',{}).get('transfer',{})
    source_env=transfer.get('source_env')
    if transfer.get('variant')=='goal_at_progress':
        if not source_env:raise ValueError('Goal/AT transfer requires source_env')
        source,expected=goal_progress_source(task,weights,target,source_env)
    elif source_env:
        import yaml
        source=yaml.safe_load(Path(source_env).read_text())['env']
        current_env=task.cfg['env']
        allowed=copy.deepcopy(source)
        friction=current_env['pushBox']['friction']
        if friction not in (.6,1.):
            raise ValueError('Warm start supports box friction0.6 or1.0 only')
        allowed['pushBox']['friction']=friction
        allowed['pushGoalRandomization']={'half_extent':1.5,'min_distance':1.0}
        if current_env.get('pushGoalContact',False):
            allowed['pushGoalContact']=True
            expected.pop('push_contact')
        allowed['pushSceneRandomization']['min_station_distance']=6.2
        if 'pushSizeRandomization' in current_env:
            from utils.open_push_sizes import size_values
            size_values(current_env)
            allowed['pushSizeRandomization']=current_env['pushSizeRandomization']
            allowed['pushSceneRandomization']['min_station_distance']=6.3
            expected.pop('push_size_randomization')
        if current_env['relationGraph']['task_probabilities']=={'OPEN':0.,'PUSH':1.}:
            allowed['relationGraph']['task_probabilities']={'OPEN':0.,'PUSH':1.}
            allowed['skillDiscProb']=[.33,0.,.67]
            expected['task_probabilities']=source['relationGraph']['task_probabilities']
        if current_env['relationReward']['contact_at'].get('success_mode')=='state_only':
            allowed['relationReward']['contact_at']['success_mode']='state_only'
            expected['contact_at']=copy.deepcopy(source['relationReward']['contact_at'])
        if current_env['pushSceneRandomization']['min_station_distance']==6.3:
            allowed['pushSceneRandomization']['min_station_distance']=6.3
        if current_env['relationReward']['push'].get('contact_mode')=='any_surface':
            allowed['relationReward']['push']['contact_mode']='any_surface'
        allowed['openPushRSI']['bank_file']=current_env['openPushRSI']['bank_file']
        if any(current_env.get(k)!=v for k,v in allowed.items()):
            raise ValueError('Only friction0.6/1.0, random-goal/contact geometry and screened bank may change')
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
    if source_env:reward=copy.deepcopy(source['relationReward'])
    else:reward.pop('contact_at')
    if weights.get('relation_metadata')!=checkpoint_metadata(reward):
        raise ValueError('Source reward metadata differs')
    snapshot=weights.get('relation_experiment_config',{})
    no_doors=getattr(task,'_no_door_policy',False)
    if no_doors and transfer.get('variant')!='goal_at_progress':
        raise ValueError('Door-free warm start requires the goal/AT source contract')
    if (snapshot.get('num_agents'),snapshot.get('num_objects'),snapshot.get('observation_size'))!=(2,4,task.get_obs_size()+(114 if no_doors else 0)):
        raise ValueError('Source observation dimensions differ')
    if snapshot['train']['params']['network']!=agent._relation_experiment_config['train']['params']['network']:
        raise ValueError('Source network configuration differs')
    current=agent.model.state_dict();source=weights['model']
    if no_doors:
        from utils.push_policy_observation import project_source_model
        source=project_source_model(source)
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
    normalizers={name:weights[name] for name in modules}
    if no_doors:
        from utils.push_policy_observation import project_source_normalizer
        normalizers['running_mean_std']=project_source_normalizer(normalizers['running_mean_std'])
    for name,module in modules.items():
        state=module.state_dict();old=normalizers[name]
        if set(state)!=set(old) or any(state[k].shape!=old[k].shape or state[k].dtype!=old[k].dtype or not torch.isfinite(old[k]).all() for k in state):
            raise ValueError('Source normalizer differs: '+name)
    # Preserve freshly initialized critic; never load optimizer/reward RMS/counters.
    agent.model.load_state_dict({k:source[k] if k in copied else v for k,v in current.items()},strict=True)
    for name,module in modules.items():module.load_state_dict(normalizers[name],strict=True)
    return dict(source_checkpoint=str(Path(path).resolve()),
        source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        source_epoch=int(weights['epoch']),source_frame=int(weights['frame']),
        copied_tensors=copied,reset_tensors=reset,
        copied_normalizers=list(modules),
        reset_state=['critic','optimizer','reward_mean_std','epoch','frame','AMP replay buffers'],
        freeze=False,task_probabilities=target['task_probabilities'],
        policy_observation='push_no_doors_30d_v2' if no_doors else 'original',
        removed_observation_width=114 if no_doors else 0)
