"""Validated Stage-1 OPEN/PUSH -> shared PUSH CA4 transfer, no optimizer import."""
import torch


def transfer_joint_push(agent, checkpoint):
    metadata=checkpoint.get('relation_metadata',{})
    if metadata.get('schema_version')!=12 or metadata.get('context_fusion')!='semantic_only' or metadata.get('relation_taxonomy',{}).get('push')!=12:
        raise ValueError('Joint Push requires koo semantic OPEN/PUSH Stage1 checkpoint')
    if 'joint_push_reward' in metadata.get('relation_reward_config', {}) or 'joint_push' in checkpoint.get('open_push_contract', {}):
        raise ValueError('Use resume for a Joint Push checkpoint')
    task=agent.vec_env.env.task
    old=checkpoint.get('relation_experiment_config',{})
    if (old.get('num_agents'),old.get('num_objects'),old.get('observation_size'))!=(2,4,task.get_obs_size()):
        raise ValueError('Stage1 observation contract differs')
    source_network=dict(old['train']['params']['network'])
    target_network=dict(agent._relation_experiment_config['train']['params']['network']);target_network.pop('coordination',None)
    if source_network!=target_network:
        raise ValueError('Stage1 encoder/network configuration differs')
    target=agent.model.state_dict();source=checkpoint['model'];new=[];copied=[]
    for key,value in target.items():
        if '.push_at_coordination.' in key:new.append(key);continue
        if key not in source:raise ValueError('Missing Stage1 tensor '+key)
        previous=source[key]
        if not torch.isfinite(previous).all():raise ValueError('Nonfinite source '+key)
        if key=='a2c_network.action_head.0.0.weight':
            if value.shape!=(previous.shape[0],128) or previous.shape[1]!=64:
                raise ValueError('Invalid action head expansion')
        elif value.shape!=previous.shape or value.dtype!=previous.dtype:
            raise ValueError('Tensor mismatch '+key)
        copied.append(key)
    if set(source)-set(target):raise ValueError('Unexpected source model tensors')
    normalizers={'running_mean_std':agent.running_mean_std,'amp_input_mean_std':agent._amp_input_mean_std}
    for name,module in normalizers.items():
        expected=module.state_dict();state=checkpoint[name]
        if set(expected)!=set(state) or any(expected[k].shape!=state[k].shape or not torch.isfinite(state[k]).all() for k in expected):
            raise ValueError('Normalizer mismatch '+name)
    for key in copied:
        if key=='a2c_network.action_head.0.0.weight':
            target[key][:,:64].copy_(source[key]);target[key][:,64:]=0
        else:target[key].copy_(source[key])
    agent.model.load_state_dict(target,strict=True)
    for name,module in normalizers.items():module.load_state_dict(checkpoint[name],strict=True)
    return dict(source_epoch=int(checkpoint['epoch']),copied_tensors=copied,new_tensors=new,
        freeze='actor_encoder_and_observation_rms',head_expansion=[64,64],rsi='loco_only',
        optimizer='fresh',counters='fresh')
