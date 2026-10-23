"""PUSH-only configuration and explicit migration of compatible checkpoints."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import yaml


def without_open_settings(env):
    env=deepcopy(env)
    for key in list(env):
        if 'door' in key.lower():env.pop(key)
    env['pushOnly']=True
    env['openPushLayout'].pop('door_x',None)
    env['relationReward'].pop('door',None)
    env['relationGraph']['task_probabilities']={'PUSH':1.}
    if env['skill']==['loco','doorOpen','push']:
        env['skill']=['loco','push']
        env['skillInitProb']=[env['skillInitProb'][i] for i in (0,2)]
        env['skillDiscProb']=[env['skillDiscProb'][i] for i in (0,2)]
        env['ampExpertMapping']={'PUSH':[env['ampExpertMapping']['PUSH'][i] for i in (0,2)]}
    return env


def project_bank(data):
    data=deepcopy(data)
    c=data['contract']
    for key in ('door','hold'):c.pop(key,None)
    c['motion_hashes']={'push':c['motion_hashes']['push']}
    c['reward'].pop('door',None);c['layout'].pop('door_x',None)
    data['candidates']={'push':data['candidates']['push']}
    for row in data['candidates']['push']:row.pop('angle',None)
    return data


def training_contract(env,train_path):
    manifest=Path(env['motion_file'])
    motions=yaml.safe_load(manifest.read_text())['motions']
    return dict(revision=1,profile='push_only_physical_v1',policy_observation='push_no_doors_30d_v2',
        object_obs_size=30,amp_label_width=2,
        motion_hashes={s:[hashlib.sha256((manifest.parent/r['file']).read_bytes()).hexdigest() for r in motions[s]] for s in env['skill']},
        skill=env['skill'],expert_mapping=env['ampExpertMapping'],init=env['skillInitProb'],
        reward=env['relationReward'],push_box=env['pushBox'],layout=env['openPushLayout'],
        push_scene=env['pushSceneRandomization'],push_goal=env['pushGoalRandomization'],
        push_size_randomization=env.get('pushSizeRandomization'),push_contact=env['pushGoalContact'],
        shuffle_token_order=env['relationGraph'].get('shuffle_token_order',False),
        rsi=env['openPushRSI'],rsi_bank_sha256=hashlib.sha256(Path(env['openPushRSI']['bank_file']).read_bytes()).hexdigest(),
        share_edge_encoder=yaml.safe_load(Path(train_path).read_text())['params']['network']['transformer'].get('share_edge_encoder',False))


def migrate_checkpoint(weights,env,train_path):
    """Remove inactive OPEN settings only; retain all model/optimizer/counters."""
    from utils.relation_task_spec import checkpoint_metadata
    from utils.edge_context_spec import task_instance
    source=weights['relation_experiment_config']
    old=source['env']
    contract=weights.get('open_push_contract',{})
    if not old.get('pushNoDoorPolicy') or contract.get('policy_observation')!='push_no_doors_30d_v2' or old['relationGraph']['task_probabilities']!={'OPEN':0.,'PUSH':1.}:
        raise ValueError('Migration requires the original 30D PUSH-only no-door policy')
    bank_path=Path(old['openPushRSI']['bank_file'])
    if hashlib.sha256(bank_path.read_bytes()).hexdigest()!=contract['rsi_bank_sha256']:
        raise ValueError('Source RSI bank hash changed')
    if project_bank(json.loads(bank_path.read_text()))!=json.loads(Path(env['openPushRSI']['bank_file']).read_text()):
        raise ValueError('PUSH RSI candidates/physics changed')
    manifest=Path(old['motion_file'])
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=contract['manifest_sha256']:
        raise ValueError('Source motion manifest changed')
    old_motions=yaml.safe_load(manifest.read_text())['motions']
    new_manifest=Path(env['motion_file']);new_motions=yaml.safe_load(new_manifest.read_text())['motions']
    for skill in ('loco','push'):
        def signature(path,rows):
            return [(hashlib.sha256((path.parent/r['file']).read_bytes()).hexdigest(),r['weight']) for r in rows]
        if signature(manifest,old_motions[skill])!=signature(new_manifest,new_motions[skill]):
            raise ValueError('PUSH/loco motions changed')
    projected=without_open_settings(old)
    target=deepcopy(env)
    # Paths, output/render options and runtime environment count are not objectives.
    for config in (projected,target):
        config['openPushRSI'].pop('bank_file')
        for key in ('motion_file','numEnvs','headless','video','enableDebugVis','cameraFollow','numActions','numObservations','render_device','seed','task_plan'):
            config.pop(key,None)
    if projected!=target:
        keys=[k for k in set(projected)|set(target) if projected.get(k)!=target.get(k)]
        raise ValueError('Migration changes additional settings: '+','.join(sorted(keys)))
    if source['train']['params']['network']!=yaml.safe_load(Path(train_path).read_text())['params']['network']:
        raise ValueError('Network changed')
    result=dict(weights)
    result['open_push_contract']=training_contract(env,train_path)
    result['relation_metadata']=checkpoint_metadata(env['relationReward'])
    result['relation_task_instance']=task_instance(env['relationGraph'],2,4)
    result['relation_experiment_config']=dict(source,env=deepcopy(env))
    result['push_only_migration']={'source_epoch':weights.get('epoch'),'change':'physical actors removed; inactive OPEN configuration removed'}
    return result
