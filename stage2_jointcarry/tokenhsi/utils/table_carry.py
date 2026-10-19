"""Explicit contract for the under-table carry fine-tune; unrelated runs stay unchanged."""
from pathlib import Path
import hashlib
import json
import torch


def verify_files(spec):
    for path_key,hash_key in [('asset','asset_sha256'),('initial_states','initial_states_sha256')]:
        path=Path(spec[path_key])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec[hash_key]:
            raise ValueError('Table carry input changed: '+str(path))
    if spec['version']!=1 or spec['size']!=[2.,1.2,.86] or abs(spec['mass']-60.5184)>1e-5:
        raise ValueError('Unsupported rectangular table contract')
    if spec['anchor_z']!=.351 or spec['initial_skill']!='carryWith':
        raise ValueError('Under-table carry requires matching anchors and held RSI')


def warmstart(agent,checkpoint,path):
    """Copy policy/discriminator/normalizers, keeping fresh PPO optimizer and counters."""
    from utils.joint_carry_amp import check_ablation_checkpoint
    from utils.relation_task_spec import checkpoint_metadata,check_checkpoint_metadata
    import copy
    env=agent.vec_env.env.task.cfg['env'];source_env=copy.deepcopy(env)
    source_env.pop('tableCarry');source_env['relationReward']['joint_carry'].pop('anchor_z')
    check_ablation_checkpoint(checkpoint,source_env)
    check_checkpoint_metadata(checkpoint,checkpoint_metadata(source_env['relationReward']))
    if checkpoint.get('table_carry') is not None:
        raise ValueError('Initial fine-tune must start from the original box grasp-view policy; use resume for table checkpoints')
    agent.model.load_state_dict(checkpoint['model'],strict=True)
    agent.set_stats_weights(checkpoint)
    report=dict(source_checkpoint=path,source_epoch=checkpoint.get('epoch'),
        model_tensors=len(checkpoint['model']),optimizer='fresh',epoch='reset',
        normalizers='source actor/value/AMP',actor_encoder='frozen as in source Stage2',
        table_contract=env['tableCarry'])
    Path(agent.experiment_dir,'table_warmstart.json').write_text(json.dumps(report,indent=2)+'\n')
    print('TABLE_WARMSTART',json.dumps(report),flush=True)
    return report
