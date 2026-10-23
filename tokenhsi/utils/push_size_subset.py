"""Derive a smaller size domain from an existing all-sizes PD-screened bank."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from utils.open_push_rsi import bank_contract
from utils.open_push_sizes import size_values


def project_size_subset(source_env, target_env):
    source_path=Path(source_env['openPushRSI']['bank_file'])
    source=json.loads(source_path.read_text())
    old,new=bank_contract(source_env),bank_contract(target_env)
    if source['contract']!=old or source['screen']!='PD_12_steps':
        raise ValueError('Source bank must match the original PD screen contract')
    old_sizes,new_sizes=size_values(source_env),size_values(target_env)
    if not set(new_sizes)<set(old_sizes):
        raise ValueError('Target must be a strict subset of screened source sizes')
    old_without=deepcopy(old);new_without=deepcopy(new)
    old_without.pop('push_size_randomization');new_without.pop('push_size_randomization')
    if old_without!=new_without:
        raise ValueError('Cannot reuse screen after changing physics, motion, or reset geometry')
    for env in (source_env,target_env):
        if not env.get('pushOnly'):
            raise ValueError('Size subset projection requires PUSH-only profiles')
    result=deepcopy(source);result['contract']=new
    result['size_subset_provenance']=dict(source_bank=str(source_path),
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        screened_sizes_m=old_sizes,retained_sizes_m=new_sizes,
        method='Identical candidates passed the original PD screen across all source sizes; only the creation-time size domain is restricted. No new PD screen claimed.')
    return result
