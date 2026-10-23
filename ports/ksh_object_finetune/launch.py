#!/usr/bin/env python3
"""Run the independent snapshot; output/assets/cache stay inside its namespace."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

PORT = Path(__file__).resolve().parent
ROOT = PORT.parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def source_hashes():
    spec=json.loads((PORT/'UPSTREAM.json').read_text())
    source=Path(spec['source'])
    return {name:sha(source/name) for name in spec['sha256']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('train','test','audit'),required=True)
    p.add_argument('--variant',choices=('sit_chair','carry_chair','carry_table','joint_adapter','joint_head','carry_chair_adapter','carry_chair_copied_adapter'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--envs',type=int,default=2048)
    p.add_argument('--max-iterations',type=int,default=0)
    p.add_argument('--episode-length',type=int,default=600)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--policy-branch',choices=('base','adapt'),default='adapt')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--assets-per-category',type=int,default=None)
    p.add_argument('--task-graph',choices=('random_scenario','sit','carry_at'),default='random_scenario')
    a=p.parse_args();out=a.output.resolve()
    if a.checkpoint is not None:a.checkpoint=a.checkpoint.resolve()
    allowed=ROOT/'output/object_tokenizer_finetune'
    if allowed not in out.parents or out.exists():p.error('Use a new output under '+str(allowed))
    if a.policy_branch=='base' and (a.mode!='test' or a.variant!='carry_chair_copied_adapter'):p.error('Base branch is test-only for the copied-tokenizer variant')
    if a.mode=='train' and a.envs!=2048:p.error('Training requires 2048 environments')
    if a.mode=='test' and a.checkpoint is None:p.error('Test requires a native fine-tuned checkpoint')
    out.mkdir(parents=True)
    before=source_hashes();weight=PORT/'weights/source_39000.pth';weight_before=sha(weight)
    # Fix all imports to this snapshot; never import the main checkout runtime.
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(PORT),str(PORT/'tokenhsi')]
    os.chdir(PORT)
    from isaacgym import gymapi  # must precede torch
    import yaml
    from native_hooks import install
    install(out,'test' if a.mode=='test' else 'train')
    if a.mode in ('train','audit'):
        from agent_hooks import install as install_agent
        install_agent(out)
    stem='object_tokenizer_finetune_'+a.variant
    cfg=yaml.safe_load((PORT/'tokenhsi/data/cfg/multi_agent'/ (stem+'.yaml')).read_text())
    if a.assets_per_category is not None:
        cfg['env']['objectFineTune']['assets_per_category']=a.assets_per_category
    cfg_path=out/'resolved_env.yaml';cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    train_path=PORT/'tokenhsi/data/cfg/train/rlg'/(stem+'.yaml')
    if a.mode=='audit':
        train=yaml.safe_load(train_path.read_text())
        train['params']['config']['minibatch_size']=a.envs*2*train['params']['config']['horizon_length']
        train['params']['config']['amp_minibatch_size']=min(train['params']['config']['amp_minibatch_size'],
                                                          train['params']['config']['minibatch_size'])
        train_path=out/'audit_train.yaml';train_path.write_text(yaml.safe_dump(train,sort_keys=False))
    sys.argv=['run.py','--task','HumanoidMACarry','--cfg_train',str(train_path),
        '--cfg_env',str(cfg_path),'--motion_file','tokenhsi/data/dataset_loco_sit_carry_climb.yaml',
        '--num_agents','2','--num_objects','4','--num_envs',str(a.envs),'--seed',str(a.seed),
        '--task_graph',a.task_graph,'--output_path',str(out),'--experiment','ObjectTokenizerFineTune'+a.variant.title().replace('_',''),
        '--no_video','--episode_length',str(a.episode_length)]
    if a.headless:sys.argv+=['--headless']
    if a.mode=='test' and a.variant=='carry_chair_copied_adapter':
        from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
        from object_tokenizer_copy import set_branch
        original_build = AMPMultiAgentBuilder.build
        def build_with_branch(self, name, **kwargs):
            network = original_build(self, name, **kwargs)
            set_branch(network, a.policy_branch)
            return network
        AMPMultiAgentBuilder.build = build_with_branch
    if a.mode=='test':
        import torch
        ck=torch.load(a.checkpoint.resolve(),map_location='cpu',weights_only=False)
        if ck.get('object_finetune',{}).get('schema')!='native_sit_carry_v1':
            raise ValueError('Only native fine-tuned checkpoints are accepted by test')
        if ck['object_finetune']['variant']!=a.variant:raise ValueError('Fine-tune variant mismatch')
        sys.argv+=['--checkpoint',str(a.checkpoint.resolve()),'--test','--eval','--eval_repeats',str(a.repeats),
                   '--eval_skills','loco','--eval_skill_probs','1.0']
    elif a.checkpoint:
        sys.argv+=['--checkpoint',str(a.checkpoint.resolve()),'--resume','1']
    if a.max_iterations:sys.argv+=['--max_iterations',str(a.max_iterations)]
    if a.mode=='audit':
        from rsi_audit import install as install_audit
        install_audit(out)
    try:
        runpy.run_path(str(PORT/'tokenhsi/run.py'),run_name='__main__')
    finally:
        integrity=dict(upstream_code_and_configs_unchanged=source_hashes()==before,
                       initialization_checkpoint_unchanged=sha(weight)==weight_before)
        (out/'integrity.json').write_text(json.dumps(integrity,indent=2))
        if not all(integrity.values()):raise RuntimeError('Source integrity check failed')


if __name__=='__main__':main()
