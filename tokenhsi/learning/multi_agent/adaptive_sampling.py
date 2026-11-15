"""Training-boundary evaluation and checkpoint hand-off for adaptive PUSH."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from utils.experiments import ROOT
from utils.stage1_push_adaptive import TASKS, EVAL_CRITERION, advance, validate_state


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def evaluate(checkpoint, directory, spec):
    """Use region-ever success from the independent 1-person/loco evaluation."""
    command = [sys.executable, str(ROOT/'tokenhsi/evaluate.py'), 'stage1_push_adaptive',
               str(checkpoint), '--profile', 'stage1_single', '--envs', str(spec['eval_envs']),
               '--seeds', str(spec['eval_seed']), '--tasks', *TASKS, '--output', str(directory)]
    print('[adaptive] evaluating five tasks: '+str(directory), flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory/'command.json', command)
    with (directory/'evaluate.log').open('a') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = json.loads((directory/'summary.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    if (summary['runs'] != len(TASKS) or manifest['horizon'] != 900
            or manifest['envs'] != spec['eval_envs'] or manifest['seeds'] != [spec['eval_seed']]
            or manifest['profile'] != 'stage1_single' or manifest['tasks'] != list(TASKS)
            or manifest['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
        raise ValueError('Incomplete or mismatched adaptive evaluation')
    rates = {}
    for task in TASKS:
        row = summary['scores'][EVAL_CRITERION][task]
        if row['trials'] != spec['eval_envs']:
            raise ValueError('Incomplete adaptive evaluation for '+task)
        rates[task] = row['successes']/row['trials']
    return rates


def replay_state(buffer):
    """Persist replay only at environment rebuilds, not at regular 500-epoch saves."""
    count = min(buffer._total_count, buffer._buffer_size)
    return dict(size=buffer._buffer_size, head=buffer._head, total_count=buffer._total_count,
                sample_head=buffer._sample_head, sample_idx=buffer._sample_idx.clone(),
                data=None if buffer._data_buf is None else {
                    k: v[:count].detach().cpu().clone() for k, v in buffer._data_buf.items()})


def restore_replay(buffer, state):
    if state['size'] != buffer._buffer_size:
        raise ValueError('Adaptive AMP replay buffer size differs')
    count = min(state['total_count'], state['size'])
    buffer._head, buffer._total_count = state['head'], state['total_count']
    buffer._sample_head = state['sample_head']
    buffer._sample_idx = state['sample_idx'].cpu().clone()
    buffer._data_buf = None
    if state['data'] is not None:
        buffer._init_data_buf(state['data'])
        for key, value in state['data'].items():
            if len(value) != count:
                raise ValueError('Truncated adaptive AMP replay buffer')
            buffer._data_buf[key][:count].copy_(value.to(buffer._device))


def after_epoch(agent):
    task = agent.vec_env.env.task
    env = task.cfg['env']
    if 'adaptiveSampling' not in env:
        return False
    if getattr(agent, '_adaptive_logged_criterion', None) != EVAL_CRITERION:
        print('[adaptive] success criterion='+EVAL_CRITERION, flush=True)
        agent._adaptive_logged_criterion = EVAL_CRITERION
    spec = env['adaptiveSampling']
    state = agent._adaptive_sampling_state
    due = state['last_eval_epoch']+spec['eval_interval']
    if agent.epoch_num < due:
        return False
    if agent.epoch_num != due:
        raise ValueError('Adaptive evaluation boundary was skipped; resume its last checkpoint')
    directory = ROOT/'output_etc/check/stage1_push_adaptive'/Path(agent.experiment_dir).name/f'epoch_{due:08d}'
    # This snapshot precedes the decision and can also recover an interrupted eval.
    # Reuse it after a crash: its hash identifies cached evaluation results.
    source = Path(getattr(task.cfg['args'], 'checkpoint', 'Base'))
    if source.name == 'evaluation_snapshot.pth' and source.parent.parent == directory:
        directory = source.parent
        snapshot = source
    else:
        directory = directory/('attempt_'+uuid.uuid4().hex[:12])
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = directory/'evaluation_snapshot.pth'
        agent.save(str(snapshot.with_suffix('')))
    rates = evaluate(snapshot, directory, spec)
    new_state = advance(state, spec, due, rates)
    changed = new_state['probabilities'] != state['probabilities']
    validate_state(new_state, spec, agent.epoch_num)
    agent._adaptive_sampling_state = new_state
    env['_adaptiveSamplingState'] = deepcopy(new_state)
    task._adaptive_sampling_state = deepcopy(new_state)
    for t in TASKS:
        agent.writer.add_scalar('adaptive/eval_'+EVAL_CRITERION+'_'+t.lower(), rates[t], agent.frame)
        agent.writer.add_scalar('adaptive/probability_'+t.lower(), new_state['probabilities'][t], agent.frame)
        agent.writer.add_scalar('adaptive/mastered_'+t.lower(), float(new_state['mastered'][t]), agent.frame)
    agent.writer.add_scalar('adaptive/evaluation_epoch', due, agent.frame)
    agent.writer.flush()
    checkpoint = Path(agent.nn_dir)/f"{agent.config['name']}_adaptive_{due:08d}.pth"
    # Rebuild boundaries retain learned policy/discriminators, optimizer, all RMS,
    # frame/epoch counters and AMP replay. Expert demos are regenerated with the
    # new family mixture. Physical episodes start anew in compatible assets.
    agent._adaptive_save_replay = changed
    try:
        agent.save(str(checkpoint.with_suffix('')))
    finally:
        agent._adaptive_save_replay = False
    decision = dict(epoch=due, criterion=EVAL_CRITERION, rates=rates, previous=state, next=new_state,
                    rebuild=changed, checkpoint=str(checkpoint.resolve()))
    atomic_json(directory/'decision.json', decision)
    print('[adaptive] '+json.dumps(dict(epoch=due, rates=rates, focus=new_state['focus'],
          probabilities=new_state['probabilities'], reason=new_state['reason'], rebuild=changed)), flush=True)
    if changed:
        marker = os.environ.get('TOKENHSI_ADAPTIVE_RESTART_FILE')
        if not marker:
            raise ValueError('Adaptive training must use its managed train wrapper')
        atomic_json(marker, dict(checkpoint=str(checkpoint.resolve()), epoch=due,
                               sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
    return changed
