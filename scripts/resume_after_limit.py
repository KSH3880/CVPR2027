"""One-shot continuation after a specified training process finishes its epoch limit."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path('/home/injesus1010/anaconda3/envs/tokenhsi118')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def identity(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if text[0] == 'Z':
            return None
        return text[19]  # process start time, field 22
    except FileNotFoundError:
        return None


def stop_known(pid, needle):
    path = Path(f'/proc/{pid}/cmdline')
    if path.exists() and needle.encode() in path.read_bytes():
        os.kill(pid, signal.SIGTERM)
        for _ in range(40):
            if identity(pid) is None:
                return
            time.sleep(.25)
        raise RuntimeError(f'{needle} did not stop')


def spawn(command, env, log):
    with Path(log).open('a') as stream:
        return subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)


def continue_training(plan):
    # Import torch only after the old GPU process has exited; checkpoint checks use CPU.
    import torch
    import yaml
    old = plan['training']
    resume_epoch = plan['expected_epoch']
    old_dir = Path(old['output_path'])
    checkpoint = Path(plan['checkpoint']) if plan.get('checkpoint') else next(old_dir.glob(f"*/nn/TokenHSIPushKD_{plan['expected_epoch']:08d}.pth"))
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert weights['epoch'] == plan['expected_epoch']
    assert weights['optimizer']['state']
    assert weights['push_kd_contract'] == read(old['contract'])
    del weights
    stamp = datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'output' / f'push4096_kd0p1_sharedv2_resume{resume_epoch}_unlimited_{stamp}'
    out.mkdir()
    cfg = yaml.safe_load((old_dir / 'train_config.yaml').read_text())
    cfg['params']['config']['max_epochs'] = -1
    train_cfg = out / 'train_config.yaml'
    train_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
    shutil.copy2(old_dir / 'env_config.yaml', out / 'env_config.yaml')
    write(out / 'previous_launch.json', old)
    env = os.environ.copy()
    for key in ('MAX_ITERATIONS', 'CHECK', 'PUSH_KD_ALLOW_COEF_CHANGE', 'PUSH_KD_GRAD_PROBE'):
        env.pop(key, None)
    env.update(PATH=str(RUNTIME / 'bin') + ':' + env['PATH'],
               LD_LIBRARY_PATH=str(RUNTIME / 'lib') + ':' + env.get('LD_LIBRARY_PATH', ''),
               PYTHONPATH=str(ROOT) + ':' + str(ROOT / 'tokenhsi'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CXX='/usr/bin/g++-11', MAX_JOBS='2',
               TOKENHSI_LOCAL_CUDA_COMPAT='1', TOKENHSI_SIM_BACKEND='shared',
               CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1',
               TORCH_EXTENSIONS_DIR=str(ROOT / 'output_etc/check/torch_extensions'))
    command = [str(RUNTIME / 'bin/python'), 'scripts/train_local.py', '--task', 'HumanoidTokenHSIPushKD',
               '--cfg_env', str(out / 'env_config.yaml'), '--cfg_train', str(train_cfg),
               '--motion_file', 'tokenhsi/data/dataset_push_kd.yaml', '--num_envs', '2048', '--headless',
               '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--pipeline', 'gpu', '--seed', '42',
               '--output_path', str(out), '--resume', '1', '--checkpoint', str(checkpoint)]
    proc = spawn(command, env, out / 'train.log')
    active = dict(old, pid=proc.pid, output_path=str(out), log=str(out / 'train.log'),
                  max_epochs=-1, resume_epoch=resume_epoch, resume_checkpoint=str(checkpoint),
                  previous_output_path=str(old_dir), status='starting',
                  started_at=datetime.datetime.now().astimezone().isoformat())
    for key in ('contract', 'last_logged_iteration', 'verified_completed_epochs'):
        active.pop(key, None)
    write(ROOT / 'output/active_training.json', active)
    write(out / 'launch.json', active)
    plan.update(status='launched', resumed_pid=proc.pid, resumed_output=str(out))
    write(plan.get('state_path', ROOT / 'output/auto_resume_after26000.json'), plan)
    for _ in range(600):
        summaries = list(out.glob('*/summaries'))
        if summaries:
            break
        if proc.poll() is not None:
            raise RuntimeError('Resumed training exited during startup')
        time.sleep(1)
    else:
        raise RuntimeError('No summaries after startup')
    connect_tensorboard(plan['tensorboard'], old_dir, out, summaries[0], env, resume_epoch)
    for _ in range(600):
        if f'ITER {resume_epoch+1}/unlimited' in (out / 'train.log').read_text():
            active.update(status='running', contract=str(summaries[0].parent / 'method_contract.json'),
                          last_logged_iteration=resume_epoch+1)
            write(ROOT / 'output/active_training.json', active)
            write(out / 'launch.json', active)
            return
        if proc.poll() is not None:
            raise RuntimeError(f'Resumed training exited before ITER {resume_epoch+1}')
        time.sleep(1)
    raise RuntimeError(f'Could not verify ITER {resume_epoch+1}')


def connect_tensorboard(tb, old_dir, out, summaries, env, resume_epoch=26000):
    from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.summary.writer.record_writer import RecordWriter
    old_view = old_dir / 'tensorboard_continuous'
    manifest = read(old_view / 'manifest.json')
    view = out / 'tensorboard_continuous'
    view.mkdir()
    for source in old_view.glob('events.out.tfevents.*.history'):
        shutil.copy2(source, view / source.name)
    offset = manifest['resume_time_offset_seconds']
    from tensorboard_history import write_history
    sources = sorted(Path(manifest['active_summaries']).glob('events.out.tfevents.*'))
    history = view / f"events.out.tfevents.{len(manifest['segments']):010d}.history"
    end = write_history(sources, history, resume_epoch, offset)
    manifest['segments'].append(dict(sources=[str(p) for p in Path(manifest['active_summaries']).glob('events.out.tfevents.*')], through_iter=resume_epoch, time_offset=offset))
    manifest.update(history_through_iter=resume_epoch, active_summaries=str(summaries), resume_time_offset_seconds=end)
    write(view / 'manifest.json', manifest)
    stop_known(tb.get('time_mirror_pid'), 'tensorboard_resume_mirror.py')
    stop_known(tb['pid'], 'tensorboard_local.py')
    display_env = dict(env, CUDA_VISIBLE_DEVICES='')
    python = str(RUNTIME / 'bin/python')
    mirror = spawn([python, 'scripts/tensorboard_resume_mirror.py', '--manifest', str(view / 'manifest.json')], display_env, view / 'mirror.log')
    for key in tb['runs']:
        if key.startswith('CURRENT'):
            tb['runs'][key] = str(view)
    spec = ','.join(k + ':' + v for k, v in tb['runs'].items())
    server = spawn([python, 'scripts/tensorboard_local.py', '--logdir_spec', spec, '--host', '127.0.0.1', '--port', '6007', '--reload_interval', '15'], display_env, tb['log'])
    tb.update(pid=server.pid, time_mirror_pid=mirror.pid, time_offset_seconds=end, logdir_spec=spec)
    write(ROOT / 'output/active_tensorboard.json', tb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    with (ROOT / 'output/auto_resume_after26000.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(args.plan)
        if plan['status'] != 'armed':
            raise RuntimeError('Plan is not armed')
        while identity(plan['training']['pid']) == plan['process_start_time']:
            time.sleep(10)
        active = read(ROOT / 'output/active_training.json')
        normal = 'MAX EPOCHS NUM!' in Path(plan['training']['log']).read_text()
        unchanged = active['pid'] == plan['training']['pid'] and active['output_path'] == plan['training']['output_path']
        if not normal or not unchanged or active['status'].startswith('stopped'):
            plan['status'] = 'cancelled_process_stopped_before_limit'
        else:
            try:
                continue_training(plan)
                plan['status'] = 'completed'
            except Exception as error:
                plan.update(status='failed', error=repr(error))
                raise
            finally:
                write(args.plan, plan)
        write(args.plan, plan)


if __name__ == '__main__':
    main()
