"""One-shot Chair residual reservation after a specific Table run succeeds."""
import argparse
import datetime
import fcntl
import json
import os
import runpy
from pathlib import Path
import subprocess
import time


def read_json(path):
    return json.loads(Path(path).read_text())


def predecessor_status(config):
    marker = Path(config['predecessor_exit'])
    if not marker.is_file():
        return 'waiting'
    result = read_json(marker)
    if result.get('status') != 'complete' or result.get('returncode') != 0:
        raise RuntimeError('Table predecessor did not finish successfully')
    if result.get('total_epochs') != 10000 or result.get('run') != config['predecessor_run']:
        raise RuntimeError('Table completion marker does not match the reserved predecessor')
    if not Path(config['predecessor_checkpoint']).is_file():
        raise RuntimeError('Table completed without its numbered 10000 checkpoint')
    return 'complete'


def process_identity(pid):
    try:
        proc = Path('/proc') / str(pid)
        stat = (proc/'stat').read_text().split(') ', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(pid=pid, start_ticks=stat[19], cmdline=(proc/'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except (FileNotFoundError, ProcessLookupError):
        return None


def matching_process(identity):
    actual = process_identity(identity['pid'])
    return actual is not None and actual['start_ticks'] == identity['start_ticks']


def other_chair_training():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc/'cmdline').read_bytes()
            if b'--task\x00HumanoidChairTokenHSIFT\x00' in cmd and b'--test\x00' not in cmd:
                return int(proc.name)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return None


def validate_output_paths(config):
    code = Path(config['code'])
    preflight = code/'output_etc/check/chair_preflight/train'
    resolver = runpy.run_path(str(code/'tokenhsi/utils/output_paths.py'))['resolve_output_path']
    resolver('chair_adapt_residual', 'train', str(preflight), check=True, max_iterations=2, root=code)
    resolver('chair_adapt_residual', 'train', str(Path(config['run'])/'train'), root=code)
    return preflight


def main(config_path):
    config = read_json(config_path)
    preflight_output = validate_output_paths(config)
    root = Path(config['run'])
    lock = (root/'scheduler.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root/'state.json').exists() and read_json(root/'state.json').get('launch_attempted'):
        raise RuntimeError('This reservation has already attempted a launch; refusing duplicate training')
    state = dict(status='scheduled', phase='waiting_table', supervisor_pid=os.getpid(),
                 gpu=config['gpu'], run=str(root), predecessor_run=config['predecessor_run'],
                 predecessor_pid=config['predecessor_process']['pid'], total_epochs=10000,
                 num_envs=4096, source_epoch=78000, mode='residual', launch_attempted=False,
                 training_pid=None, wrapper_pid=None)
    active = Path(config['active_metadata'])

    def publish(**updates):
        state.update(updates)
        state['updated_at'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
        for path in (root/'state.json', active):
            temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
            temp.write_text(json.dumps(state, indent=2))
            temp.replace(path)
        print(state['updated_at'], state['status'], state['phase'], flush=True)

    try:
        publish()
        orphan_since = None
        while predecessor_status(config) != 'complete' or matching_process(config['predecessor_process']):
            if not matching_process(config['predecessor_process']) and predecessor_status(config) == 'waiting':
                orphan_since = orphan_since or time.monotonic()
                if time.monotonic() - orphan_since > 120:
                    raise RuntimeError('Table exited without a successful completion marker')
            publish(status='scheduled', phase='waiting_table')
            time.sleep(10)
        import torch
        final = torch.load(config['predecessor_checkpoint'], map_location='cpu', weights_only=False)
        assert final['epoch'] == 10000
        assert all(torch.isfinite(v).all() for v in final['model'].values())
        del final
        while True:
            occupied = other_chair_training()
            query = subprocess.check_output(['nvidia-smi', '-i', str(config['gpu']),
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
            free = int(query.strip())
            if not occupied and free >= config['minimum_free_mib']:
                break
            publish(status='scheduled', phase='waiting_gpu', gpu_free_mib=free, existing_chair_pid=occupied)
            time.sleep(10)
        # Fixed snapshot and explicit launch environment; never inherit CHECK/resume/RSI overrides.
        from hashlib import sha256
        code = Path(config['code'])
        for name, digest in read_json(root/'code_sha256.json').items():
            assert sha256((code/name).read_bytes()).hexdigest() == digest, name
        env = os.environ.copy()
        for key in ('CHECK','MAX_ITERATIONS','RESUME_CHECKPOINT','OUTPUT_PATH','CUDA_VISIBLE_DEVICES',
                    'CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','TABLE_RSI_VIEW','TABLE_RSI_PROB','TABLE_RSI_BANK'):
            env.pop(key, None)
        env.update(TOKENHSI_GPU=str(config['gpu']), TOKENHSI_CONDA_ENV='tokenhsi118', PYTORCH_JIT='0',
            LD_LIBRARY_PATH='/home/injesus1010/anaconda3/envs/tokenhsi118/lib',
            TORCH_EXTENSIONS_DIR='/tmp/table_hold_at_torch_extensions', SEED='42', HEADLESS='1',
            PYTHONDONTWRITEBYTECODE='1', HOLD_AT_OBJECT_ROOT=config['object_root'])
        command = ['bash', str(code/'tokenhsi/scripts/multi_agent/chair_adapt_residual_train.sh')]

        def run_train(phase, output, extra):
            launch_env = dict(env, OUTPUT_PATH=str(output), **extra)
            publish(status='starting', phase=phase, launch_attempted=True, output_path=str(output))
            with (root/(phase+'.log')).open('ab') as log:
                child = subprocess.Popen(command, cwd=code, env=launch_env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                publish(status='running', wrapper_pid=child.pid)
                for _ in range(60):
                    if child.poll() is not None:
                        break
                    try:
                        children = Path(f'/proc/{child.pid}/task/{child.pid}/children').read_text().split()
                        pid = next((int(p) for p in children if b'--task\x00HumanoidChairTokenHSIFT' in Path(f'/proc/{p}/cmdline').read_bytes()), None)
                        if pid:
                            publish(training_pid=pid)
                            break
                    except FileNotFoundError:
                        pass
                    time.sleep(1)
                rc = child.wait()
            (root/(phase+'_exit.json')).write_text(json.dumps(dict(returncode=rc, phase=phase), indent=2))
            publish(wrapper_pid=None, training_pid=None, returncode=rc)
            if rc != 0:
                raise RuntimeError(f'Chair {phase} failed with returncode {rc}')

        check = preflight_output
        run_train('preflight', check, dict(CHECK='1', MAX_ITERATIONS='2'))
        candidates = list(check.glob('ChairAdaptResidual_*/nn/*.pth'))
        assert candidates, 'Preflight checkpoint missing'
        weights = None
        for path in candidates:
            candidate = torch.load(path, map_location='cpu', weights_only=False)
            if candidate['epoch'] == 2:
                weights = candidate
                break
        assert weights is not None, 'No completed epoch2 preflight'
        source = torch.load(code/'checkpoints/Stage1V2Fixed_00078000.pth', map_location='cpu', weights_only=False)
        experiment = path.parent.parent
        flags = read_json(experiment/'adaptation_trainable.json')
        assert sum(flags.values()) == 18
        frozen = [k for k,v in flags.items() if not v]
        assert len(frozen) == 153
        assert all(torch.equal(weights['model'][k], source['model'][k]) for k in frozen)
        assert all(torch.equal(v, source['running_mean_std'][k]) for k,v in weights['running_mean_std'].items())
        assert all(torch.isfinite(v).all() for v in weights['model'].values())
        for i in (1,2):
            assert sum(read_json(experiment/f'adaptation_gradient_{i}.json').values()) > 0
        (root/'preflight_validation.json').write_text(json.dumps(dict(passed=True, epoch=2, num_envs=4096,
            trainable_tensors=18, frozen_tensors=153, scene_rms_exact=True, positive_adapter_gradient=True), indent=2))
        del weights, source, candidate
        # Main training starts fresh at78k, not from the two synthetic preflight updates.
        run_train('train', root/'train', {})
        finals = list((root/'train').glob('ChairAdaptResidual_*/nn/ChairAdaptResidual_00010000.pth'))
        assert len(finals) == 1, 'Final Chair10000 checkpoint missing'
        final = torch.load(finals[0], map_location='cpu', weights_only=False)
        assert final['epoch'] == 10000 and all(torch.isfinite(v).all() for v in final['model'].values())
        publish(status='complete', phase='complete', final_checkpoint=str(finals[0]))
        (root/'completed.json').write_text(json.dumps(state, indent=2))
    except Exception as error:
        publish(status='failed', phase=state['phase'], error=repr(error))
        (root/'failed.json').write_text(json.dumps(state, indent=2))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    main(parser.parse_args().config)
