"""Short full-size training run with per-process GPU and allocator measurements."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('VRAM_PROFILE_OUT', str(ROOT/'output_etc/check/vram_push70')))


def gpu_processes():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    return {int(a): int(b) for a, b in (line.split(', ') for line in text.strip().splitlines()) if b.isdigit()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(ROOT/'tokenhsi'), str(ROOT)]
    import isaacgym
    import torch
    from scripts.local_cuda_compat import install
    install()
    from learning.push_kd_agent import PushKDAgent
    baseline = gpu_processes()
    samples = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                rows = gpu_processes()
                selected = {p: m for p, m in rows.items() if p == os.getpid() or p not in baseline}
                samples.append(dict(time=time.time(), processes_mib=selected, total_mib=sum(selected.values())))
                (OUT/'gpu_samples.json').write_text(json.dumps(samples))
            except Exception as error:
                print('VRAM_MONITOR_ERROR', repr(error), flush=True)
            stop.wait(1)

    original = PushKDAgent.train_epoch
    def measured(self):
        torch.cuda.reset_peak_memory_stats()
        try:
            return original(self)
        finally:
            stats = dict(epoch=self.epoch_num, allocated_peak_gib=torch.cuda.max_memory_allocated()/2**30,
                         reserved_peak_gib=torch.cuda.max_memory_reserved()/2**30)
            print('VRAM_EPOCH '+json.dumps(stats), flush=True)
            with (OUT/'allocator.jsonl').open('a') as f:
                f.write(json.dumps(stats)+'\n')
    PushKDAgent.train_epoch = measured
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        runpy.run_path(str(ROOT/'tokenhsi/run.py'), run_name='__main__')
    finally:
        stop.set()
        thread.join(5)
        summary = dict(main_pid=os.getpid(), baseline_processes_mib=baseline,
                       sampled_training_peak_mib=max((s['total_mib'] for s in samples), default=0))
        (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
        print('VRAM_SUMMARY '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
