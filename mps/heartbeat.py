"""Print a heartbeat only after a real CUDA operation completes."""

import argparse
import os
import signal
import time
from datetime import datetime

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Display label, e.g. GPU4")
    args = parser.parse_args()
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    value = torch.zeros(1024, dtype=torch.int64, device="cuda:0")
    print(
        f"START {args.label} pid={os.getpid()} "
        f"device={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"pipe={os.environ.get('CUDA_MPS_PIPE_DIRECTORY')}",
        flush=True,
    )
    while not stopping:
        value.add_(1)
        torch.cuda.synchronize()
        print(
            f"{datetime.now().isoformat(timespec='seconds')} "
            f"{args.label} pid={os.getpid()} step={value[0].item()} CUDA_OK",
            flush=True,
        )
        time.sleep(1)

    torch.cuda.synchronize()
    print(f"STOP {args.label}", flush=True)


if __name__ == "__main__":
    main()
