#!/bin/bash
# GPU·큐 상태 조회 진입점. Markdown 수정은 queue_status.py 한 곳만 담당한다.
#
#   bash scripts/gpumap.sh
#   bash scripts/gpumap.sh --write
set -u
exec python3 /home/hwanhee/CVPR2027/scripts/queue_status.py "$@"
