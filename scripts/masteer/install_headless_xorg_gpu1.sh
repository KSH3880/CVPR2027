#!/bin/bash
# 이 로컬 PC의 GPU 1(PCI 05:00.0)에 DISPLAY=:43을 한 번 설치한다.
set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "사용법: sudo bash scripts/masteer/install_headless_xorg_gpu1.sh" >&2
    exit 1
fi

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
EXPECTED_BUS=00000000:05:00.0
ACTUAL_BUS=$(nvidia-smi -i 1 --query-gpu=pci.bus_id --format=csv,noheader)
if [ "${ACTUAL_BUS,,}" != "${EXPECTED_BUS,,}" ]; then
    echo "GPU 1 PCI 주소 불일치: 예상=$EXPECTED_BUS 실제=$ACTUAL_BUS" >&2
    exit 2
fi

install -o root -g root -m 0644 \
    "$ROOT/scripts/masteer/xorg_headless_gpu1.conf" \
    /etc/X11/tokenhsi-headless-xorg-gpu1.conf
install -o root -g root -m 0644 \
    "$ROOT/scripts/masteer/tokenhsi-headless-xorg-gpu1.service" \
    /etc/systemd/system/tokenhsi-headless-xorg-gpu1.service
systemctl daemon-reload
systemctl enable --now tokenhsi-headless-xorg-gpu1.service

for _ in $(seq 1 50); do
    if DISPLAY=:43 xdpyinfo >/dev/null 2>&1; then
        echo "GPU 1 headless Xorg 준비 완료: DISPLAY=:43"
        exit 0
    fi
    sleep 0.1
done

systemctl --no-pager --full status tokenhsi-headless-xorg-gpu1.service || true
echo "GPU 1 headless Xorg :43가 준비되지 않았다." >&2
echo "로그: /var/log/tokenhsi-headless-xorg-gpu1.log" >&2
exit 1
