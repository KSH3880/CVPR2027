#!/bin/bash
set -e
ROOT=$(cd "$(dirname "$0")" && pwd)
exec bash "$ROOT/output_etc/check/view_relfusion_3500_random8/launch.sh"
