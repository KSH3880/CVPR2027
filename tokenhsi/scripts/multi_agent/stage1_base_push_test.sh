#!/bin/bash
set -eu
. "$(dirname "$0")/stage1_base_push_common.sh"
stage1_base_push_run test "$@"
