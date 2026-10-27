#!/bin/bash
set -eu
exec bash "$(dirname "$0")/ksh_jointpush_launch.sh" train
