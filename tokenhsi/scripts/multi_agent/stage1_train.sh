#!/bin/bash
# stage1: train
exec bash "$(dirname "$0")/launch.sh" stage1 train "$@"
