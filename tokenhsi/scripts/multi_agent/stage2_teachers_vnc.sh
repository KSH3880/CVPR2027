#!/bin/bash
export TEACHER_VIEW_MODE=vnc
exec bash "$(dirname "$0")/stage2_teachers_test.sh" "$@"
