#!/bin/bash
# Local entry used by the shared noVNC wrapper; view.sh sets viewer options.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$HERE/launch_ms18.sh" "$@"
