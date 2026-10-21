# Private ksh Stage 1 port

This snapshot runs independently from the main checkout. Source hashes and
origin are recorded in `UPSTREAM.json`; large read-only assets link back to
the main checkout. Use the main `tokenhsi/scripts/multi_agent/ksh_*` launchers.

See [Korean usage and isolation guide](../../markdowns/ksh_stage1_port.md).
Run CPU checks with `bash ports/ksh_stage1/verify.sh` from the repository root.
