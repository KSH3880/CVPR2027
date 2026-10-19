TeamHOI algorithm reference (MIT, Sea AI Lab 2026).

Commit: `6fdc9885f8a9c82854be99adca6f2570e67ac63a`.
Source: https://github.com/sail-sg/TeamHOI

Adapted dual-discriminator, pre-normalization body masking, separate window RMS, distance-history sigmoid blending, outside-only full discriminator updates and paired replay into `tokenhsi/learning/multi_agent/dual_amp*` and `tokenhsi/utils/dual_amp.py`. Native 129-D features and fixed two-agent scene batching differ from TeamHOI125-D/variable-agent padding. Original source files were inspected in /tmp; this directory preserves provenance/license.
