# BONES-SEED left-hand push AMP reference

Source: `bones_seed/soma_uniform/bvh/231115/inside_door_handle_right_side_open_walk_R_001__A515_M.bvh` (mirrored source; left hand opens towards left).

The source stays read-only. The generated `motions/.../phys_humanoid_v3/ref_motion.npy` is a Poselib SkeletonMotion: local XYZW rotations `(103,15,4)`, world Z-up/meter root translation `(103,3)`, 30 FPS, 3.4 seconds. SOMA's absolute Hips channels are used without adding Hips OFFSET. Original temporal opening/forward-motion description is 1.25932–5.2s; crop 0–3.4s retains reach, opening and early forward motion, before the late stop. This crop is provisional and does not establish full doorway passage.

Retargeting uses source positions and anatomical orientations, humanoid bone lengths, constrained 32-DOF IK, negative-Y elbow hinges and fixed hand rotations. Positions are interpolated and source rotations use SLERP. A constant grounding offset keeps the lowest foot-mesh vertex above the floor; velocities are recomputed by SkeletonMotion. The conversion is calibrated for this selected Uniform clip, not a general BONES retargeter.

`conversion_report.json` records source SHA256, mapping, scale, crop/resampling, endpoint errors and limits. `door_preview.html`, `door_preview.gif` and `door_contact_sheet.jpg` show the target humanoid and current push-door geometry. **The preview door angle is inferred from hand XY direction, not measured object data, and is never passed to AMP.** No door pose, grasp, contact labels or second actor is available in the source. The closed-door reset and OPEN reward remain simulator-owned.

AMP manifest: `tokenhsi/data/dataset_door_bones_left.yaml`. Separate experiment: `door_open_bones_left_amp`. Reset remains loco-only; initial OPEN expert mixture is loco 30% / doorOpen 70%, unconditioned/full AMP. The mixture is a starting choice, not a measured optimum. Existing experiments/checkpoints stay separate.

Reproduction and train/test/VNC commands are in `markdowns/config.md`, section **BONES Door — 왼손·왼쪽 개방 AMP**. Non-training validation results are in `output/door_bones_left_amp_check/validation_report.json` and `REPORT.md`. Training and policy success have not been tested.
