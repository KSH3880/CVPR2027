Corrected cooperative RSI snapshots

Each skill has a complete corrected reference and screened_rsi_snapshots.npz.
Snapshots are independent initial states, not a concatenated expert trajectory.
The correction uses physical 32-DOF elbows/wrists, arm IK, and a constant 1.4-2.6cm
root lift per clip. Leg/torso/head joint rotations are unchanged. Box dimensions
and the 15cm interior grip placement are retained. Arm DOFs can change up to
42 degrees: this is not an unchanged-joint rigid translation.

Snapshot root_state: [sample,2,13] = xyz, quaternion xyzw, linear velocity xyz,
angular velocity xyz. dof_position/dof_velocity: [sample,2,32].
box_state: [sample,1,13], one shared 16.64kg box at density 100kg/m^3.
source_frame/source_time identify the frame in the full corrected reference.

Screen: native asset PD at the initial pose, free roots, CPU PhysX 0.1s,
repository shock thresholds plus <=2cm initial geometry penetration and original
RSI exclusions. CarryWith also requires both people to make hand contact for
at least 3/6 steps. The report describes geometry approximations and limits.

This is a short reset screen, not long-term stability certification. Clone foot
sliding remains in the continuous reference. Native-PD playback still loses
balance; no trained balancing controller was used. Current training reset loads these paired snapshots through
tokenhsi/utils/joint_carry_rsi.py.
Use the snapshot state arrays, not the full video, as candidate initial states.
