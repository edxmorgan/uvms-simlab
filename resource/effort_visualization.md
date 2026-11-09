# RViz vehicle and arm efforts

The normal `robot_system_multi_interface.launch.py` launch starts the read-only
`simlab/effort_visualization` adapter when RViz is enabled. No control commands,
EKF updates or dynamics changes are made by this node. Headless launches with
`launch_rviz:=false` do not start it. Multi-robot and `robot_real_` prefixes use
the same binding scheme; hardware is not exercised by the verification tests.

RViz groups each robot's effort displays under `Robot … efforts`:

- **Orange:** vehicle controller command / arm actuator effort. Vehicle values are the body wrench command
  read back through `IOs/force.*` and `IOs/torque.*`; arm values are
  `effort` (current-derived actuator effort after limiting and simulated motor-off locks). These are not
  force/torque sensor measurements. In PWM mode, the vehicle readback is the
  thruster-mapped body wrench command.
- **Blue:** estimated external disturbance, where the estimator supplies a
  valid, observable estimate. These are not controller commands or measured
  contact forces. Locked joints and joints at a stop show status text instead
  of a misleading estimate. The existing estimator's observability flag is
  respected; this view does not invent its own observability claim.
- **Force:** straight arrow, newtons. **Torque:** circular direction indicator (and a native axis arrow for the vehicle), newton-metres, right-hand rule.
- The arm's gripper is prismatic and shows actuator force only. There is no
  gripper disturbance estimator, so no estimated gripper wrench is fabricated.

Vehicle surge, sway, heave, roll, pitch and yaw commands and external disturbance
estimates appear in the left column, with the arm table alongside on the right,
in the same bottom-aligned 2D panel. Vehicle numeric signs
use the dynamics body FRD convention (forward/right/down);
the 3D wrench graphics still use the corresponding FLU transform. Translational
rows are in N and rotational rows in Nm. Invalid or missing vehicle estimates
show unavailable, not zero.

Per-joint signed values and estimator status appear in the compact 2D
`Effort values and status` panel. This display and each wrench can be toggled
independently to reduce clutter. The panel reserves room for high-DPI fonts;
RViz position/font overrides can resize it for smaller screens. Optional `Joint effort labels (3D)` are
disabled by default: they overlap in compact arm configurations and can be
hard to read in the 3D view. A valid zero has a numeric zero label but no vector geometry.
Small values are rounded to three decimals in text; wrench topics retain full
precision.

## Frames and data

The adapter subscribes only to `/dynamic_joint_states` with sensor-data QoS,
then publishes at 20 Hz. The source stamp is retained. This is a latest-sample
visualisation, not a recorder. Full-rate data remain in the original topics.

Vehicle dynamics use FRD; URDF vehicle frames use FLU. The proper rotation
`diag(1,-1,-1)` is applied to both force and torque, exactly once. Vehicle
wrenches are located at `<prefix>base_link`.

For every arm joint, its axis, type and child-link frame are read from the
actual `robot_description`. A revolute scalar torque is multiplied by the
normalised joint axis; a prismatic scalar effort becomes force along that
axis. The joint axis is invariant under its own joint rotation, so the child
frame supplies the correct direction with normal robot TF. No guessed world
axes or extra gravity wrenches are used. Joint b's negative-Z URDF axis is
explicitly covered by tests.

Topics: `/<prefix>/visualization/effort/{vehicle,e,d,c,b,a}/{command,estimated_disturbance}`.
There is no `a/estimated_disturbance`. Text markers use
`/effort_visualization/labels`; the 2D status panel uses
`/effort_visualization/status`. Existing `/fts_broadcaster_*/wrench` topics are
unchanged; the new RViz config replaces their duplicate display.

Missing, invalid or stale (>0.5 seconds without receipt) data publish NaN on
these **visualisation-only** wrench topics. The vehicle Wrench display uses
`Accept NaN Values` to hide them; the arm markers are explicitly deleted.
Labels say unavailable, locked, unobservable or stale.
This does not publish a fake zero measurement. If the entire adapter dies,
RViz's native Wrench display retains its last sample: check the topic status
before treating a static display as current. Optional 3D text markers expire after one second. The 2D overlay, like native
wrenches, retains its last frame if this adapter process itself dies.

## Display scaling

Vehicle display remains the native Wrench plugin: force 0.3 m/N, torque
0.5 m/Nm, width0.3 m, full opacity. A 3 N force gives a 0.9 m arrow,
extending beyond the hull more readily than the previous 0.3 m arrow.
Very small vectors can still be occluded by the opaque vehicle; no minimum
magnitude or nonlinear amplification is applied.

Arm displays use standard RViz MarkerArray geometry with **linear scaling**:

- Torque arc radius = 0.25 metres per N m times absolute torque.
  For example0.1Nm gives2.5cm radius;1Nm gives25cm radius.
- Gripper arrow length = 0.1 metres per N times absolute force.
- Stroke/shaft width remains12mm for visibility. Head dimensions scale with
  magnitude too. No minimum size, upper cap or nonzero display deadband.
- Zero or unavailable values delete the geometry. Torque direction follows
  the right-hand rule; force direction follows the joint axis and sign.
- Command and estimate arcs are offset25mm in opposite directions along the
  joint axis for separation. This placement offset does not depend on magnitude.
- Markers expire after1s without refresh. Source-data staleness deletes after
  0.5s. Latest TF with frame locking keeps them attached between updates;
  raw wrench timestamps and all numerical data remain unchanged.

Per-joint graphics use the corresponding wrench topic with `/markers` appended.
Vehicle styling is unchanged. Tests cover linear proportionality (including
very small and large values), thickness, sign, invalid/zero clearing and TF.

## Locked joints and motor-off behavior

The simulated lock is a motor-off brake. Hysteresis evaluates the freshly
current/position-limited torque request (lock below 0.05 Nm, unlock above
0.10 Nm by default), then zeros motor current and actuator effort while locked.
The same gated effort feeds plant integration and the completed-step EKF input.
External loads remain present and are resisted by the constraint; brake reaction
is not motor effort and is not displayed as a measured torque.
`computed_effort` preserves the controller request, including during a lock.
Hardware `effort` remains derived from measured motor current, never artificially
zeroed by this viewer. Orange arm graphics use `effort`; the legacy `/command`
visualization topic names are retained for layout compatibility.
Blue disturbance graphics remain hidden when locked/unobservable.
