Controller performance metrics: definitions and interpretation
=============================================================

Purpose and compatibility
-------------------------

The live metrics retain the existing ``ControllerPerformanceMetrics`` accumulator,
``ControllerPerformance`` ROS message and RViz Robot Control Status overlay.
They describe tracking over the current active behavior window. They do not
replace the recorded benchmark protocol or establish a controller ranking.

Metric version 2 corrects geometric attitude, vector frames, input validity and
time integration. Existing names and the eight-term tracking-score composition
remain recognizable, but numbers from different metric versions must not be
pooled. Preserve old recordings and label their version; recompute only when
original state, reference and timestamps are available. A successful computation
is not a successful tracking trial.

Input contract
--------------

Current and reference data must contain finite ``pose`` (six components),
``body_vel`` (six), ``body_acc`` (six), and ``q``, ``dq``, ``ddq`` (at least four).
The first four arm coordinates are e,d,c,b; a fifth gripper coordinate does not
participate in the legacy arm score. Current ``sim_time`` is the measurement's
simulation timestamp, not the overlay publication time. Zero is a valid timestamp;
missing data is not zero. The metric producer must also check feedback freshness:
republication of a cached state is not a new measurement. The producer supplies
``feedback_valid`` and monotonic receive age ``feedback_age_sec``. Receive age
detects missing ROS deliveries, not the age of each sensor inside a delivered
message; it must not be advertised as hardware sensor-time synchronization.

Pose stores world position and roll/pitch/yaw with the existing extrinsic xyz
rotation convention. Let R map current body vectors into the world frame, and
Rd map desired body vectors into that same world frame. Body velocity contains
linear velocity v and angular velocity omega. Body acceleration contains the
component derivatives dv/dt and domega/dt in the moving body frame, not Euler
angle derivatives and not inertial linear acceleration already rotated to body.

Effort, accumulated mechanical work and tool pose are separate optional metric
groups. Their absence does not manufacture zero effort/energy/tool error and
does not invalidate otherwise complete vehicle/arm tracking data. Each group
needs an availability indication. Inactive or invalid tracking displays as
inactive/unavailable, never as zero error or confirmed settling.

Geometric and physical tracking terms
-------------------------------------

* Position error is desired minus current world position, in metres.
* Attitude distance is the principal rotation angle
  ``||Log(Rd.T R)^vee||``, in [0, pi] radians. Euler angles are merely the input
  representation: their components are never subtracted for attitude error.
* Linear velocity error compares ``Rd vd - R v``, in metres/second.
* Angular velocity error compares ``Rd omega_d - R omega``, in radians/second.
  Its norm equals that obtained by transporting the desired vector into the
  current body before subtraction.
* Inertial linear acceleration error compares
  ``Rd (dvd/dt + omega_d cross vd) - R (dv/dt + omega cross v)``.
  Rotating body derivatives alone omits moving-frame acceleration.
* Angular acceleration error compares ``Rd domega_d/dt - R domega/dt``.
* Arm errors subtract matching actuator coordinates and their derivatives.
  Do not wrap limited-joint positions through 2*pi merely to reduce the error.

The cross-track and along-track terms retain their existing meaning: project
world position error onto the target trajectory tangent. The tangent is supplied
explicitly, taken from desired world linear velocity, or inferred from
reference-position motion. Along-track is signed;
its normalized score term uses its absolute value. Cross-track is the norm of
the perpendicular remainder. Without a usable tangent, all position error is
classified as cross-track and along-track is zero. This is reference-point
tracking, not shortest distance to an entire geometric path.

Normalization and the retained combined score
---------------------------------------------

Defaults retain the existing scales:

* Position/cross-track: 0.4 m.
* Along-track: max(desired speed times 1 s, 0.4 m).
* Attitude: pi radians.
* Linear velocity: max(desired speed, 0.1 m/s).
* Linear acceleration: max(desired inertial acceleration norm, 0.1 m/s^2).
* Angular velocity/acceleration: corresponding reference angular norm with
  separately named angular floors (0.1 rad/s and 0.1 rad/s^2); these are not
  linear units.
* Arm position: four-joint RMS divided by pi radians.
* Arm velocity/acceleration: four-joint error RMS divided by desired joint RMS,
  with positive floors in the matching physical units.

The existing tracking score is the root mean square of eight normalized terms:

  cross-track, along-track, attitude, linear velocity, linear acceleration,
  arm position, arm velocity, arm acceleration.

Vehicle angular velocity/acceleration remain separate metrics, as does tool
tracking. They are not silently inserted into the historical composite. Lower
score means smaller error under these declared scales; it is not a measure of
robustness, efficiency or physical feasibility. Reference-dependent scales mean
that unlike trajectories cannot be ranked by this score alone. Compare physical
errors alongside it and freeze scales across paired controller trials.

Time accumulation and data gaps
--------------------------------

RMS is ``sqrt(integral(error^2 dt) / valid_duration)``. The implementation
approximates the integral by trapezoids between valid, strictly increasing
measurement timestamps. The first point has no duration and cannot independently
establish a time RMS. Sample count is an observation count, not elapsed time.

Do not bridge missing, invalid or overlong intervals. Reanchor after an interval
exceeding the configured maximum sample gap (default 0.5 s). Duplicate samples
must not increase count, integrals or dwell. A clock reversal cannot be treated
as ordinary positive elapsed time; it must invalidate/reset temporal continuity.
A new inactive-to-active behavior starts a new measurement window.

Only instantaneous error/effort signals receive RMS statistics. Do not compute
RMS of first-entry sentinel values, cumulative ratios or running maxima. An
unavailable RMS has no valid integrated support and must remain unavailable.
``valid_duration_sec`` counts supported tracking intervals; ``coverage_fraction``
is this duration divided by elapsed active-window time. Optional effort/tool RMS
use their own valid support and must not borrow tracking intervals where their
inputs were unavailable.

First entry and sustained score-band status
-------------------------------------------

``time_to_tolerance_sec`` retains its historical definition: first time after
behavior activation that the composite score is at or below its threshold
(default 0.1). It is not revoked by a later departure and is not a settling-time
claim. Before first entry, the documented sentinel is -1.

``settling_time_sec`` is the start of the currently confirmed in-band interval,
relative to activation; it is -1 until confirmed or after confirmation is lost.
``within_tolerance_duration_sec`` gives its current duration.

The additional sustained status requires continuous valid observations inside
the same score band for the configured dwell (default 1 s). Leaving the band or
breaking measurement continuity revokes the status and starts a new dwell.
Report this explicitly as score-band dwell/settling, not tool-pose settling.
A moving reference can be tracked inside the band without reaching an endpoint.
Conventional endpoint settling needs a declared reference-completion time and
physical position, orientation and speed tolerances in the offline protocol.

Effort, work and power
---------------------

Keep tracking and effort as separate outcomes. The legacy normalized effort
uses configured force, torque and arm effort scales; it is not automatically a
fraction of actuator capacity. Any cap-based saturation statistic must use the
actual per-actuator limits and state whether it is per-component occupancy or
fraction of time any component saturates.

``effort_per_tracking_score`` is retained only as a compatibility diagnostic.
It is an effort/error ratio, not efficiency: reducing tracking error can make it
arbitrarily large, and either minimizing or maximizing it can reward undesirable
behavior. Do not use it for ranking or gain selection.

The simulator's energy counters integrate the sum of absolute generalized
mechanical powers, ``sum(abs(effort_i * velocity_i))``. They are an absolute
mechanical-work proxy, not electrical consumption or signed net work. At rest,
holding effort can consume electrical energy despite zero mechanical work.
Vehicle and arm counters must refer to a common observation window; counter
resets and unavailable values cannot silently become zero consumption.

``energy_per_second`` is average absolute mechanical power over intervals with
valid energy-counter increments. ``energy_per_meter`` uses actual traveled
distance over those same intervals; there is no artificial 0.4 m denominator
floor in version 2. Zero distance or unsupported duration makes the corresponding
ratio unavailable. Excess travel or noise can increase the denominator, so it
is not a standalone efficiency ranking for spiraling trajectories.

Tool tracking and display
-------------------------

Tool position and orientation are supplementary physical errors in metres and
principal SO(3) radians. The producer evaluates the existing forward kinematics
for current vehicle/arm state and for commanded vehicle/arm references, using
the same mounting and tool geometry as IK. Both use the same NWU frame after
the existing NED-to-NWU conversion. No asynchronous TF lookup is needed.

These fields measure tracking of the commanded joint/vehicle reference, not
error against the original requested Cartesian marker endpoint. Preserve that
distinction for a four-joint arm whose requested orientation may be infeasible.
Kinematic calibration errors are also outside this model-based pose comparison.

RViz should identify inactive/unavailable groups and show score/RMS with the
existing nXTE, nATE, nVel, nAtt and nArm values. Tool errors, score-band dwell and
valid duration complement those fields. Work and power labels must state their
mechanical meaning. Speed, configured payload/gravity and waypoint status are
state/configuration displays, not tracking benchmarks.

Benchmark boundaries
--------------------

These changes improve measurement correctness. They do not recover missing raw
recordings, rerun historical controller comparisons or demonstrate DR robustness.
Those require new paired trials with preserved code/configuration/seed versions,
actual applied disturbances, an active observer, and physical tracking/effort
results. Live composite score alone cannot support a superiority claim.

MCAP controller attribution and metric inputs
--------------------------------------------

Each robot now records two additional versioned JSON ``std_msgs/String`` topics:

* ``/<prefix>/controller/context``: selected controller, bound arm and vehicle
  implementations, effective configuration snapshots, control mode/space,
  timestamp, process session UUID, revision, and configuration hash. A selected
  replay controller may be armed while a different feedback controller remains
  bound. In joystick or stopped modes, a bound controller is not evidence that
  its output was executed; consult control mode and published commands.
* ``/<prefix>/performance/kinematics``: the exact current and reference samples
  supplied to live metric calculation, computed tool poses and geometry, metric
  results, context session/revision, and the published efforts after execution
  gating. Timestamp matches the corresponding ``performance/controller`` sample.

Context is reliable/transient-local with depth one. Starting a recording after
selection still receives the current context. Its payload timestamp is the time
that context changed; the MCAP receipt timestamp can be later. Always join using
session/revision, not bag order alone. Successful normal controller selections
publish immediately. Binding/mode changes are additionally detected at each
metric publication. Configuration service changes publish immediately; other
in-place configuration mutations are checked at most once per second.

Kinematics means the measured/estimated vehicle and joint motion and desired
motion used as metric inputs. Performance metrics are the errors and accumulated
statistics computed from those inputs. The recording includes both, so an
analyst can inspect the source values rather than trusting only a score.

Vehicle pose inputs use map NED, with XYZ Euler angles as a storage convention;
body twists use FRD. Tool pose inputs use map NWU. Attitude *errors* remain SO(3)
rotation distances, not Euler differences. Joint order is e,d,c,b. Gripper is
excluded from the four-joint tracking metric, but its published command is
included in the five-effort command vector. Nonfinite values become JSON null,
with the metric validity and coverage flags retained. Published effort does not
establish delivered torque or electrical energy.

Existing recordings cannot gain this missing metadata retroactively. The
original November 16 13:42 recording still requires external controller-switch
logs for attribution.
