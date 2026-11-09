# uvms_simlab

A field-ready ROS 2 lab for **Underwater Vehicle–Manipulator Systems**. `uvms_simlab` layers interactive teleoperation, collision-aware planning, and hardware-in-the-loop tooling on top of [uvms-simulator](https://github.com/edxmorgan/uvms-simulator) so you can go from concept to wet tests without rebuilding infrastructure.


## Kinematics

`simlab/motion_planning/kinematics/` contains forward and inverse modules:

- `forward/__init__.py`: shared loader for the exported FK model.
- `inverse/__init__.py`: IK registry and solver selection.
- `inverse/base.py`: request/result types, interfaces and shared validation.
- `inverse/whole_body.py`: adapter for the exported Floating-KinDyn-Graph endpoint solver.
- `inverse/arm.py`: arm-only position solver.

`simlab.motion_planning.kinematics.inverse` provides the IK registry. `Robot` selects its
implementations through `whole_body_ik_name` (default `whole_body_endpoint`)
and `manipulator_ik_name` (default `manipulator_position`). Each robot owns its
solver instances. These are Python construction options, not ROS launch arguments
or new menu entries; each capability currently has one implementation.

Implement `WholeBodyIKTemplate` or `ManipulatorIKTemplate` in `kinematics/inverse/base.py` and
register the class in `DEFAULT_IK_CLASSES` to add a solver. Selection uses the
`IKCapability` enum, rejects mismatches and duplicate names. The old `task=`
selector remains a compatibility adapter; production uses explicit capabilities.

Both implementations expose `solve_request(request) -> IKResult`:

- `ManipulatorIKRequest`: arm-base target XYZ. Vehicle remains fixed; orientation
  is unconstrained. The existing analytic function is evaluated unchanged.
- `WholeBodyIKRequest`: world NWU target position and SO(3) rotation, current vehicle
  XYZ/extrinsic XYZ Euler angles, arm joints, mounting/tip transforms, bounds and
  tolerances. Frame conversion and marker-axis selection remain outside the solver.
- `IKResult`: status, joint solution, optional vehicle pose, message, per-call
  diagnostics and explicit validation guarantees. Failures contain no solution.
  Runtime is measured with a monotonic clock around solving and result validation;
  it is not end-to-end ROS latency. Legacy `solve` methods remain available for
  existing callers, but Robot uses the typed request path.

Whole-body success checks finiteness, joint limits and FK endpoint tolerances.
Robot independently checks the result with its own FK model through the same
validator before converting/applying commands. Command and marker updates happen
only after frame conversion succeeds. A failed arm solve is caught by the backend
and leaves its previous target intact.

Arm-only success guarantees finite output, **not** endpoint accuracy, joint-limit
validation or orientation tracking. The existing workspace precheck remains; the
analytic mapping can have a nonzero endpoint residual. This refactor does not
silently upgrade that guarantee or change the analytic branch/formulation. A future
bounded arm solver can be added without changing callers, but needs its own
numerical validation and deployment tests.

The adapter delegates numerical solving to the source runtime; model generation
remains in the source notebook. An IK endpoint is not a collision-checked trajectory.

## Disturbance-estimation demo

For an unlocked-arm disturbance experiment, stop the old simulator and launch:

```bash
source /home/mr-robot/ros_ws/install/setup.bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
  use_manipulator_hardware:=false use_vehicle_hardware:=false \
  sim_robot_count:=1 task:=interactive robot_name:=dory \
  launch_voxelviz:=true arm_joint_locks:=false
```

This disables only artificial arm locks. Gravity, friction, actuator limiting,
and joint-position limits remain enabled. The plant and EKF share the resulting
zero lock mask. The default is `arm_joint_locks:=true`; physical arm hardware is
unaffected. This startup choice persists through simulation resets and plant
profile changes. Startup logs report `Artificial arm joint locks: disabled`.
With live telemetry, all four `estimator_joint_locked.*` fields should be zero.
Regenerate the PlotJuggler layout after launching, as described below.
Removing locks permits estimation from motion; it does not guarantee small
uncertainty at hard joint stops or under other unobservable conditions.

Injected disturbances now use the dedicated per-robot interface described under
**Simulation disturbance injection** below. They no longer travel through the
interactive controller or `/contact_wrench_body`. No physical contact-response
model is added. Injected loads are plant-only; the native EKF receives actuator
effort, not the injection command. Locks/limits can prevent observable motion.

For PlotJuggler, subscribe to `/dynamic_joint_states`,
`/robot_1_/disturbance/vehicle/state`, and `/robot_1_/disturbance/arm/state`.
The state messages expose `applied_body_wrench` and `applied_joint_torques`
(e,d,c,b), sampled at completed plant steps. The old contact/arm-array bindings
must be updated; old layouts are not automatically migrated. The built-in
`disturbance_plot` observer already uses these new fields and hides invalid or
unobservable estimates. Applied injection is not total disturbance truth when
plant/model mismatch or physical contact is also present.

The arm EKF now predicts the known lock dynamics. While a joint is locked,
its disturbance mean is held and its uncertainty grows: a load balanced by a
rigid lock cannot be recovered from joint motion alone. Lock status is available
as `estimator_joint_locked.e/d/c/b` in the arm GPIO group. DR control rejects
the arm disturbance sample while any joint is locked.

The EKF also uses the existing Xacro position limits and predicts the plant's
end-of-step clamp. The completed plant contact state selects the EKF mode;
release restores free-motion sensitivities even if the prior predicts a stop.
During reported hard-stop contact, clamped coordinates are excluded from
torque evidence; free-coordinate innovations still update the correlated
posterior. Unresolved torque combinations retain uncertainty rather than
freezing every joint or artificially capping variance. `estimator_joint_at_limit.e/d/c/b` reports
completed-step contact and `estimator_disturbance_observable.e/d/c/b` marks
full-vector controller eligibility. DR rejects these contact samples.
PlotJuggler keeps valid arm posterior means/bands visible and shows an arm
status plot: 0 = constrained, 1 = free. At status 0, individual torques may
not be uniquely determined; a visible mean is not proof of observability.
Full-coordinate updates resume after leaving the stop. No extra joint-limit configuration is
introduced. A valid injection comparison requires an active controller holding
the arm away from its stops, not simply disabled artificial locks.

The default arm plant and fixed nominal observer both use gravity `[0,0,9.81]`.
Later plant-only profile changes do not update the observer. The blue line is
**not total disturbance truth** when there is model mismatch or other unmodeled
physics. Use matched parameters and unlocked joints for injection-only
validation; do not disable locks on a running system just to improve the plot.
Injected loads are never fed to either EKF. The generator refuses old/missing arm interfaces and leaves the
previous layout untouched; restart the updated simulator rather than loading
stale numerical array bindings.

The injection receivers expire commands locally after one steady-clock second
without renewal. A simulator reset clears the session and aborts the run.
Hardware drivers do not implement these injection receivers.

## Highlights

- **Direct RViz manipulation** – interactive markers drive the vehicle and arm-base targets without custom plugins.
- **Vehicle waypoint missions** – save multiple vehicle waypoints from RViz and execute them sequentially.
- **Dynamic obstacle source plugins** – A selected whole-scene behavior source drives obstacles; RViz authoring supplies geometry, size, pose, and velocity independently of vehicle motion.
- **Collision + clearance monitoring** – FCL-backed checks visualize contacts, environment bounds, and clearance markers.
- **SE(3) planning with live visualization** – OMPL planners + Ruckig execution stream candidate paths and waypoints to RViz.
- **Control modes** – PS4 teleop, joint-space torque control, or direct thruster PWM via launch args.
- **Visualization tooling** – workspace clouds, vehicle-base clouds, backend-published overlays, and opt-in voxel/collision debug markers.
- **Data logging** – rosbag2 MCAP recorder for repeatable datasets.
- **Perception extras** – optional RGB-to-pointcloud (MiDaS) for quick depth-based clouds.

### Obstacle behavior runtime and authoring

`simlab/dynamic_obstacle_sources/behavior.py` defines a whole-scene source lifecycle:
`initialize`, `step`, `on_edit`, `reset`, and `close`. A selected script can drive
independent obstacles or a coupled multi-agent/neural policy; it receives detached
scene and observation snapshots, elapsed experiment time, timestep, and seed.
Controllers and planners are not dependencies of this interface.

`ScriptedMotionSource` keeps obstacles stationary unless the caller specifies a
twist. Linear and angular velocities are world-frame values. The source preserves
caller-supplied IDs, geometry, dimensions, mesh scale, and initial placement.
Sphere, box, cylinder, and rigid mesh geometry use the same contract.

`simlab.obstacle_runtime.ObstacleRuntime` hosts exactly one source and offers
revision-checked atomic `edit(upsert=..., remove=...)`, detached snapshots, stepping,
reset, and shutdown. Revisions advance on both edits and motion; callers must
refresh and retry stale edits. A source step may update poses/twists, not silently
resize geometry or add/remove IDs. A failed source holds the last valid scene and
requires reset because its internal policy state may already have advanced.
Reset restores the scene supplied at construction and calls the source reset hook
with the original seed. Stateful sources must implement deterministic reset.

The launch starts `simlab/dynamic_obstacle_sim_node` with
`dynamic_obstacle_source:=scripted_motion`, `obstacle_source_config:='{}'`, and
`obstacle_seed:=0`. The old simulator motion integrator is removed. Existing
`/dynamic_obstacles` and `/dynamic_obstacle_markers` topics remain the inputs for
collision checking, visualization, cameras, and recording. `/obstacle_scene` adds
revision, selected source, and source-fault information.

In RViz, use the independent cyan **New obstacle preview** marker, initially at
`[2, 0, -1]` in the world frame. Drag the preview surface to move it in the
camera plane, or use its axes/rings for constrained translation/rotation.
Handles grow with the displayed geometry so large obstacles do not bury them. Right-click
to open **Obstacle settings (preview)** for shape, size, and initial velocity,
then choose **Add obstacle**.
No vehicle path or automatic replanning toggle is required. Preview geometry is
not an obstacle until the service accepts it. Use **Select obstacle** to load an
existing obstacle, edit its preview, then **Update selected obstacle**. Add is
shown only in new-preview mode; Update, Discard edits, and Delete appear only
with an existing selection. **New obstacle (preview)** clears the selection but
keeps the current preview geometry and placement for creating another object.
Checkmarks show the selected obstacle and matching shape/size/velocity presets;
custom values do not falsely select a preset. Delete, Clear, and Reset require
their confirmation submenu. Size presets mean sphere radius,
box side length, cylinder radius (height twice radius), or maximum mesh scale
component. Mesh resizing preserves nonuniform proportions and the authored
visual-to-collision scale ratio.
The typed service supports arbitrary dimensions and all linear/angular velocities.
For a mesh, set the interactive controller's `obstacle_mesh_resource` parameter
to a `package://`, `file://`, or absolute path before selecting Mesh. Meshes are
rigid; articulated/skinned animation is not supported.

The default **Obstacle settings (preview) → Shape → Mesh catalog** includes
38 bundled marine animals, grouped into Marine mammals, Sharks, Fish, Rays,
Turtles, and Other marine animals. Whales include blue, humpback, sperm, pilot,
and orca; fish include tuna, marlin, swordfish, barracuda, mahi-mahi, and more.
Select an animal, position the preview, then Add or Update. No downloads or
extra launch arguments are needed. The labels show the preset maximum dimension:
1 m for fish and 2 m for the other groups. These are experiment-sized defaults,
not biological scale. The Size control changes the mesh scale multiplier.

The CC0 assets are from [3DAssets.dev](https://3dassets.dev/packs/open-ocean-and-deep-sea-life).
They are stylized rigid models, not animated or scientifically validated anatomy.
Visual COLLADA files retain the original separate colour materials; collision STL
files have identical transformed triangles. The source pack has no image textures.
RViz uses embedded materials without the orange obstacle tint; the camera also
retains the material colours (its lighting is not an exact glTF PBR reproduction).
Assets use ROS Z-up and +X heading. The breaching whale, floating otter and resting
turtle retain their deliberately authored poses. Selecting another mesh preserves
the preview pose; **Orientation → Reset to asset axes (+X forward, Z up)** clears
previous marker rotations without changing position or velocity. Then Update.
Provenance, hashes,
licensing and coordinate conversions are in `resource/obstacle_meshes/`.
`tools/import_marine_assets.py` reproduces the asset conversion; ROS never runs it.

For a replacement custom catalog, set the interactive controller's
`obstacle_mesh_catalog` parameter to a local JSON file, then choose
**Mesh catalog → Reload catalog**. An empty parameter disables the catalog.
The file maps display names to mesh configurations; `Group / Name` creates
a submenu, for example:

```json
{
  "Whale": {
    "collision_mesh_resource": "file:///absolute/path/whale.stl",
    "collision_mesh_scale": [1.0, 1.0, 1.0]
  }
}
```

Custom meshes must exist locally. Catalog entries use the
same geometry validation as scene edits and may specify a separate
`visual_mesh_resource` and `visual_dimensions` (mesh scale). Selection changes
only the draft, retaining its placement and velocity; Add/Update commits it to the
selected behavior source. Invalid catalogs are reported, not silently substituted.

The obstacle marker is the single obstacle/source menu. It also owns
**Scene / source** (status, clear, reset, world profiles) and **Replanning**
(enable, disable, status). Robot target menus contain no obstacle controls.
These menu actions call the existing services; behavior sources remain separate
from planners and controllers. Source selection remains a launch setting.

`/dynamic_obstacle_sim_node/edit_dynamic_obstacles` (`simlab/srv/EditDynamicObstacles`)
supports `get`, `add`, `update`, `remove`, `clear`, `replace`, and `reset`.
`add` rejects duplicate IDs; `update` replaces only named existing obstacles;
other obstacles retain their current state. Set `check_revision=true` and
`expected_revision` for conditional edits. Without it, an explicit edit applies
to the current scene (the RViz draft intentionally uses this mode).
Responses report actual completion and include the authoritative scene.

All obstacle edits, including full-scene world-profile replacement, use the typed
edit service. There is no path-based creation source or compatibility service.
Position and size come from the frontend; the selected source controls behavior.
Replanning policy is independent of the obstacle behavior source.

### Dynamic navigation safety and recovery

The OMPL search, simplification, dense interpolation and resampling use the
remote baseline behavior. The isolated approximate-solution check rejects paths
that do not reach the requested goal.

Ruckig uses the original incremental update loop, including the original moving
handoff velocity projection. Requests are sent immediately using the original
preemption behavior; there is no added measured-stop queue or upfront trajectory
validation gate. Planner success is followed by Ruckig calculation on the next
control update, as in the baseline.

The obstacle editor, behavior sources, mesh collision geometry and marker
transport fixes remain independent of this restored execution pipeline.
When dynamic replanning is enabled, its supervisor reads timed samples from
Ruckig's active output without advancing it. Before the first control update,
no calculated preview exists; current-clearance/braking monitoring still runs.
Automatic replans include the configured clearance margin. This monitoring is
not a guarantee that every smoothed trajectory is collision-free.

Urgent clearance/braking checks precede planner-busy, cooldown, and hysteresis
suppression. Stop decisions use measured speed, tracking error, the replan period,
a latency allowance (`dynamic_replanning_latency_budget`, default 1.5 s), and an
estimated deceleration (`dynamic_braking_deceleration`, default 0.1 m/s², limited
by the trajectory acceleration setting). These estimates require experimental
calibration; they are not a certified physical stopping guarantee.

Safety holds preserve the active goal and waypoint index. In simulation only,
the supervisor retries after the vehicle settles below 0.02 m/s and the current
position meets the clearance margin. Failed attempts back off for at least two
seconds. Explicit stop/new goals invalidate pending recovery. Hardware never
auto-resumes. A vehicle inside the margin remains blocked until clearance is
restored; automatic retreat from overlap or an inflated shell is not implemented.

Validation samples at 50 ms and uses the existing obstacle prediction model;
it is not continuous collision detection or a guarantee for arbitrary nonlinear
source behavior. The live monitor rechecks timed lookahead. End-to-end testing
under representative dynamics and planner/trajectory latency remains necessary.

## Requirements

- ROS 2 jazzy plus the [uvms-simulator](https://github.com/edxmorgan/uvms-simulator) stack installed exactly as documented in its README (system packages, `vcs import`, `rosdep`, CasADi, etc.).
- ROS packages: `ros-$ROS_DISTRO-interactive-markers`, `ros-$ROS_DISTRO-cv-bridge`.
- Python deps: `pyPS4Controller`, `pynput`, `scipy`, `casadi`, `ruckig`, `python-fcl`, `trimesh`, `pycollada`.
- OMPL with Python bindings (`install-ompl-ubuntu.sh --python` from Kavraki Lab works well).
- Optional perception extras: `torch`, `torchvision`, `timm`, `opencv-python` (MiDaS RGB-to-pointcloud).
- Optional hardware: BlueROV2 Heavy + Reach Alpha 5 + Blue Robotics A50 DVL (or any robot stack you map through the provided interfaces).

## Quick start

1. **Install uvms-simulator and dependencies**  
   Follow the [uvms-simulator installation guide](https://github.com/edxmorgan/uvms-simulator/blob/main/README.md). 

2. **Install simlab extras**
   When this repo is pulled into the workspace with `vcs import`, install the extras and rebuild.

   ```bash
   cd ~/ros_ws
   sudo apt install ros-$ROS_DISTRO-interactive-markers ros-$ROS_DISTRO-cv-bridge

   sudo pip install pyPS4Controller pynput scipy casadi==3.7.2 ruckig python-fcl trimesh pycollada
   # Optional: RGB-to-pointcloud (MiDaS)
   pip install torch torchvision timm opencv-python

   wget https://ompl.kavrakilab.org/install-ompl-ubuntu.sh
   chmod u+x install-ompl-ubuntu.sh
   ./install-ompl-ubuntu.sh --python

   colcon build
   source install/setup.bash
   ```

## Launch recipes

**Interactive planner & RViz**

```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    sim_robot_count:=1 task:=interactive \
    use_manipulator_hardware:=false use_vehicle_hardware:=false
```

**PS4 joystick teleop**

```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    task:=manual
```

**Joint-space control**

```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    task:=joint
```

**Direct thruster PWM (keyboard)**

```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    task:=direct_thrusters
```

**Headless data collection**

```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    gui:=false task:=manual record_data:=true
```

> 💡 Recording: `record_data:=true` starts rosbag2 MCAP logging under `~/ros_ws/recordings/mcap/uvms_bag_YYYYmmdd_HHMMSS`.

> 💡 Hardware swap: set `use_vehicle_hardware:=true` and `use_manipulator_hardware:=true` to put your BlueROV2 Heavy, Reach Alpha 5, and A50 DVL directly into the loop.

## Task modes

| task | Simlab node | What it does | Input |
| --- | --- | --- | --- |
| `interactive` | `interactive_controller` | RViz markers + planner execution | RViz mouse/menus |
| `manual` | `joystick_controller` | PS4 teleop with PID control | PS4 controller |
| `joint` | `joint_controller` | Skeleton node for custom joint-space torque commands | Your node/scripts |
| `direct_thrusters` | `direct_thruster_controller` | Direct PWM commands | Keyboard |

## Interactive workflow

In `task:=interactive`, the vehicle marker menu exposes the main planning workflow:

- `Plan & Execute`
- `Add Vehicle Waypoint`
- `Delete Vehicle Waypoint >`
- `Clear Vehicle Waypoints`
- `Stop Vehicle Waypoints`
- `Reset Simulation`
- `Release Simulation`

### Single-goal planning

Move the vehicle marker to the desired pose and select `Plan & Execute`.

### Vehicle waypoint missions

For multi-point vehicle motion:

1. Move the vehicle marker to the first target.
2. Select `Add Vehicle Waypoint`.
3. Repeat for each additional target.
4. Select `Plan & Execute`.

The robot plans and executes the waypoint list in order.

Notes:

- `Delete Vehicle Waypoint` is a dynamic submenu built from the currently saved waypoints for the selected robot.
- `Clear Vehicle Waypoints` clears the saved waypoint queue for the selected robot.
- `Reset Simulation` also clears the selected robot waypoint queue and its waypoint visualization.
- Waypoint completion currently uses:
  - position tolerance
  - `yaw_blend_factor >= yaw_finish_threshold`

### Overlay information

The SimLab backend publishes RViz overlay data independently of the optional
voxel and collision debug nodes:

- `chatter`: research-use session text consumed by `string_to_overlay_text`,
  which publishes `/chatter_overlay_text` for RViz.
- `/robot_metrics_overlay_text`: live robot metrics overlay.

The robot metrics overlay includes, per robot:

- selected controller
- hold/release state
- vehicle linear speed
- manipulator gravity
- manipulator payload mass
- waypoint mission summary such as:
  - `WP none`
  - `WP queued N`
  - `WP 2/5 TRACKING`

Simulator dynamics can be changed online through the combined service provided by `uvms-simulator`:

```bash
ros2 service call /robot_1_set_sim_uvms_dynamics ros2_control_blue_reach_5/srv/SetSimDynamics \
  "{use_coupled_dynamics: false, set_vehicle_dynamics: false, set_manipulator_dynamics: true, manipulator: {gravity_vector: [0.0, 0.0, 9.81], payload_mass: 0.15, payload_inertia: [0.0, 0.0, 0.0]}}"
```

In RViz interactive mode, use `Dynamics Profile` to apply an installed whole-robot dynamics profile during live simulation.

## Project layout

```
simlab/
├── simlab/uvms_backend.py            # Core backend, FCL world, planners, TFs, waypoint missions
├── simlab/interactive_control.py     # RViz markers + menus
├── simlab/vehicle_waypoint_mission.py# Vehicle waypoint queue state + RViz waypoint markers
├── simlab/controllers/               # One controller class per file
├── simlab/utils/                     # Shared geometry, frame, mesh, marker, and path-obstacle helpers
├── simlab/uvms_parameters.py         # Shared manipulator and vehicle controller parameters
├── simlab/motion_planning/planners/  # Planner plugins, including OMPL SE(3) planning
├── simlab/motion_planning/trajectory_generators/ # Vehicle trajectory generator plugins
├── simlab/motion_planning/dynamic_replanners/ # Dynamic replanning supervisor plugins
├── simlab/dynamic_obstacle_sources/  # Whole-scene behavior lifecycle plugins
├── simlab/joystick_control.py        # PS4 teleop node
├── simlab/joint_control.py           # Joint-space torque control
├── simlab/direct_thruster_control.py # Thruster PWM keyboard control
├── simlab/collision_contact.py       # Opt-in FCL contact markers + clearance
├── simlab/voxel_viz.py               # Opt-in bathymetry voxel clouds
├── simlab/bag_recorder.py            # rosbag2 MCAP recorder
└── resource/model_functions/         # Generated model functions
```

## Motion planning plugins

Motion planning lives under `simlab/motion_planning/`. The framework supports both split pipelines and integrated algorithms:

```text
OMPL path planner -> incremental multi-waypoint Ruckig trajectory -> controller
CHOMP/GPMP optimizer -> path or timed trajectory -> controller
MPC/integrated planner-controller -> direct references or controls
```

All new motion-planning algorithms should return `MotionPlanResult` from `simlab.motion_planning.result`. The result declares what the algorithm produced:

| Result kind | Meaning | Current runtime behavior |
| --- | --- | --- |
| `MotionPlanKind.PATH` | Geometric waypoints with `xyz` and `quat_wxyz` | Executed through the current `PlanVehicle` action, then time-parameterized by the selected trajectory generator such as Ruckig |
| `MotionPlanKind.TIMED_TRAJECTORY` | Timed trajectory samples, optionally with velocity/acceleration | Supported by the Python plugin contract; needs a richer execution transport before it can bypass Ruckig at runtime |
| `MotionPlanKind.CONTROL_SEQUENCE` | Direct controls or short-horizon references | Supported by the Python plugin contract; intended for future MPC/integrated execution paths |

A simple path planner plugin looks like this:

```python
import numpy as np

from simlab.motion_planning.planners.base import PlannerTemplate
from simlab.motion_planning.result import MotionPlanKind, MotionPlanResult


class MyPlanner(PlannerTemplate):
    registry_name = "MyPlanner"
    visible = True

    def plan_vehicle(
        self,
        *,
        start_xyz,
        start_quat_wxyz,
        goal_xyz,
        goal_quat_wxyz,
        time_limit,
        robot_collision_radius,
        dynamic_obstacle_prediction_speed=0.0,
    ):
        xyz = np.asarray([start_xyz, goal_xyz], dtype=float)
        quat = np.asarray([start_quat_wxyz, goal_quat_wxyz], dtype=float)
        length = float(np.linalg.norm(xyz[-1] - xyz[0]))
        return MotionPlanResult(
            is_success=True,
            kind=MotionPlanKind.PATH,
            xyz=xyz,
            quat_wxyz=quat,
            path_length_cost=length,
            geom_length=length,
            message="MyPlanner returned a straight-line path.",
        )
```

Register planner classes in `simlab/motion_planning/planners/__init__.py` by adding them to `DEFAULT_PLANNER_CLASSES`. The RViz planner menu and planner action server both read that registry.

Trajectory generators and dynamic replanning supervisors are separate plugin registries under `simlab/motion_planning/trajectory_generators/` and `simlab/motion_planning/dynamic_replanners/`. Use them for split pipelines. `MotionPlanResult` can represent paths, timed trajectories, and control sequences in Python, but the current `PlanVehicle` ROS action transports geometric paths only. The action server explicitly rejects other result kinds; integrated planners require richer transport and execution support before they can be used through this action.

## Adding a controller

Controllers live in `simlab/controllers/`. Each controller gets its own file and inherits `ControllerTemplate`.

1. Create a controller file, for example `simlab/controllers/my_controller.py`.

   ```python
   import numpy as np

   from simlab.controllers.base import ControllerTemplate


   class MyController(ControllerTemplate):
       registry_name = "MyController"

       def __init__(self, node, arm_dof=4):
           super().__init__(node, arm_dof)
           self.arm_kp = np.ones(self.arm_dof + 1, dtype=float)
           self.arm_u_max = np.ones(self.arm_dof + 1, dtype=float)
           self.arm_u_min = -self.arm_u_max

       def vehicle_controller(self, state, target_pos, target_vel, target_acc, dt) -> np.ndarray:
           state = self.vector(state, 12, "state")
           target_pos = self.vector(target_pos, 6, "target_pos")
           return np.zeros(6, dtype=float)

       def arm_controller(
           self,
           q,
           q_dot,
           q_ref,
           dq_ref,
           ddq_ref,
           dt,
       ) -> np.ndarray:
           q = self.arm_vector(q, "q")
           return np.zeros(self.arm_dof + 1, dtype=float)
   ```

2. Register the class in `simlab/controllers/__init__.py`.

   ```python
   from simlab.controllers.my_controller import MyController

   DEFAULT_CONTROLLER_CLASSES = [
       LowLevelPidController,
       LowLevelInvDynController,
       MyController,
   ]
   ```

   `Robot` reads `DEFAULT_CONTROLLER_CLASSES` and registers every class in that list. You do not need to edit `simlab/robot.py` for a normal new controller.

3. Rebuild and source the workspace.

   ```bash
   colcon build --packages-select simlab
   source install/setup.bash
   ```

The controller will appear in the RViz interactive controller menu using `registry_name`. Keep controller-specific gains, limits, and model parameters inside the controller class. `Robot` only passes state, references, and `dt` into the standard `vehicle_controller()` and `arm_controller()` methods.

## Planning and endpoint IK

The experimental collision-checked whole-body RRT, 30 cm task clearance and
collision-monitor worker were removed. Task mode now uses the bounded SE(3)
endpoint IK exported by Floating-KinDyn-Graph, with the source runtime policy
and independent SimLab command acceptance described above. It is not the old
remote-main IK and does not certify a collision-free path. Joint-space vehicle
planning retains its existing planner and trajectory-generator interfaces.
Restart running nodes after rebuilding changed runtime modules or models.

## Contributing

Contributions are welcome.

## Simulation disturbance injection

The standard simulation launch starts one idle `disturbance_injection_node`
per robot, independent of the selected controller/task. Hardware launches do
not start it. Restart an already-running simulator after installing this update.
For a custom simulation launch, start it manually:

```bash
ros2 run simlab disturbance_injection_node --ros-args -p robot_prefix:=robot_1_
```

Start a finite run with the action client (which also starts an independent
read-only live plot):

```bash
ros2 run simlab disturbance_run --dof surge yaw --amplitude 3 0.1 \
  --waveform sine triangle --frequency 0.1 --cycles 3 \
  --phase 0 -90 --delay 0 2
```

Use `--robot-prefix robot_2_` for another simulated robot. Supported DOFs are
surge/sway/heave (N), roll/pitch/yaw and arm_e/arm_d/arm_c/arm_b (Nm).
Amplitudes require one value per DOF. Waveform, phase (degrees) and delay
(seconds) accept one value for all axes or one per axis. Positive phase leads;
negative phase lags. Shapes: sine, step, square, triangle, chirp, gaussian. Step supports
delay only. Chirp uses `--end-frequency`; cycles/frequency specifies duration,
not the number of chirp oscillations. Commands are sampled at 20 Hz; sharp
transitions are not smoothed. Frequency is limited to below 5 Hz.

Gaussian noise is zero-mean with `--amplitude` specifying **standard deviation**
(N or Nm), not a peak bound. Samples are independent per DOF and held on a
20 Hz simulation-time grid. `--seed` (uint32, default 0) selects a repeatable
sequence; phase must be zero, while delay is supported. For Gaussian channels,
`cycles / frequency` specifies active duration, not a noise oscillation frequency.
For example, 30 seconds of surge noise with standard deviation 3 N:

```bash
ros2 run simlab disturbance_run --robot-prefix robot_1_ --dof surge \
  --waveform gaussian --amplitude 3 --frequency 0.1 --cycles 3 --seed 42
```

Gaussian and periodic waveforms can be combined across DOFs. Samples use NumPy
PCG64 with `SeedSequence([seed, physical_dof_index, sample_index])`, where
`sample_index = floor(20 * channel_elapsed_seconds)`. Pausing repeats the same
sample; skipped time bins are skipped, not replayed in a burst. This is sampled
noise, not continuous white noise. Callback and plant-step scheduling can affect
which samples are actually applied; use the standard recorded applied-state
messages for the exact experiment. Seed and generator settings travel in the
existing status topic; no extra files or recorder are created.

A run has a 2-second baseline, each channel's delay and full active duration,
and 3 seconds of zeros after the latest channel finishes. Time follows the
vehicle's accumulated completed dynamics-step durations, freezing on pause.
The independent steady-clock receiver watchdog still runs while paused;
the generator renews its lease without advancing the waveform. An inconsistent
vehicle/arm pause state aborts rather than advancing only one component.

Ctrl+C in `disturbance_run` requests cancellation. Closing the plot stops only
the observer. Stop the active generator session independently with:

```bash
ros2 service call /robot_1_/disturbance/stop std_srvs/srv/Trigger '{}'
```

ROS interfaces (replace robot prefix as needed):

- `/robot_1_/disturbance/run`: `simlab/action/RunDisturbance` action.
- `/robot_1_/disturbance/stop`: idempotent `std_srvs/srv/Trigger` stop request.
- `/robot_1_/disturbance/status`: `simlab/msg/DisturbanceRunStatus` with exact configuration.
- `/robot_1_/disturbance/command`: simulation-only `DisturbanceCommand` transport.
- `/robot_1_/disturbance/{vehicle,arm}/state`: `DisturbanceState` receiver heartbeat,
  lease owner, paused state, generation, completed-step time/ID and applied load.

The action rejects overlapping runs and unavailable/owned receivers. Low-level
transport requires a zero-valued acquisition with a unique session ID, exact
body frame and joint names, and increasing sequence numbers. Each receiver
rejects other owners and expires its lease after 1 second. This is operational
ownership, not authentication against hostile publishers. The generator acquires
both components with zeros before sending loads. Vehicle and arm apply commands
at their next respective steps; atomic whole-body application is not claimed.
Use both recorded step clocks when analyzing synchronization.

Vehicle dynamics sum actuator, injected and contact loads; stopping/expiring
injection changes only the injected term. `contact_body_wrench` is recorded
separately from `applied_body_wrench`. Arm injection is added after actuator
limiting and before constrained dynamics. Applied external torque does not mean
net torque or motion: a joint lock/stop may oppose it. Estimator inputs remain
actuator-only. No contact model, controller tuning or EKF tuning is changed.

Task-space injection applies a wrench **at the IK tool origin** and maps it
through the corrected exported geometric Jacobian transpose into vehicle and
arm loads. Example: a 1 N sine force along the moving tool's X axis:

```bash
ros2 run simlab disturbance_run --robot-prefix robot_1_ \
  --space task --wrench-frame tool --dof fx --amplitude 1 \
  --waveform sine --frequency 0.1 --cycles 3
```

Task DOFs are `fx fy fz tx ty tz` (N and Nm). All existing waveforms, seeds,
delays and cycles work. `--wrench-frame tool` follows the tool orientation;
`body` uses the vehicle URDF's FLU axes; `world` uses world NWU axes. **Every
input torque is about the tool origin**, even for body/world axes. The mapping
includes the force lever arm about the vehicle origin and rotates the resulting
vehicle loads into dynamics FRD axes. Default `--space generalized` preserves
the existing surge/sway/heave/roll/pitch/yaw and arm joint-torque interface.

The node uses `tool_geometric_jacobian.casadi`, exported from the corrected
Floating-KinDyn-Graph `geo_J_full` by the final section of
`usage/robot_inverse_kinematics_analysis.ipynb`. It uses the same four-joint
Alpha model, mounting and tip offsets as simulated IK. Joint positions update
the mapping every injection tick. Missing/stale positions or unavailable
world/body TF reject a new run or abort/release an active run. World-frame
mapping selects a fresh joint state overlapping TF history and queries that
same timestamp; the mapping does not extrapolate stale poses. Configuration
samples can still precede the two plants' independently completed steps.

This adds an external wrench; it does not activate motors, unlock joints,
create contact, or add the vehicle-arm inertial coupling omitted by the plant.
Locked-joint loads are opposed by the existing constraints. It is a sampled
20 Hz wrench mapping, not a new coupled dynamics model.

Task-mode live plots show six **equivalent tool-wrench estimates**, always in
tool axes about the tool origin, against the requested tool wrench. The read-only
observer is hosted by the existing disturbance node and publishes
`/<robot_prefix>/disturbance/task_estimate` (`TaskWrenchEstimate`) at 20 Hz,
also through the standard MCAP recorder. It uses only measured joint positions
and EKF generalized disturbance estimates from the same joint-state message;
injection requests never enter estimation. Locked, at-limit, invalid and missing
arm estimates are excluded. A covariance-weighted SVD must identify all six
wrench directions; otherwise the estimate and covariance are NaN and invalid.
Stale telemetry also yields invalid output rather than held estimates.

The covariance and ±2σ bands are approximate: published marginal variances form
a diagonal input covariance, with no cross-covariance or kinematic uncertainty.
The output includes rank, number of observations and the unexplained generalized
residual (vehicle force in N; vehicle/arm torques in Nm). With exactly six usable
observations there is no redundant consistency check. A valid solve does **not**
prove tool contact: distributed drag/model error can appear as an equivalent
tool wrench. The requested line is not a measured applied contact force. The standard MCAP recorder also includes
`/<robot_prefix>/disturbance/task_wrench` (`WrenchStamped`): the requested wrench
expressed at/in `<robot_prefix>joint_4`, stamped with the configuration sample
used for mapping (release zeros use the current timestamp). The original input
space/frame and waveform settings are in run status; the mapped command and
completed-step applied loads remain in the existing command/state topics.
Restart the simulation and viewer after rebuilding the action interface.

Time-correlated Gaussian loads are available as `correlated_gaussian`:

```bash
ros2 run simlab disturbance_run --robot-prefix robot_1_ \
  --dof surge roll pitch --waveform correlated_gaussian \
  --amplitude 1 0.1 0.1 --correlation-time 2 \
  --frequency 0.1 --cycles 7 --seed 42
```

This is a stationary, zero-mean Ornstein–Uhlenbeck process sampled and held at
20 Hz. Amplitudes are standard deviations (N for force, Nm for torque), not
peak bounds. Its covariance is `sigma² exp(-|lag|/tau)`; `--correlation-time`
sets `tau` in seconds (positive, default 2). At a lag of `tau`, correlation is
about 0.368. Larger values produce slower changes. Channels have independent
seeded noise. `--frequency` and `--cycles` set the active duration, not the
noise bandwidth: the example runs for 70 seconds plus baseline/tail.
The process starts in its stationary distribution; baseline/tail remain zero,
so an onset/termination jump is possible. Phase is unsupported; delay works.
This is a correlated external-load test, not a calibrated ocean-current model.

The implementation evaluates exact Gaussian conditional samples on the 20 Hz
grid using seeded anchors and bridges, with a bounded cache. Repeating a seed
and channel reproduces values regardless of callback timing, pause, skipped
samples or cache eviction. The action carries `correlation_time_s`; accepted
status records this value and the waveform implementation metadata through the
standard MCAP pipeline. Rebuild and restart the simulation after updating the
RunDisturbance action interface; already-running clients/servers use the old
interface.

The live plot shades the EKF estimate ±2σ using the published disturbance
variance (σ is its square root, in N or Nm). This is the filter’s marginal
model-based uncertainty, not a calibrated coverage guarantee or the injected
noise standard deviation. Invalid/stale estimates, unobservable arm joints,
and unavailable/negative/nonfinite variances have no band; missing variance is
never treated as zero. The applied disturbance and estimate remain separate traces.

The viewer stays open with the completed traces. Repeating `disturbance_run`
for the same robot reuses that window, clears the old traces and updates the
layout for the selected DOFs and waveform settings. Closing it allows the next
run to open a new window. Different robots use separate viewers.

`--no-plot` skips opening or reconfiguring the viewer. `--close-on-finish` closes
the live plot when the run finishes. Neither the client nor the viewer writes CSV,
JSONL, configuration files or automatic plot images. There is no separate
output directory or `--output-dir` option.

All data uses the existing rosbag2 MCAP recorder. Start it with the normal UI
recording controls, `record_data:=true`, or the existing services:

```bash
ros2 service call /bag_recorder_node/start_recording std_srvs/srv/Trigger '{}'
# Run the disturbance and robot experiment, then:
ros2 service call /bag_recorder_node/stop_recording std_srvs/srv/Trigger '{}'
```

The recorder includes each configured robot's disturbance `command`,
`vehicle/state`, `arm/state` and `status` topics. Status contains the accepted
waveform configuration; applied-state messages contain dynamics-step times and
ownership information. The existing `dynamic_joint_states` topic contains EKF
estimates and validity/observability fields. These are stored together with the
normal camera, reference and environment data under the configured bag directory
(default `~/ros_ws/recordings/mcap`). Start recording before starting the run to
capture the full experiment. Injection does not automatically start/stop a bag.

This adds disturbance data to standard recordings; the MCAP-to-profile converter
and CmdReplay do not yet reapply disturbance tracks. Recording support must not
be mistaken for automatic disturbance replay.

To observe independently (without controlling or starting injection):

```bash
ros2 run simlab disturbance_plot --dof surge yaw --amplitude 3 0.1
```

When starting a viewer manually, supply the same waveform/timing options for
matching plot titles and axes; the accepted configuration is available on the status topic and in MCAP.
Legacy `/disturbance_injection/body`, `/disturbance_injection/wholebody` and
`/robot_1_external_arm_torques` injection bridges are removed. Use
`ros2 run simlab disturbance_run` for injection and the standard MCAP controls
for recording.

### User-provided disturbance waveforms

Waveforms live under `simlab/disturbances`, using named implementations like the
planner/controller registries. External Python packages register the entry-point
group `simlab.disturbances`; no changes to the injection node are needed.
For example, in your package's `pyproject.toml`:

```toml
[project.entry-points."simlab.disturbances"]
cosine = "my_disturbances:Cosine"
```

```python
import math
from simlab.disturbances.base import DisturbanceWaveform

class Cosine(DisturbanceWaveform):
    registry_name = 'cosine'

    @staticmethod
    def sample(c):
        return math.cos(2 * math.pi * c.frequency * c.elapsed
                        + math.radians(c.phase_deg))
```

Install the package into the Python environment used by both the simulator and
CLI, then restart them. Select `--waveform cosine` with the usual amplitude,
frequency, cycles, phase and delay options. Each DOF can select a different form.
The immutable context includes channel-local elapsed seconds, frequency Hz,
duration seconds, optional chirp end frequency, phase degrees, seed and physical
channel index. Return a finite dimensionless value; SimLab handles amplitude,
delay, zero outside the active interval, leases, stop/pause and MCAP topics.
Custom forms use these shared parameters; arbitrary per-plugin parameters are
not currently exposed. Plugin names and run settings are in the standard status
recording; preserve your plugin package/version alongside experiments.

Implementations must be deterministic for the same context, fast and nonblocking,
without ROS, file or hardware side effects. Plugins execute trusted Python in the
server process, not a sandbox. Duplicate names cannot override built-ins.
Exceptions/nonfinite output abort the run through the existing release/zero path.

For local files without a separately installed package, place your class in
`simlab/disturbances/my_waveform.py`, then edit only `simlab/disturbances/local.py`:

```python
from .my_waveform import Cosine
LOCAL_WAVEFORMS = (Cosine,)
```

Rebuild `simlab` and restart the simulator/client. Do not edit `_registry()` or
`BUILTINS`. Names must be unique across built-ins, local classes and installed
entry points; abstract or incorrectly named implementations are rejected.

Waveforms own `validate_channel`, `normalize_end_frequency`, `plot_title`,
`plot_style` and `metadata` hooks; the base class supplies normal defaults.
Use `uses_end_frequency=True` to opt into the shared end-frequency option.
Override `version` when changing an implementation. The accepted configuration
in the standard status topic records each implementation's qualified name,
version, source-file SHA-256 (null if unavailable), and metadata. This hashes the
implementation file, not its entire dependency tree; retain package/environment
versions when reproducing custom experiments. No additional recorder is created.

### Estimated-state feedback

Controllers and high-level planning share the state returned by `Robot.get_state()`.
Vehicle pose and body velocity use the `position_estimate`, `*_estimate` attitude,
`velocity_estimate` and `angular_velocity_estimate` interfaces. Body acceleration
uses the posterior model's `linear_acceleration_estimate` and
`angular_acceleration_estimate` interfaces. Arm/gripper feedback uses
`filtered_position`, `filtered_velocity` and `estimated_acceleration`; the arm's
filtered joint-state broadcaster feeds TF through
`/joint_states`. The original broadcaster keeps
`/dynamic_joint_states` unaliased for recording and exposes raw joints only at
`/joint_state_broadcaster/joint_states`. Auxiliary joints without filters
(e.g. thrusters) use a separate, disjoint TF feed on the same estimated-joint topic;
its diagnostic messages remain local. These are the vehicle
and arm EKF posteriors (the gripper retains its separate filter).

Task-wrench estimation reconstructs the tool wrench using filtered joints. External
wrench injection is a plant operation: it uses physical joints and, for world-frame
loads, physical vehicle attitude with the static world/map transform. It must not
use the estimated body TF to determine the physical load applied to the simulator.

Truth/raw interfaces remain available in the standard recording for evaluation;
they are not a fallback for missing estimator feedback. Tracking diagnostics now
measure reference-versus-estimated-state error, not simulator ground-truth error.
Use separately recorded physical state when evaluating actual tracking accuracy.
Rebuild both simulator and SimLab and restart after changing this interface contract.

### Vehicle attitude feedback

PID, InvDyn and OGES use a shortest SO(3) rotation-vector
error rather than subtracting Euler angles. Equivalent Euler representations
(including accumulated turns and the alternate roll/pitch/yaw chart) therefore
produce the same feedback. Position feedback and integral attitude error use
world NED axes; commanded wrenches remain body FRD. Reference velocities are
physical twists expressed in the reference body frame. InvDyn transports both
velocity and its body-component derivative into the current body frame,
including the derivative of the frame rotation.

This is shared controller behavior for task endpoints, waypoint following,
replay tracking/holding and controller settling. Time-varying references and
angular velocity still specify intentional turns; a static orientation endpoint
does not encode a winding count. Arm joints remain bounded real coordinates.
Configuration snapshots identify `attitude_error: so3_log_body`.

A shortest rotation is ambiguous at exactly 180 degrees. The implementation
chooses a deterministic branch; it does not claim globally smooth attitude
feedback or a global OGES stability certificate. Removing Euler-rate matrices
from these feedback laws does not replace the Euler-coordinate plant dynamics
or all trajectory conversion utilities. Those still have pitch-chart limits.

The PID/InvDyn source and reproducible export are in diff_uv:
`diffUV/controller.py`, `diffUV/utils/rotation_error.py`, and
the existing `usage/pid_controller.ipynb` and `usage/F_serialize.ipynb` workflow.
Run `F_serialize.ipynb` from the diff_uv `usage` directory, then copy
`uv_reg_pid_controller.casadi`, `uv_trackingController.casadi`, and
`rotation_log.casadi` into SimLab's `resource/model_functions/vehicle/`.

Rebuild SimLab and restart controllers to load changed CasADi resources and
clear integral memory. No physical-hardware validation is implied by simulator
or offline native-plant tests.

SimLab selects `OGES(n_dof=6, cyclic_dims=(3,4,5))` from namor for the vehicle.
The geometric CLF formulation lives in namor; the arm uses its unchanged default.

The `whole_body_endpoint` IK adapter loads `whole_body/endpoint_ik_ipopt.casadi`
and the shared forward-kinematics export. The optimization is generated by
Floating-KinDyn-Graph's `usage/robot_inverse_kinematics_analysis.ipynb`.
SimLab owns solver retries, candidate selection and independent final command
acceptance. No Floating-KinDyn-Graph Python runtime installation is required.

### Running the Python tests

After building, run from the ROS workspace with its installed environment loaded:

```bash
source install/setup.bash
python3 -m pytest src/uvms-simlab/test -q
```

The tests need the generated ROS messages and native simulator extension from
that overlay. Adding only source folders to `PYTHONPATH` is insufficient and
can cause missing-message errors during test collection.

### Integration rule for generated models

Keep kinematics, dynamics and estimator generation in the existing source
notebooks. SimLab and the simulator consume the exported `.casadi` functions
or compiled `.so` libraries through their existing adapters. Update the existing
export cells when fixing those models; do not add a parallel `export_*.py`
script, a source-directory import, or a new runtime Python package for them.
Retries, candidate selection, ROS integration and command validation stay in
the consuming adapter. Regenerate affected artifacts and verify their interfaces
and numerical behavior before deploying them.

### Exported model compatibility

The public `.casadi` model bundle is generated with **CasADi 3.7.2**. Use that
Python version for the documented runtime and public model exports. Newer local
CasADi development builds may write a serialization format that 3.7.2 cannot read;
a successful local load is not sufficient for release compatibility.

Regenerate through the existing source notebooks, preserving function signatures,
and validate with stock 3.7.2 before replacing the public artifacts. Vehicle
exports use `diff_uv/usage/code_generation.ipynb`; arm controllers, FK, regressors
and endpoint IK use their existing Floating-KinDyn-Graph notebooks. These source
repositories are needed for regeneration, not normal SimLab runtime. Native CUDA
code generation is a separate toolchain capability, not a requirement for loading
the public serialized models. Private `namor` controllers remain optional.

## Updating the coordinated 0.1 interfaces

Update both `uvms-simulator` and `uvms-simlab` together. SimLab 0.1 requires
`ros2_control_blue_reach_5 >= 0.1.0`; rebuild both packages and restart running
ROS processes after updating, since disturbance and replay interfaces changed.
Use Git LFS to obtain bundled model binaries (`git lfs pull` in each repository).
The public Python model runtime uses CasADi 3.7.2. Private `namor` controllers
and licensed identification tooling are optional and are not required for
ordinary simulation. The vehicle model now assumes full immersion; the rendered
water surface does not supply artificial support. Simulation sensor noise is
enabled by default and can be toggled through the existing menu.

### Sliding-mode controller (SMC)

The controller menu and `/backend/robot_command` accept `SMC`. PID remains the
default. SMC loads the source-generated vehicle and arm `.casadi` exports, with
no runtime import of diff_uv or Floating-KinDyn-Graph. Both joint-space and
whole-body/task-space reference paths use this same low-level controller.

The current gain profile is **simulation-only** (`uvms_parameters_sim.py`).
Vehicle gains use body-wrench units; arm switching gains use joint acceleration
units. Do not copy one subsystem's gains to the other. Positive boundary widths
reduce chattering. The four-joint arm includes gravity/friction compensation;
the gripper uses its separate mass/damping/stiffness. Outputs follow the normal
actuator limits, motor hysteresis and joint-stop handling. Locked joints cannot
track a reference until normal actuator logic releases them.

Switching between feedback controllers (PID, InvDyn and SMC) preserves the active
plan, waypoint mission and references, and resets the incoming controller’s internal
state. Re-selecting the same controller does not reset it. Different control laws
can still produce a change in commanded effort at the switch. Reference replay can select SMC through
`subsystem_mode.feedback_controller`. Model configuration uses the existing
controller configuration service and includes gripper parameters; nonzero-current
nominal models are rejected because this SMC export assumes zero current.

```
ros2 service call /backend/robot_command simlab/srv/BackendRobotCommand \
  "{robot_index: 0, command: set_controller, name: SMC}"
```

The exports' reaching-law checks are not guarantees under actuator saturation,
locks, stops, arbitrary disturbances or unvalidated hardware gains. Simulation
verification and its reproducible standard replay are recorded in the workspace
`diagnostics/sliding_mode_integration/` directory.

For reproducible simulation comparisons, `/backend/controller_configuration`
also accepts `operation: set_gains` with a JSON object in `gains_json`. PID and
InvDyn accept `kp`, `ki`, `kd`, `arm_kp`, `arm_ki`, and `arm_kd`; SMC accepts
`vehicle_` or `arm_` followed by `surface`, `damping`, `switching`, or `boundary`.
Arrays must match the controller's existing dimensions. Values must be finite
and nonnegative; SMC surface and boundary values must be strictly positive.
Updates are atomic, reset controller memory, and require an inactive or reset-held
simulation with replay stopped. Hardware, model parameters, and effort limits
cannot be changed through this operation. Use `operation: get` to record the
effective configuration. Rebuild service clients after updating the interface.
PID vehicle output now respects the same configured wrench limits as InvDyn and
SMC; its existing integral-state clamp remains in effect.

Simulation state resets wait for a `dynamic_joint_states` message stamped after
the reset service completes before releasing feedback or starting replay. This
prevents queued pre-reset observations from becoming the first replay sample
and controller input. Commands remain zero while waiting; a ten-second timeout
fails the reset and leaves commands held. This also applies between replay passes.
