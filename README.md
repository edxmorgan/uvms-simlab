# uvms_simlab

A field-ready ROS 2 lab for **Underwater Vehicle–Manipulator Systems**. `uvms_simlab` layers interactive teleoperation, collision-aware planning, and hardware-in-the-loop tooling on top of [uvms-simulator](https://github.com/edxmorgan/uvms-simulator) so you can go from concept to wet tests without rebuilding infrastructure.


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

   sudo pip install pyPS4Controller pynput scipy casadi ruckig python-fcl trimesh pycollada
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

## Contributing

Contributions are welcome.
