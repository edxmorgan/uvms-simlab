# uvms_simlab

A field-ready ROS 2 lab for **Underwater Vehicle–Manipulator Systems**. `uvms_simlab` layers interactive teleoperation, collision-aware planning, and hardware-in-the-loop tooling on top of [uvms-simulator](https://github.com/edxmorgan/uvms-simulator) so you can go from concept to wet tests without rebuilding infrastructure.


## Highlights

- **Direct RViz manipulation** – interactive markers drive the vehicle and arm-base targets without custom plugins.
- **Vehicle waypoint missions** – save multiple vehicle waypoints from RViz and execute them sequentially.
- **Collision + clearance monitoring** – FCL-backed checks visualize contacts, environment bounds, and clearance markers.
- **SE(3) planning with live visualization** – OMPL planners + Ruckig execution stream candidate paths and waypoints to RViz.
- **Control modes** – PS4 teleop, joint-space torque control, or direct thruster PWM via launch args.
- **Visualization tooling** – workspace clouds, vehicle-base clouds, backend-published overlays, and opt-in voxel/collision debug markers.
- **Data logging** – rosbag2 MCAP recorder for repeatable datasets.
- **Perception extras** – optional RGB-to-pointcloud (MiDaS) for quick depth-based clouds.

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
   cd ~/ros2_ws
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
├── simlab/uvms_parameters.py         # Shared manipulator and vehicle controller parameters
├── simlab/se3_ompl_planner.py        # OMPL SE(3) planning
├── simlab/cartesian_ruckig.py        # Ruckig trajectory generation
├── simlab/joystick_control.py        # PS4 teleop node
├── simlab/joint_control.py           # Joint-space torque control
├── simlab/direct_thruster_control.py # Thruster PWM keyboard control
├── simlab/collision_contact.py       # Opt-in FCL contact markers + clearance
├── simlab/voxel_viz.py               # Opt-in bathymetry voxel clouds
├── simlab/bag_recorder.py            # rosbag2 MCAP recorder
└── resource/model_functions/         # Generated model functions
```

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
