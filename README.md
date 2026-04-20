# uvms_simlab

A field-ready ROS 2 lab for **Underwater Vehicle–Manipulator Systems**. `uvms_simlab` layers interactive teleoperation, collision-aware planning, and hardware-in-the-loop tooling on top of [uvms-simulator](https://github.com/edxmorgan/uvms-simulator) so you can go from concept to wet tests without rebuilding infrastructure.


## Highlights

- **Direct RViz manipulation** – interactive markers drive the vehicle and arm-base targets without custom plugins.
- **Collision + clearance monitoring** – FCL-backed checks visualize contacts, environment bounds, and clearance markers.
- **SE(3) planning with live visualization** – OMPL planners + Ruckig execution stream candidate paths and waypoints to RViz.
- **Control modes** – PS4 teleop, joint-space torque control, or direct thruster PWM via launch args.
- **Mocap + viz tooling** – OptiTrack/mocap4r2 publishing, pose/path trails, workspace clouds, and voxelized bathymetry.
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
   This repo is usually pulled into the workspace via `vcs import`. Install the extras and rebuild.

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

> 💡 Recording: `record_data:=true` starts rosbag2 MCAP logging to `uvms_bag_YYYYmmdd_HHMMSS`.

> 💡 Hardware swap: set `use_vehicle_hardware:=true` and `use_manipulator_hardware:=true` to put your BlueROV2 Heavy, Reach Alpha 5, and A50 DVL directly into the loop.

## Task modes

| task | Simlab node | What it does | Input |
| --- | --- | --- | --- |
| `interactive` | `interactive_controller` | RViz markers + planner execution | RViz mouse/menus |
| `manual` | `joystick_controller` | PS4 teleop with PID control | PS4 controller |
| `joint` | `joint_controller` | Skeleton node for custom joint-space torque commands | Your node/scripts |
| `direct_thrusters` | `direct_thruster_controller` | Direct PWM commands | Keyboard |

## Project layout

```
simlab/
├── simlab/uvms_backend.py            # Core backend, FCL world, planners, TFs
├── simlab/interactive_control.py     # RViz markers + menu control
├── simlab/controllers/               # One controller class per file
├── simlab/se3_ompl_planner.py        # OMPL SE(3) planning
├── simlab/cartesian_ruckig.py        # Ruckig trajectory generation
├── simlab/joystick_control.py        # PS4 teleop node
├── simlab/joint_control.py           # Joint-space torque control
├── simlab/direct_thruster_control.py # Thruster PWM keyboard control
├── simlab/collision_contact.py       # FCL contact markers + clearance
├── simlab/voxel_viz.py               # Bathymetry voxel clouds
├── simlab/bag_recorder.py            # rosbag2 MCAP recorder
├── simlab/rgb2cloudpoint.py          # RGB to pointcloud (MiDaS)
└── resource/                         # CasADi controllers + models
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
       LowLevelOptimalModelbasedController,
       OgesModelbasedController,
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
