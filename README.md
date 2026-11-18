# uvms-simlab

ROS 2 controllers, planning, RViz interaction, teleoperation and experiment tools
for [uvms-simulator](https://github.com/edxmorgan/uvms-simulator).
Exported ROS package: `simlab`.

## Use SimLab

Install and build both packages using the shared
[installation guide](https://github.com/edxmorgan/uvms-simulator/blob/main/doc/installation.rst).
Follow [First Simulation](https://github.com/edxmorgan/uvms-simulator/blob/main/doc/userdoc.rst)
for launch and operation.

- [Controls, waypoints and obstacles](https://github.com/edxmorgan/uvms-simulator/blob/main/doc/controls_and_menus.rst)
- [Replay, recording and disturbance injection](https://github.com/edxmorgan/uvms-simulator/blob/main/doc/replay_and_experiments.rst)
- [ROS API](https://github.com/edxmorgan/uvms-simulator/blob/main/doc/services_and_interfaces.rst)
- [Batch RL training](https://github.com/edxmorgan/uvms-simulator/blob/main/doc/rl_training.rst)

## Extend SimLab

The [developer guide](https://github.com/edxmorgan/uvms-simulator/blob/main/doc/developer_guide.rst)
covers controller, planner, trajectory, IK, obstacle and waveform plugins, model
exports and testing.

- `simlab/controllers/`: feedback and replay controllers.
- `simlab/motion_planning/`: planners, trajectories, replanning and kinematics.
- `simlab/disturbances/`: disturbance waveforms.
- `simlab/dynamic_obstacle_sources/`: obstacle behavior sources.
- `resource/`: model exports, profiles and assets.
- `uvms_rl/`: batch environments, tasks and training.

## License

[AGPL-3.0-or-later](LICENSE). Imported assets retain their bundled licenses.
