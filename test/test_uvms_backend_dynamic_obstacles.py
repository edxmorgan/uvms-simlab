from types import SimpleNamespace


class FakeLogger:
    def __init__(self):
        self.warns = []
        self.infos = []

    def warn(self, message):
        self.warns.append(message)

    def info(self, message):
        self.infos.append(message)


class FakeNode:
    def __init__(self):
        self.logger = FakeLogger()

    def get_logger(self):
        return self.logger


class FakeObstacleArray:
    def __init__(self, obstacle_id=""):
        self.header = SimpleNamespace(frame_id="world")
        self.obstacles = [] if not obstacle_id else [SimpleNamespace(id=obstacle_id)]


class FakeDynamicWorld:
    def __init__(self):
        self.updated = []

    def update_from_msg(self, msg):
        self.updated.append(msg)


class FakeFuture:
    def __init__(self, response):
        self._response = response

    def result(self):
        return self._response

    def add_done_callback(self, callback):
        callback(self)


class FakeClient:
    def __init__(self, *, success, message=""):
        self.success = success
        self.message = message
        self.requests = []

    def service_is_ready(self):
        return True

    def call_async(self, request):
        self.requests.append(request)
        return FakeFuture(SimpleNamespace(success=self.success, message=self.message))


def _backend(monkeypatch, *, success):
    import simlab.uvms_backend as uvms_backend

    backend = uvms_backend.UVMSBackendCore.__new__(uvms_backend.UVMSBackendCore)
    backend.node = FakeNode()
    backend.world_frame = 'world'
    backend.obstacle_edit_client = FakeClient(success=success, message="service response")
    backend.dynamic_world = FakeDynamicWorld()
    backend.dynamic_obstacle_snapshot = FakeObstacleArray("old")
    return backend


def test_apply_dynamic_obstacles_rejection_does_not_update_local_snapshot(monkeypatch):
    backend = _backend(monkeypatch, success=False)
    from ros2_control_blue_reach_5.msg import DynamicObstacleArray
    requested = DynamicObstacleArray()
    requested.header.frame_id = 'world'

    assert backend._apply_dynamic_obstacles(requested, "test obstacle")

    assert [obstacle.id for obstacle in backend.dynamic_obstacle_snapshot.obstacles] == ["old"]
    assert backend.dynamic_world.updated == []
    assert "service response" in backend.node.logger.warns[-1]
    assert backend.obstacle_edit_client.requests[-1].operation == "replace"


def test_acknowledgement_does_not_overwrite_authoritative_snapshot(monkeypatch):
    backend = _backend(monkeypatch, success=True)
    from ros2_control_blue_reach_5.msg import DynamicObstacleArray
    requested = DynamicObstacleArray()
    requested.header.frame_id = 'world'

    backend._on_dynamic_obstacle_snapshot(FakeObstacleArray("live"))
    assert backend._apply_dynamic_obstacles(requested, "test obstacle")
    assert [obstacle.id for obstacle in backend.dynamic_obstacle_snapshot.obstacles] == ["live"]
    assert backend.dynamic_world.updated == []


def test_live_snapshot_updates_recording_without_aliasing(monkeypatch):
    backend = _backend(monkeypatch, success=True)
    message = FakeObstacleArray("moving")
    message.obstacles[0].position = 3.0
    backend._on_dynamic_obstacle_snapshot(message)
    message.obstacles[0].position = 8.0
    recorded = backend.dynamic_obstacle_snapshot_for_recording()
    assert recorded.obstacles[0].position == 3.0
    recorded.obstacles[0].position = 9.0
    assert backend.dynamic_obstacle_snapshot.obstacles[0].position == 3.0


def test_generated_robots_are_not_replayed_as_editable_obstacles(monkeypatch):
    backend = _backend(monkeypatch, success=True)
    from ros2_control_blue_reach_5.msg import DynamicObstacleArray, DynamicObstacle
    scene = DynamicObstacleArray()
    scene.header.frame_id = 'world'
    scripted = DynamicObstacle(id='box', source='scripted')
    observed = DynamicObstacle(id='robot/robot_2_', source='robot', owner_robot='robot_2_')
    scene.obstacles = [scripted, observed]
    assert backend._apply_dynamic_obstacles(scene, 'recorded scene')
    request = backend.obstacle_edit_client.requests[-1]
    assert [obstacle.id for obstacle in request.obstacles.obstacles] == ['box']
    assert len(scene.obstacles) == 2








def test_collision_geometry_refresh_does_not_command_robots():
    from unittest.mock import Mock
    from simlab.uvms_backend import UVMSBackendCore
    backend = UVMSBackendCore.__new__(UVMSBackendCore)
    backend.fcl_world = Mock()
    backend.tf_buffer = Mock()
    robot = Mock()
    backend.robots = [robot]
    backend.fcl_update_callback()
    backend.fcl_world.update_from_tf.assert_called_once()
    assert robot.mock_calls == []


def test_replanning_timer_exists_only_while_enabled():
    from unittest.mock import Mock
    from simlab.uvms_backend import UVMSBackendCore
    backend = UVMSBackendCore.__new__(UVMSBackendCore)
    backend.node = Mock()
    backend.dynamic_replanning_enabled = False
    backend.dynamic_replanning_rate = 3.
    backend.dynamic_replanner_timer = None
    backend._dynamic_replanning_status_message = lambda: 'status'
    backend._recreate_dynamic_replanner_timer()
    backend.node.create_timer.assert_not_called()
    backend._enable_dynamic_replanning()
    timer = backend.dynamic_replanner_timer
    assert timer is not None
    backend.node.create_timer.assert_called_once()
    backend._disable_dynamic_replanning()
    timer.cancel.assert_called_once()
    backend.node.destroy_timer.assert_called_once_with(timer)
    assert backend.dynamic_replanner_timer is None


def test_queued_waypoint_cannot_take_over_joystick():
    from unittest.mock import Mock
    from simlab.robot import ControlMode
    from simlab.uvms_backend import UVMSBackendCore
    backend = UVMSBackendCore.__new__(UVMSBackendCore)
    robot = Mock(control_mode=ControlMode.TELEOP)
    mission = Mock(executing=True, active_index=None)
    assert not backend._dispatch_vehicle_waypoint_if_ready(robot, mission)
    mission.stop.assert_called_once()
    robot.set_control_mode.assert_not_called()
    robot.plan_vehicle_trajectory_action.assert_not_called()


def test_late_replay_reset_cannot_override_mode_roundtrip():
    from unittest.mock import Mock
    from simlab.robot import ControlMode
    from simlab.uvms_backend import UVMSBackendCore
    backend = UVMSBackendCore.__new__(UVMSBackendCore)
    controller = Mock(execution_id=1)
    controller.playback_status.return_value = 'idle'
    controller.vehicle_subsystem_mode.return_value = 'direct'
    controller.manipulator_subsystem_mode.return_value = 'direct'
    controller.reset_mode.return_value = 'simulator'
    backend._cmd_replay_controller = lambda robot: controller
    robot = Mock(controller_name='CmdReplay', _replay_reset_pending=False,
                 control_mode=ControlMode.REPLAY, _control_mode_epoch=2)
    assert backend.start_replay(robot)[0]
    controller.playback_status.return_value = 'resetting'
    callback = robot.reset_simulation_with_state.call_args.kwargs['on_success']
    robot._control_mode_epoch = 4 # REPLAY -> TELEOP -> REPLAY
    callback()
    robot.set_control_mode.assert_not_called()
    controller.mark_reset_succeeded.assert_not_called()


def test_delayed_waypoint_release_cannot_restart_after_mode_roundtrip():
    from unittest.mock import Mock
    from simlab.robot import ControlMode
    from simlab.uvms_backend import UVMSBackendCore
    backend = UVMSBackendCore.__new__(UVMSBackendCore)
    backend.node = Mock()
    robot = Mock(k_robot=0, control_mode=ControlMode.PLANNER, sim_reset_hold=True,
                 task_based_controller=False, _control_mode_epoch=1)
    backend.robot_selected = robot
    mission = Mock(executing=False, waypoints=[object()])
    backend.vehicle_waypoint_missions = {0: mission}
    backend._dispatch_vehicle_waypoint_if_ready = Mock()
    assert backend.execute_selected_vehicle_waypoints()
    callback = robot.release_simulation.call_args.kwargs['on_success']
    robot._control_mode_epoch = 3
    callback()
    mission.start.assert_not_called()
    backend._dispatch_vehicle_waypoint_if_ready.assert_not_called()


def test_stop_replay_clears_preparation_without_stale_callback_rearming():
    from unittest.mock import Mock
    from simlab.robot import ControlMode
    from simlab.uvms_backend import UVMSBackendCore
    backend = UVMSBackendCore.__new__(UVMSBackendCore)
    controller = Mock(execution_id=1)
    controller.playback_status.return_value = 'idle'
    controller.vehicle_subsystem_mode.return_value = 'direct'
    controller.manipulator_subsystem_mode.return_value = 'direct'
    controller.reset_mode.return_value = 'simulator'
    backend._cmd_replay_controller = lambda robot: controller
    robot = Mock(controller_name='CmdReplay', _replay_reset_pending=False,
                 control_mode=ControlMode.REPLAY, _control_mode_epoch=2)
    assert backend.start_replay(robot)[0]
    callback = robot.reset_simulation_with_state.call_args.kwargs['on_success']
    assert robot._replay_reset_pending
    backend.stop_replay(robot)
    assert not robot._replay_reset_pending
    callback()
    controller.mark_reset_succeeded.assert_not_called()
    assert backend.start_replay(robot)[0]
