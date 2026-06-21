from types import SimpleNamespace
import io
import json

from simlab.robot import Robot


class _Logger:
    def warn(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass


class _ClockTime:
    nanoseconds = 1234567890


class _Clock:
    def now(self):
        return _ClockTime()


class _Node:
    def get_logger(self):
        return _Logger()

    def get_clock(self):
        return _Clock()


def _vec(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def _quat(x, y, z, w):
    return SimpleNamespace(x=x, y=y, z=z, w=w)


def _obstacle_msg():
    obstacle = SimpleNamespace(
        id="path_blocker",
        collision_type=1,
        collision_dimensions=[0.8],
        visual_type=1,
        visual_dimensions=[0.8],
        visual_mesh_resource="",
        pose=SimpleNamespace(position=_vec(1.0, 2.0, 3.0), orientation=_quat(0.0, 0.0, 0.0, 1.0)),
        twist=SimpleNamespace(linear=_vec(0.1, 0.2, 0.3), angular=_vec(0.4, 0.5, 0.6)),
        color=SimpleNamespace(r=1.0, g=0.2, b=0.1, a=0.8),
    )
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="world", stamp=SimpleNamespace(sec=7, nanosec=8)),
        obstacles=[obstacle],
    )


def _robot(tmp_path):
    robot = Robot.__new__(Robot)
    robot.node = _Node()
    robot.prefix = "robot_1_"
    robot.sim_time = 4.2
    robot._replay_obstacle_record_path = tmp_path / "session_dynamic_obstacles.json"
    robot._replay_obstacle_record = {"schema": "simlab_cmd_replay_dynamic_obstacles_v1", "snapshots": []}
    robot.dynamic_obstacle_snapshot_provider = _obstacle_msg
    return robot


def test_replay_obstacle_summary_uses_current_snapshot(tmp_path):
    robot = _robot(tmp_path)

    count, ids = robot._dynamic_obstacle_summary_for_recording()

    assert count == 1
    assert ids == "path_blocker"


def test_replay_obstacle_sidecar_records_start_stop_snapshots(tmp_path):
    robot = _robot(tmp_path)

    robot._append_replay_obstacle_snapshot("start")
    robot._append_replay_obstacle_snapshot("stop")

    data = json.loads(robot._replay_obstacle_record_path.read_text(encoding="utf-8"))
    assert [snapshot["event"] for snapshot in data["snapshots"]] == ["start", "stop"]
    snapshot = data["snapshots"][0]
    assert snapshot["dynamic_obstacle_count"] == 1
    assert snapshot["dynamic_obstacle_ids"] == "path_blocker"
    obstacle = snapshot["dynamic_obstacles"]["obstacles"][0]
    assert obstacle["id"] == "path_blocker"
    assert obstacle["pose"]["position"] == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert obstacle["twist"]["angular"] == {"x": 0.4, "y": 0.5, "z": 0.6}


def test_stop_replay_recording_clears_obstacle_sidecar_state(tmp_path):
    robot = _robot(tmp_path)
    robot._replay_record_handle = io.StringIO()
    robot._replay_record_writer = object()
    robot._replay_record_path = tmp_path / "session.csv"
    robot._replay_record_controller = object()
    robot._replay_last_cmd_body_wrench = [1.0] * 6
    robot._replay_last_cmd_arm_tau = [1.0] * 5
    robot._replay_last_recorded_sim_time = 1.0

    robot._stop_replay_session_recording("done")

    assert robot._replay_record_handle is None
    assert robot._replay_record_writer is None
    assert robot._replay_record_path is None
    assert robot._replay_record_controller is None
    assert robot._replay_obstacle_record_path is None
    assert robot._replay_obstacle_record is None
    assert robot._replay_last_recorded_sim_time is None
