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
