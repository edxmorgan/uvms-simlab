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

    def wait_for_service(self, timeout_sec=0.0):
        return True

    def call_async(self, request):
        self.requests.append(request)
        return FakeFuture(SimpleNamespace(success=self.success, message=self.message))


class FakeSetDynamicObstacles:
    class Request:
        def __init__(self):
            self.obstacles = None


def _backend(monkeypatch, *, success):
    import simlab.uvms_backend as uvms_backend

    monkeypatch.setattr(uvms_backend, "SetDynamicObstacles", FakeSetDynamicObstacles)
    backend = uvms_backend.UVMSBackendCore.__new__(uvms_backend.UVMSBackendCore)
    backend.node = FakeNode()
    backend.dynamic_obstacles_client = FakeClient(success=success, message="service response")
    backend.dynamic_world = FakeDynamicWorld()
    backend.dynamic_obstacle_snapshot = FakeObstacleArray("old")
    return backend


def test_apply_dynamic_obstacles_rejection_does_not_update_local_snapshot(monkeypatch):
    backend = _backend(monkeypatch, success=False)
    requested = FakeObstacleArray("rejected")

    assert backend._apply_dynamic_obstacles(requested, "test obstacle")

    assert [obstacle.id for obstacle in backend.dynamic_obstacle_snapshot.obstacles] == ["old"]
    assert backend.dynamic_world.updated == []
    assert "rejected" in backend.node.logger.warns[-1]


def test_apply_dynamic_obstacles_acceptance_updates_local_snapshot(monkeypatch):
    backend = _backend(monkeypatch, success=True)
    requested = FakeObstacleArray("accepted")

    assert backend._apply_dynamic_obstacles(requested, "test obstacle")

    assert [obstacle.id for obstacle in backend.dynamic_obstacle_snapshot.obstacles] == ["accepted"]
    assert len(backend.dynamic_world.updated) == 1
    assert [obstacle.id for obstacle in backend.dynamic_world.updated[0].obstacles] == ["accepted"]
