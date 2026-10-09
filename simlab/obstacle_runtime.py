"""Transport-independent, single-writer host for obstacle behavior sources.

No ROS publications or robot-path assumptions. The ROS adapter uses this host
as the sole owner of obstacle motion.
"""
import copy
import math
import threading

from bringup.obstacle_description import normalize_obstacle, validate_unique_obstacle_ids
from simlab.dynamic_obstacle_sources.behavior import ObstacleBehaviorContext


class ObstacleRevisionConflict(ValueError):
    pass


class ObstacleRuntime:
    def __init__(self, source, scene, *, config=None, seed=0, validator=None):
        self._lock = threading.RLock()
        self._source = source
        self._validator = validator
        self._closed = False
        self._fault = None
        self._seed = int(seed)
        self._revision = 0
        self._time = 0.0
        self._frame = scene.header.frame_id
        if not self._frame:
            raise ValueError("obstacle scene requires an explicit world frame")
        self._scene = self._validate(scene)
        self._initial = copy.deepcopy(self._scene)
        try:
            source.initialize(copy.deepcopy(self._scene), copy.deepcopy(config or {}), self._seed)
        except Exception:
            source.close()
            raise

    def _validate(self, scene):
        result = copy.deepcopy(scene)
        if result.header.frame_id != self._frame:
            raise ValueError("obstacle scene frame cannot change")
        if any(not obstacle.id.strip() for obstacle in result.obstacles):
            raise ValueError("every obstacle requires an explicit stable ID")
        validate_unique_obstacle_ids(result.obstacles)
        result.obstacles = [normalize_obstacle(o) for o in result.obstacles]
        if self._validator is not None:
            self._validator(copy.deepcopy(result))
        return result

    def _ensure_active(self):
        if self._closed:
            raise RuntimeError("obstacle runtime is closed")
        if self._fault is not None:
            raise RuntimeError("obstacle source faulted; reset is required") from self._fault

    def snapshot(self):
        """Return (revision, detached authoritative scene) under one lock."""
        with self._lock:
            return self._revision, copy.deepcopy(self._scene)

    def initial_snapshot(self):
        with self._lock:
            return copy.deepcopy(self._initial)

    def edit(self, *, expected_revision, upsert=(), remove=()):
        """Atomic ID-based edits; stale callers cannot overwrite newer state."""
        with self._lock:
            self._ensure_active()
            if expected_revision != self._revision:
                raise ObstacleRevisionConflict(f"expected revision {expected_revision}, current {self._revision}")
            upsert, remove = list(upsert), list(remove)
            validate_unique_obstacle_ids(upsert)
            if len(set(remove)) != len(remove) or set(remove) & {o.id for o in upsert}:
                raise ValueError("duplicate or conflicting obstacle edits")
            items = {o.id: o for o in self._scene.obstacles}
            for name in remove:
                if name not in items:
                    raise ValueError(f"unknown obstacle '{name}'")
                del items[name]
            items.update({o.id: o for o in upsert})
            candidate = copy.deepcopy(self._scene)
            candidate.obstacles = list(items.values())
            candidate = self._validate(candidate)
            try:
                self._source.on_edit(copy.deepcopy(candidate))
            except Exception as exc:
                self._fault = exc
                raise
            self._scene = candidate
            self._revision += 1
            return self.snapshot()

    def step(self, dt, *, observations=None):
        with self._lock:
            self._ensure_active()
            dt = float(dt)
            if not math.isfinite(dt) or dt < 0:
                raise ValueError("dt must be finite and nonnegative")
            if dt == 0:
                return self.snapshot()
            context = ObstacleBehaviorContext(self._time, dt, self._seed,
                                              copy.deepcopy(observations or {}))
            try:
                candidate = self._validate(self._source.step(copy.deepcopy(self._scene), context))
                old = {o.id: o for o in self._scene.obstacles}
                new = {o.id: o for o in candidate.obstacles}
                if old.keys() != new.keys():
                    raise ValueError("source must preserve obstacle IDs; use explicit edits for topology")
                for name, obstacle in new.items():
                    unchanged = copy.deepcopy(obstacle)
                    unchanged.pose, unchanged.twist = old[name].pose, old[name].twist
                    if unchanged != old[name]:
                        raise ValueError("source step may change only pose and twist")
            except Exception as exc:
                # A stateful policy may already have advanced internally. Do not
                # retry it on the old scene until its reset hook has run.
                self._fault = exc
                raise
            self._scene = candidate
            self._time += dt
            self._revision += 1
            return self.snapshot()

    def reset(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("obstacle runtime is closed")
            candidate = self._validate(self._initial)
            try:
                self._source.reset(copy.deepcopy(candidate), self._seed)
            except Exception as exc:
                self._fault = exc
                raise
            self._scene = candidate
            self._time = 0.0
            self._fault = None
            self._revision += 1
            return self.snapshot()

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._source.close()
