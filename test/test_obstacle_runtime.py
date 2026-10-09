import copy
import math

import pytest
import trimesh

from bringup.obstacle_description import obstacle_from_config
from ros2_control_blue_reach_5.msg import DynamicObstacleArray
from simlab.dynamic_obstacle_sources import obstacle_behavior_source_class
from simlab.dynamic_obstacle_sources.scripted_motion import ScriptedMotionSource
from simlab.obstacle_runtime import ObstacleRuntime, ObstacleRevisionConflict


def obstacle(name='one', **kwargs):
    return obstacle_from_config({'id': name, 'type': 'sphere', 'dimensions': [0.8], **kwargs}, 0)


def scene(*items):
    result = DynamicObstacleArray()
    result.header.frame_id = 'world'
    result.obstacles = list(items)
    return result


def runtime(*items, source=None, **kwargs):
    return ObstacleRuntime(source or ScriptedMotionSource(), scene(*items), **kwargs)


def test_registry():
    assert obstacle_behavior_source_class('scripted_motion') is ScriptedMotionSource
    with pytest.raises(ValueError):
        obstacle_behavior_source_class('missing')


def test_add_without_robot_or_path_preserves_requested_pose_size():
    host = runtime()
    host.edit(expected_revision=0, upsert=[obstacle(position=[3., 2., -1.])])
    _, state = host.step(10.)
    item = state.obstacles[0]
    assert (item.pose.position.x, item.pose.position.y, item.pose.position.z) == (3., 2., -1.)
    assert list(item.collision_dimensions) == [0.8]


def test_motion_integrated_once_with_world_frame_angular_velocity():
    host = runtime(obstacle(linear_velocity=[2., 0., 0.], angular_velocity=[0., 0., math.pi]))
    _, state = host.step(0.5)
    assert state.obstacles[0].pose.position.x == 1.
    q = state.obstacles[0].pose.orientation
    assert q.z == pytest.approx(math.sqrt(0.5))
    assert q.w == pytest.approx(math.sqrt(0.5))


def test_stale_edit_after_motion_cannot_rewind_world():
    host = runtime(obstacle(linear_velocity=[1., 0., 0.]))
    host.step(1.)
    with pytest.raises(ObstacleRevisionConflict):
        host.edit(expected_revision=0, upsert=[obstacle('two')])
    assert host.snapshot()[1].obstacles[0].pose.position.x == 1.


def test_edit_is_atomic_and_snapshots_detached():
    host = runtime(obstacle())
    invalid = obstacle('two')
    invalid.collision_dimensions = [-1.]
    with pytest.raises(ValueError):
        host.edit(expected_revision=0, upsert=[invalid], remove=['one'])
    revision, state = host.snapshot()
    assert revision == 0
    state.obstacles.clear()
    assert len(host.snapshot()[1].obstacles) == 1
    host.edit(expected_revision=0, upsert=[obstacle('two')], remove=['one'])
    assert host.snapshot()[1].obstacles[0].id == 'two'


@pytest.mark.parametrize('dt', [-1., float('nan'), float('inf')])
def test_bad_time_does_not_fault_or_advance_source(dt):
    host = runtime(obstacle())
    with pytest.raises(ValueError):
        host.step(dt)
    assert host.step(0.)[0] == 0
    assert host.step(1.)[0] == 1


def test_coupled_policy_sees_all_agents_and_observations():
    class Coupled(ScriptedMotionSource):
        def step(self, state, context):
            assert context.seed == 42
            assert context.observations['robot'] == [1, 2, 3]
            context.observations['robot'].clear()
            state.obstacles[0].pose.position.x = state.obstacles[1].pose.position.x
            return state
    host = runtime(obstacle('a'), obstacle('b', position=[5., 0., 0.]), source=Coupled(), seed=42)
    observations = {'robot': [1, 2, 3]}
    assert host.step(1., observations=observations)[1].obstacles[0].pose.position.x == 5.
    assert observations == {'robot': [1, 2, 3]}


@pytest.mark.parametrize('change', ['geometry', 'ids', 'frame', 'nan'])
def test_bad_policy_output_holds_last_state_and_requires_reset(change):
    class Bad(ScriptedMotionSource):
        def step(self, state, context):
            if change == 'geometry':
                state.obstacles[0].collision_dimensions = [2.]
            elif change == 'ids':
                state.obstacles.clear()
            elif change == 'frame':
                state.header.frame_id = 'other'
            else:
                state.obstacles[0].pose.position.x = float('nan')
            return state
    host = runtime(obstacle(), source=Bad())
    before = host.snapshot()
    with pytest.raises(ValueError):
        host.step(1.)
    assert host.snapshot() == before
    with pytest.raises(RuntimeError, match='reset'):
        host.step(1.)
    assert host.reset()[0] == 1


def test_reset_replays_initial_scene_and_close_is_idempotent():
    class Source(ScriptedMotionSource):
        closed = 0
        def close(self):
            self.closed += 1
    source = Source()
    host = runtime(obstacle(linear_velocity=[1., 0., 0.]), source=source)
    first = host.step(2.)[1]
    host.edit(expected_revision=1, upsert=[obstacle('second')])
    host.reset()
    assert host.step(2.)[1] == first
    host.close()
    host.close()
    assert source.closed == 1
    with pytest.raises(RuntimeError, match='closed'):
        host.step(1.)


@pytest.mark.parametrize('kind,dims', [('sphere', [0.4]), ('box', [1., 2., 3.]), ('cylinder', [0.4, 2.]), ('mesh', [])])
def test_all_geometry_kinds_keep_scale_and_shape(tmp_path, kind, dims):
    config = {'type': kind, 'dimensions': dims}
    if kind == 'mesh':
        path = tmp_path / 'arbitrary.obj'
        trimesh.creation.box().export(path)
        config.update(collision_mesh_resource=str(path), collision_mesh_scale=[2., 3., 4.])
    item = obstacle(**config, linear_velocity=[1., 0., 0.])
    host = runtime(item)
    moved = host.step(1.)[1].obstacles[0]
    original = copy.deepcopy(moved)
    original.pose = item.pose
    assert original == item
