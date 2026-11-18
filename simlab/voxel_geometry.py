"""Asynchronous, content-keyed voxelization of shared collision surfaces."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import os
import tempfile
import multiprocessing
import threading
import time

import numpy as np
from bringup.collision_geometry import collision_surface_mesh


class LocalVoxelCache:
    def __init__(self, directory, pitch):
        if not np.isfinite(pitch) or pitch <= 0:
            raise ValueError('voxel pitch must be finite and positive')
        self.directory = Path(directory)
        self.pitch = float(pitch)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='voxel_geometry')
        self.jobs = {}
        self.cancellations = {}

    def request(self, spec):
        """Return immutable local centers when ready, otherwise None."""
        if spec not in self.jobs:
            cancel = threading.Event()
            self.cancellations[spec] = cancel
            self.jobs[spec] = self.executor.submit(self._run, spec, cancel)
        future = self.jobs[spec]
        return future.result() if future.done() else None

    def retain(self, specs):
        for spec in self.jobs.keys() - set(specs):
            self.cancellations.pop(spec).set()
            self.jobs.pop(spec).cancel()

    def _run(self, spec, cancel):
        # Native voxelization cannot be interrupted in a thread. A spawned
        # worker can be terminated on scene replacement, timeout or shutdown.
        if cancel.is_set():
            raise RuntimeError('Voxel build cancelled')
        ctx = multiprocessing.get_context('spawn')
        receiver, sender = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_voxel_worker,
            args=(sender, spec, self.pitch, self.directory), daemon=True)
        process.start()
        sender.close()
        deadline = time.monotonic() + 60.
        try:
            while not receiver.poll(.05):
                if cancel.is_set():
                    raise RuntimeError('Voxel build cancelled')
                if time.monotonic() >= deadline:
                    raise RuntimeError('Voxel visualization build exceeded 60 s; collision geometry is unchanged')
                if not process.is_alive():
                    raise RuntimeError('Voxel visualization worker exited without a result')
            success, value = receiver.recv()
            if not success:
                raise ValueError(value)
            value.setflags(write=False)
            return value
        finally:
            receiver.close()
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.)
            if process.is_alive():
                process.kill()
                process.join()

    def close(self):
        for cancel in self.cancellations.values():
            cancel.set()
        self.executor.shutdown(wait=True, cancel_futures=True)


def display_point_subsets(arrays, budget):
    """Bound display work, sharing capacity fairly among nonempty objects.

    Select existing voxel centers deterministically across each object's array.
    Small objects keep all their points; unused shares go to larger objects.
    Zero budget requests the complete diagnostic cloud.
    """
    if budget < 0:
        raise ValueError('display point budget must be nonnegative')
    counts = [len(points) for points in arrays]
    if budget == 0 or sum(counts) <= budget:
        return arrays
    allocations = [0] * len(arrays)
    remaining = budget
    order = sorted(range(len(arrays)), key=lambda i: counts[i])
    for position, i in enumerate(order):
        allocations[i] = min(counts[i], remaining // (len(order) - position))
        remaining -= allocations[i]
    return [points if count == len(points) else
            points[np.linspace(0, len(points) - 1, count, dtype=np.int64)]
            for points, count in zip(arrays, allocations)]


def transform_centers(points, rotation, translation):
    return points @ np.asarray(rotation).T + np.asarray(translation)


def surface_voxel_centers(mesh, pitch):
    """Rasterize triangle/voxel intersections in bounded candidate batches.

    Project on each triangle's dominant plane, then apply the separating-axis
    test to candidate cubes. Work scales with surface area, without creating a
    subdivided mesh or a dense grid covering empty volume.
    """
    vertices = np.asarray(mesh.vertices, dtype=float) / pitch
    lower = np.ceil(vertices.min(axis=0) - .5 - 1e-9).astype(np.int64)
    upper = np.floor(vertices.max(axis=0) + .5 + 1e-9).astype(np.int64)
    shape = upper - lower + 1
    hits = []
    basis = np.eye(3)
    for face in mesh.faces:
        triangle = vertices[face]
        edges = np.roll(triangle, -1, axis=0) - triangle
        normal = np.cross(edges[0], edges[1])
        depth = int(np.argmax(np.abs(normal)))
        if abs(normal[depth]) < 1e-15:
            continue
        plane = [axis for axis in range(3) if axis != depth]
        lo = np.ceil(triangle.min(axis=0) - .5 - 1e-9).astype(np.int64)
        hi = np.floor(triangle.max(axis=0) + .5 + 1e-9).astype(np.int64)
        width = hi[plane[1]] - lo[plane[1]] + 1
        count = int((hi[plane[0]] - lo[plane[0]] + 1) * width)
        axes = np.vstack([basis, normal, np.cross(edges[:, None, :], basis).reshape(-1, 3)])
        lengths = np.linalg.norm(axes, axis=1)
        axes = axes[lengths > 1e-15] / lengths[lengths > 1e-15, None]
        projection = triangle @ axes.T
        radius = .5 * np.abs(axes).sum(axis=1)
        minimum = projection.min(axis=0) - radius - 1e-9
        maximum = projection.max(axis=0) + radius + 1e-9
        for start in range(0, count, 16384):
            index = np.arange(start, min(start + 16384, count))
            centers = np.empty((len(index), 3), dtype=np.int64)
            centers[:, plane[0]] = lo[plane[0]] + index // width
            centers[:, plane[1]] = lo[plane[1]] + index % width
            z = (triangle[0] @ normal - centers[:, plane] @ normal[plane]) / normal[depth]
            base = np.floor(z).astype(np.int64)
            for offset in (-1, 0, 1, 2):
                centers[:, depth] = base + offset
                projections = centers @ axes.T
                valid = np.all((projections >= minimum) & (projections <= maximum), axis=1)
                if np.any(valid):
                    cells = centers[valid] - lower
                    hits.append((cells[:, 0] * shape[1] + cells[:, 1]) * shape[2] + cells[:, 2])
    if not hits:
        return np.empty((0, 3), dtype=np.float32)
    indices = np.unique(np.concatenate(hits))
    points = np.column_stack((indices // (shape[1] * shape[2]),
        (indices // shape[2]) % shape[1], indices % shape[2])) + lower
    return (points * pitch).astype(np.float32)


def _build_points(spec, pitch, directory):
    mesh = collision_surface_mesh(*spec)
    digest = hashlib.sha256(b'collision-surface-voxels-sat-v2')
    digest.update(np.asarray([pitch], dtype='<f8').tobytes())
    digest.update(np.asarray(mesh.vertices, dtype='<f8').tobytes())
    digest.update(np.asarray(mesh.faces, dtype='<i8').tobytes())
    path = directory / (digest.hexdigest() + '.npy')
    try:
        points = np.load(path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError('invalid voxel cache')
    except (OSError, ValueError):
        points = surface_voxel_centers(mesh, pitch)
        temporary = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=directory, suffix='.npy', delete=False) as stream:
                temporary = stream.name
                np.save(stream, points)
            os.replace(temporary, path)
            temporary = None
        except OSError:
            pass  # Disk caching is optional, even in read-only deployments.
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
    points.setflags(write=False)
    return points


def _voxel_worker(connection, spec, pitch, directory):
    try:
        connection.send((True, _build_points(spec, pitch, directory)))
    except Exception as exc:
        connection.send((False, str(exc)))
    finally:
        connection.close()
