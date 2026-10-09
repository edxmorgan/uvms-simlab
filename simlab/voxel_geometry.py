"""Asynchronous, content-keyed voxelization of shared collision surfaces."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import os
import tempfile

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

    def request(self, spec):
        """Return immutable local centers when ready, otherwise None."""
        if spec not in self.jobs:
            self.jobs[spec] = self.executor.submit(self._build, spec)
        future = self.jobs[spec]
        return future.result() if future.done() else None

    def retain(self, specs):
        for spec in self.jobs.keys() - set(specs):
            self.jobs.pop(spec).cancel()

    def _build(self, spec):
        mesh = collision_surface_mesh(*spec)
        digest = hashlib.sha256(b'collision-surface-voxels-v1')
        digest.update(np.asarray([self.pitch], dtype='<f8').tobytes())
        digest.update(np.asarray(mesh.vertices, dtype='<f8').tobytes())
        digest.update(np.asarray(mesh.faces, dtype='<i8').tobytes())
        path = self.directory / (digest.hexdigest() + '.npy')
        try:
            points = np.load(path, allow_pickle=False)
            if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
                raise ValueError('invalid voxel cache')
        except (OSError, ValueError):
            points = mesh.voxelized(pitch=self.pitch, method='subdivide').points.copy()
            temporary = None
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=self.directory, suffix='.npy', delete=False) as stream:
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

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


def transform_centers(points, rotation, translation):
    return points @ np.asarray(rotation).T + np.asarray(translation)
