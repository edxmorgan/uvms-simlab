"""Named waveform registry with installed user plugins."""
import hashlib
import inspect
from pathlib import Path
from functools import lru_cache
from importlib.metadata import entry_points
from .base import DisturbanceWaveform
from .builtin import BUILTINS
from .local import LOCAL_WAVEFORMS


@lru_cache(maxsize=1)
def _registry():
    registry = {}

    def register(cls, name=None):
        if not isinstance(cls, type) or not issubclass(cls, DisturbanceWaveform) or inspect.isabstract(cls):
            raise ValueError(f'{name or cls} must be a concrete DisturbanceWaveform')
        if not cls.registry_name or (name is not None and cls.registry_name != name):
            raise ValueError(f'{name}: registry_name must be nonempty and match entry point')
        if cls.registry_name in registry:
            raise ValueError(f'Duplicate disturbance waveform: {cls.registry_name}')
        registry[cls.registry_name] = cls

    for cls in (*BUILTINS, *LOCAL_WAVEFORMS):
        register(cls)
    for entry in entry_points(group='simlab.disturbances'):
        register(entry.load(), entry.name)
    return registry


def waveform_names():
    return list(_registry())


def waveform_class(name):
    try:
        return _registry()[name]
    except KeyError:
        raise ValueError(f'Unknown waveform: {name}') from None


def waveform_identity(name):
    cls = waveform_class(name)
    source = inspect.getsourcefile(cls)
    try:
        digest = hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None
    except OSError:
        digest = None
    return dict(name=name, implementation=f'{cls.__module__}.{cls.__qualname__}',
                version=cls.version, source_file=source, source_sha256=digest,
                metadata=cls.metadata())
