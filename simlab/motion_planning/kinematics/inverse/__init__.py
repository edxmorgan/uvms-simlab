"""IK registry and implementation selection."""
from .base import IKCapability
from .whole_body import EndpointIK
from .arm import ManipulatorPositionIK


DEFAULT_IK_CLASSES = [EndpointIK, ManipulatorPositionIK]


def ik_class(name, *, capability=None, task=None):
    """Typed capability selection; task is retained for existing Python callers."""
    if capability is None:
        capability = IKCapability(task)
    if not isinstance(capability, IKCapability):
        raise TypeError('capability must be IKCapability')
    if task is not None and IKCapability(task) != capability:
        raise ValueError('Conflicting IK capability and task')
    names = [cls.registry_name for cls in DEFAULT_IK_CLASSES]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate IK registry names')
    for cls in DEFAULT_IK_CLASSES:
        if cls.registry_name == name:
            if cls.capability != capability:
                raise ValueError(f"IK '{name}' supports {cls.capability.value}, not {capability.value}")
            return cls
    raise KeyError(f"Unknown IK '{name}'; available: {', '.join(names)}")
