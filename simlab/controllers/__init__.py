from simlab.controllers.base import ControllerTemplate
from simlab.controllers.invDyn import LowLevelInvDynController
from simlab.controllers.pid import LowLevelPidController

try:
    from simlab.controllers.oges import NAMOR_IMPORT_ERROR, OgesModelbasedController
except ImportError:
    NAMOR_IMPORT_ERROR = True
    OgesModelbasedController = None

DEFAULT_CONTROLLER_CLASSES = [
    LowLevelPidController,
    LowLevelInvDynController,
]

if OgesModelbasedController is not None and NAMOR_IMPORT_ERROR is None:
    DEFAULT_CONTROLLER_CLASSES.append(OgesModelbasedController)

__all__ = [
    "ControllerTemplate",
    "DEFAULT_CONTROLLER_CLASSES",
    "LowLevelInvDynController",
    "LowLevelPidController",
]

if OgesModelbasedController is not None and NAMOR_IMPORT_ERROR is None:
    __all__.append("OgesModelbasedController")
