from simlab.controllers.base import ControllerTemplate
from simlab.controllers.invDyn import LowLevelInvDynController
from simlab.controllers.oges import OgesModelbasedController
from simlab.controllers.pid import LowLevelPidController
from simlab.controllers.mpc import MPCController

DEFAULT_CONTROLLER_CLASSES = [
    LowLevelPidController,
    LowLevelInvDynController,
    OgesModelbasedController,
    MPCController,
]

__all__ = [
    "ControllerTemplate",
    "DEFAULT_CONTROLLER_CLASSES",
    "LowLevelInvDynController",
    "LowLevelPidController",
    "OgesModelbasedController",
    "MPCController",
]
