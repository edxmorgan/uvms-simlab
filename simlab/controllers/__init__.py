from simlab.controllers.base import ControllerTemplate
from simlab.controllers.cmd_replay import CmdReplayController
from simlab.controllers.invDyn import LowLevelInvDynController
from simlab.controllers.pid import LowLevelPidController

try:
    from simlab.controllers.oges import NAMOR_IMPORT_ERROR, OgesModelbasedController
except ImportError:
    NAMOR_IMPORT_ERROR = True
    OgesModelbasedController = None

try:
    from simlab.controllers.dr_oges import (
        NAMOR_DR_IMPORT_ERROR,
        DistributionallyRobustOgesController,
    )
except ImportError:
    NAMOR_DR_IMPORT_ERROR = True
    DistributionallyRobustOgesController = None

DEFAULT_CONTROLLER_CLASSES = [
    LowLevelPidController,
    LowLevelInvDynController,
    CmdReplayController,
]

if OgesModelbasedController is not None and NAMOR_IMPORT_ERROR is None:
    DEFAULT_CONTROLLER_CLASSES.append(OgesModelbasedController)
if (
    DistributionallyRobustOgesController is not None
    and NAMOR_IMPORT_ERROR is None
    and NAMOR_DR_IMPORT_ERROR is None
):
    DEFAULT_CONTROLLER_CLASSES.append(DistributionallyRobustOgesController)

__all__ = [
    "CmdReplayController",
    "ControllerTemplate",
    "DEFAULT_CONTROLLER_CLASSES",
    "LowLevelInvDynController",
    "LowLevelPidController",
]

if OgesModelbasedController is not None and NAMOR_IMPORT_ERROR is None:
    __all__.append("OgesModelbasedController")
if (
    DistributionallyRobustOgesController is not None
    and NAMOR_IMPORT_ERROR is None
    and NAMOR_DR_IMPORT_ERROR is None
):
    __all__.append("DistributionallyRobustOgesController")
