"""Load the existing exported forward model without altering its mathematics."""
from pathlib import Path
import casadi as ca


def load_forward_kinematics(resource_directory):
    """Return FK(q, mounting XYZ/RPY, world XYZ/RPY, tip XYZ/RPY).

    The model returns link poses in world NWU as XYZ + quaternion WXYZ.
    The final output is the tool pose. Inputs use the exported joint order.
    Each call loads a separate model instance for its owner.
    """
    return ca.Function.load(str(Path(resource_directory) / 'model_functions/arm/fk_eval.casadi'))
