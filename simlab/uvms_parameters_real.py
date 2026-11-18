# Copyright (C) 2025 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.

import numpy as np

from simlab.uvms_parameters_sim import (
    ReachParams as SimReachParams,
    VehicleControllerParams as SimVehicleControllerParams,
)


class ReachParams(SimReachParams):
    profile_name = "real"

    # Match the established hardware joystick effort range (Nm). The driver
    # independently enforces actuator current, position and feedback limits.
    u_max = np.array([3.0, 2.0, 2.0, 1.0])
    u_min = -u_max

    # First four entries are arm joints; fifth is the prismatic gripper.
    # Gripper SMC: noise-replay tuning for 20 Hz feedback, pending physical
    # motion validation. With nominal m=300 and b=400, these gains reduce
    # near-target velocity-to-effort sensitivity from 1850 to 237.5 N/(m/s).
    # The boundary is in m/s, switching in m/s^2; no PID substitution.
    # Wrist commissioning candidate: with nominal I=0.007172 kg m^2,
    # near-surface position stiffness is I*8*(12+25/2)=1.406 Nm/rad.
    # The broad wrist boundary avoids a high switching slope at 20 Hz feedback.
    smc_surface = np.array([6., 6., 4., 8., 1.])
    smc_damping = np.array([2., 2., 2., 12., 1.])
    smc_switching = np.array([.1, .1, .1, 25., .0005])
    smc_boundary = np.array([.05, .05, .05, 2., .004])

    # Explicit hardware commissioning gains (acceleration units), not calibrated
    # hardware tracking results. Keep the previous values pending physical tuning.
    invdyn_kp = np.array([40., 40., 40., 750.])
    invdyn_ki = np.zeros(4)
    invdyn_kd = np.array([7., 7., 7., 50.])

    pid_kp = np.array([2.0, 3.0, 1.0, 1.5])
    pid_ki = np.array([0.0, 0.0, 0.0, 0.0])
    pid_kd = np.array([0.0, 0.0, 0.0, 0.0])

class VehicleControllerParams(SimVehicleControllerParams):
    profile_name = "real"
