# Copyright (C) 2025 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import itertools

import casadi as cs
import numpy as np


class ReachParams:
    root = "base_link"
    tip = "alpha_standard_jaws_base_link"

    relative_urdf_path = "/resources/urdf/alpha_5_robot.urdf"

    joint_min = np.array([1.00, 0.01, 0.01, 0.01])
    joint_max = np.array([5.50, 3.40, 3.40, 5.70])
    joint_home = np.array([3.1, 0.7, 0.4, 2.1])
    endeffector_wrt_base_home = np.array([0.09669330716133118, 0.0, 0.1003517135977745])

    grasper_open = 0.014
    grasper_close = 0.0

    joint_limits = list(zip(joint_min.tolist(), joint_max.tolist()))
    joint_limits_configurations = np.array(list(itertools.product(*joint_limits)))

    u_min = np.array([-1.5, -1.0, -1.0, -0.54])
    u_max = np.array([1.5, 1.0, 1.0, 0.54])

    pid_kp = np.array([2.0, 3.0, 1.0, 3.0])
    pid_ki = np.array([0.0, 0.0, 0.0, 0.0])
    pid_kd = np.array([0.0, 0.0, 0.0, 0.0])

    invdyn_kp = np.array([40.0, 40.0, 40.0, 750.0])
    invdyn_ki = np.array([0.0, 0.0, 0.0, 0.0])
    invdyn_kd = np.array([7.0, 7.0, 7.0, 50.0])

    grasper_kp = np.array([1000.0])
    grasper_ki = np.array([0.0])
    grasper_kd = np.array([0.0])

    grasper_u_min = np.array([-10.0])
    grasper_u_max = np.array([10.0])

    gravity = 0.0

    # Transformation of UV body frame to manipulator base.
    base_T0_new = [0.190, 0.000, -0.120, 3.141592653589793, 0.000, 0.000]
    tipOffset = [0.00, 0.00, 0.04, 0.00, 0.00, 0.00]
    sim_p = cs.vertcat(
        1.94000000e-01, 4.29000000e-01, 1.14999999e-01, 3.32999998e-01,
        -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -4.29000003e-02,
        1.96649101e-02, 4.29000003e-02, 2.88077923e-03, 7.23516749e-03,
        9.16434754e-03, 2.16416476e-03, -1.19076924e-03, 8.07346553e-03,
        7.10109586e-01, 7.10109586e-01, 1.99576149e-06, -0.00000000e+00,
        -0.00000000e+00, -0.00000000e+00, 1.10178508e-01, 1.83331277e-01,
        1.04292121e-01, -3.32240937e-02, -8.30350362e-02, -3.83631263e-02,
        1.18956416e-01, 1.22363853e-01, 4.34411664e-03, -3.96112974e-04,
        -2.13904668e-02, -1.77228242e-03, 1.92510932e-02, 2.56548460e-02,
        7.17220917e-03, 1.48789886e-03, 4.53687373e-04, -1.09861913e-03,
        2.39569756e+00, 2.23596482e+00, 8.19671021e-01, 3.57249665e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
        0, 0, 0, 0,
        0, 0, gravity,
        0, 0, 0, 0,
        0.19, 0, -0.12, 3.14159, 0, 0,
        0, 0, 0, 0, 0, 0,
    )


class VehicleControllerParams:
    model_params = np.array(
        [
            3.72028553e+01,
            2.21828075e+01,
            6.61734807e+01,
            3.38909801e+00,
            6.41362046e-01,
            6.41362034e-01,
            3.38909800e+00,
            1.39646394e+00,
            4.98032205e-01,
            2.53118738e+00,
            1.05000000e+02,
            9.78296453e+01,
            8.27479545e-01,
            1.36822559e-01,
            4.25841171e+00,
            -7.36416666e+01,
            -3.36082112e+01,
            -8.94055107e+01,
            -2.98736214e+00,
            -1.57921531e+00,
            -3.39766499e+00,
            -1.47912104e-04,
            -5.16373030e-04,
            -9.85522538e+01,
            -3.05907788e-02,
            -1.27877517e-01,
            -1.63514832e+00,
        ],
        dtype=float,
    )

    pid_kp = np.array([40.0, 40.0, 40.0, 10.0, 10.0, 10.0], dtype=float)
    pid_ki = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    pid_kd = np.array([15.0, 15.0, 15.0, 2.0, 2.0, 5.0], dtype=float)

    invdyn_kp = np.array([3.0, 3.0, 3.0, 0.5, 5.0, 0.4], dtype=float)
    invdyn_ki = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    invdyn_kd = np.array([5.0, 5.0, 5.0, 1.5, 10.0, 1.5], dtype=float)

    u_min = np.array([-20.0, -20.0, -20.0, -5.0, -5.0, -5.0], dtype=float)
    u_max = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 5.0], dtype=float)
    i_limit = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=float)
    v_c = np.zeros(6, dtype=float)
