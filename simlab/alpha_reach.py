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

import numpy as np
import casadi as cs
import itertools

class Params:
    root = "base_link"
    tip = "alpha_standard_jaws_base_link"

    relative_urdf_path = f"/resources/urdf/alpha_5_robot.urdf"

    A_hull= np.array([[ 0.05230925, -0.99863093],
    [-0.99994784, -0.0102135 ],
    [ 0.9999956 , -0.00296812],
    [ 0.99014868, -0.14001997],
    [ 0.07555474,  0.99714166],
    [ 0.94837875,  0.31713994],
    [ 0.08636132,  0.99626388],
    [-0.0913111 , -0.99582242],
    [-0.98080565, -0.19498787],
    [-0.07451808,  0.99721966],
    [-0.0232651 ,  0.99972933]])
    b_hull = np.array([-0.74414354,  2.54983534, -3.75960604, -3.5294386 , -2.43412911,
    -4.22436006, -2.4698489 , -0.31876642,  2.67460935, -1.96197269,
    -2.11955509])
    
    joint_min = np.array([1.00, 0.01, 0.01, 0.01])
    joint_max = np.array([5.50, 3.40, 3.40, 5.70])

    joint_limits = list(zip(joint_min.tolist(), joint_max.tolist()))
    joint_limits_configurations = np.array(list(itertools.product(*joint_limits)))

    u_min = np.array([-1.5, -1, -1, -0.54])
    u_max = np.array([1.5, 1, 1, 0.54])
    
    Kp = cs.vertcat(10.0, 10.0, 10.0, 2.0)
    Ki = cs.vertcat(0.0, 0.0, 0.0, 0.0)
    Kd = cs.vertcat(1.0, 1.0, 1.0, 1.0)

    gravity = 0.0

    base_T0_new = [0.190, 0.000, -0.120, 3.141592653589793, 0.000, 0.000] # underarm #transformation of uv body frame to manipulator base
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
            0, 0, 0, 0, 0, 0 )