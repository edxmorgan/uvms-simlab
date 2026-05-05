std_msgs/Header header

# World-frame reference in the stack's world convention.
geometry_msgs/Pose world_pose
geometry_msgs/Twist world_twist
geometry_msgs/Accel world_accel

# Vehicle reference in NED/body coordinates.
# ned_pose = [x, y, z, roll, pitch, yaw]
float64[6] ned_pose
# body_velocity = [u, v, w, p, q, r]
float64[6] body_velocity
# body_acceleration = [du, dv, dw, dp, dq, dr]
float64[6] body_acceleration

# Manipulator reference, ordered [axis_e, axis_d, axis_c, axis_b, grasper].
float64[5] arm_position
float64[5] arm_velocity
float64[5] arm_acceleration
