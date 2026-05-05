std_msgs/Header header

string robot_prefix
bool active

# Instantaneous trajectory tracking terms.
float64 vehicle_cross_track_m
float64 vehicle_along_track_m
float64 vehicle_n_cross_track
float64 vehicle_n_along_track
float64 vehicle_n_position
float64 vehicle_n_attitude
float64 vehicle_n_linear_velocity
float64 vehicle_n_angular_velocity
float64 vehicle_n_linear_acceleration
float64 vehicle_n_angular_acceleration

# Instantaneous manipulator tracking terms.
float64 arm_n_position
float64 arm_n_velocity
float64 arm_n_acceleration

# Aggregate tracking/control terms for the current active behavior window.
float64 tracking_score
float64 tracking_score_rms
float64 normalized_control_effort
float64 effort_per_tracking_score
float64 energy_per_meter
float64 energy_per_second
# Seconds from behavior activation until tracking_score first reaches tolerance; -1 until reached.
float64 time_to_tolerance_sec
float64 peak_tracking_score
float64 sample_count
