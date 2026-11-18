std_msgs/Header header

string robot_prefix
bool active
# Version 2: geometric/frame-consistent errors and elapsed-time statistics.
uint32 metric_version
# false means tracking values are unavailable, not zero error.
bool valid
string status
float64 elapsed_sec
float64 valid_duration_sec
float64 coverage_fraction
# Age since local receipt of feedback; not individual sensor acquisition age.
float64 feedback_age_sec
bool effort_valid
bool energy_valid
bool tool_valid

# Instantaneous trajectory tracking terms.
float64 vehicle_cross_track_m
float64 vehicle_along_track_m
float64 vehicle_n_cross_track
float64 vehicle_n_along_track
float64 vehicle_n_position
# Principal SO(3) rotation distance divided by pi, in [0, 1].
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
# Diagnostic effort/error ratio only; not an efficiency or ranking score.
float64 effort_per_tracking_score
float64 energy_per_meter
float64 energy_per_second
# Seconds from behavior activation until tracking_score first reaches tolerance; -1 until reached.
float64 time_to_tolerance_sec
float64 peak_tracking_score
float64 sample_count

# Sustained composite-score band occupancy, NOT physical endpoint settling.
# Confirmation resets on departure, invalid data, or a discontinuity.
bool settled
float64 settling_time_sec
float64 within_tolerance_duration_sec
float64 settling_dwell_sec

# Physical instantaneous errors. Unavailable scalars/arrays contain NaN.
float64 vehicle_position_error_m
float64 vehicle_attitude_error_rad
float64 vehicle_linear_velocity_error_mps
float64 vehicle_angular_velocity_error_radps
float64 vehicle_linear_acceleration_error_mps2
float64 vehicle_angular_acceleration_error_radps2
# Actuator order e,d,c,b; direct actuator-coordinate differences.
float64[4] arm_position_error_rad
float64[4] arm_velocity_error_radps
float64[4] arm_acceleration_error_radps2
float64 arm_position_rms_rad
float64 arm_max_position_error_rad
# Desired tool reference must be supplied explicitly; inspect tool_valid.
float64 tool_position_error_m
float64 tool_attitude_error_rad
# Time-weighted RMS over intervals with valid tool endpoints.
float64 tool_position_error_m_rms
float64 tool_attitude_error_rad_rms
