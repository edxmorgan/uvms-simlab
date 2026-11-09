# Equivalent external wrench about the tool origin, expressed in header.frame_id.
# Not a contact detector. Assumes one tool wrench explains generalized disturbance.
std_msgs/Header header
bool valid
string status
geometry_msgs/Wrench wrench
# Row-major 6x6 [force xyz, torque xyz]. Status reports input covariance limitations.
float64[36] covariance
uint8 rank
uint8 observations
# NaN for excluded rows. Order FRD vehicle force/torque xyz, arm edcb.
float64[10] generalized_residual
float64 force_residual_norm
float64 torque_residual_norm
