from visualization_msgs.msg import Marker, InteractiveMarker,InteractiveMarkerControl
from geometry_msgs.msg import Pose


def makeBox(fixed, scale, marker_type, initial_pose):
    marker = Marker()
    marker.type = marker_type
    marker.pose = initial_pose
    marker.scale.x = scale * 0.25
    marker.scale.y = scale * 0.25
    marker.scale.z = scale * 0.25

    if fixed:
        marker.color.r = 1.0 
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
    else:
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 1.0
    return marker

def makeBoxControl(msg, fixed, interaction_mode, marker_type,
                    scale=1.0, show_6dof=False, initial_pose=Pose(), ignore_dof=[]):
    control = InteractiveMarkerControl()
    control.always_visible = True
    control.markers.append(makeBox(fixed, scale, marker_type, initial_pose))
    control.interaction_mode = interaction_mode
    msg.controls.append(control)

    if show_6dof:
        if 'roll' not in ignore_dof:
            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 1.0
            control.orientation.y = 0.0
            control.orientation.z = 0.0
            control.name = "roll"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            msg.controls.append(control)

        if 'surge' not in ignore_dof:
            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 1.0
            control.orientation.y = 0.0
            control.orientation.z = 0.0
            control.name = "surge"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            msg.controls.append(control)

        if 'yaw' not in ignore_dof:
            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 0.0
            control.orientation.y = 1.0
            control.orientation.z = 0.0
            control.name = "yaw"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            msg.controls.append(control)

        if 'heave' not in ignore_dof:
            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 0.0
            control.orientation.y = 1.0
            control.orientation.z = 0.0
            control.name = "heave"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            msg.controls.append(control)

        if 'pitch' not in ignore_dof:
            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 0.0
            control.orientation.y = 0.0
            control.orientation.z = 1.0
            control.name = "pitch"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            msg.controls.append(control)

        if 'sway' not in ignore_dof:
            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 0.0
            control.orientation.y = 0.0
            control.orientation.z = 1.0
            control.name = "sway"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            msg.controls.append(control)

    return control

def make_UVMS_Dof_Marker(name, description, frame_id, control_frame, fixed,
                        interaction_mode, initial_pose, scale,
                        arm_base_pose=None, show_6dof=False, ignore_dof=[]):
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = frame_id
    int_marker.pose = initial_pose
    int_marker.scale = scale
    int_marker.name = name
    int_marker.description = description
    marker_type = Marker.CUBE
    if 'task' in control_frame:
        marker_type = Marker.SPHERE
    makeBoxControl(int_marker, fixed, interaction_mode, marker_type,
                        int_marker.scale, show_6dof, Pose(), ignore_dof)
    if control_frame == 'uv':
        makeBoxControl(int_marker, True, InteractiveMarkerControl.NONE, Marker.CUBE, 0.2, False, arm_base_pose, ignore_dof)
    return int_marker

def make_menu_control():
    menu_control = InteractiveMarkerControl()
    menu_control.interaction_mode = InteractiveMarkerControl.MENU
    menu_control.name = "robots_control_menu"
    menu_control.description = "target"
    menu_control.always_visible = True
    return menu_control