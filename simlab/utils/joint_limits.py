"""Position endpoints from the robot description, including actuator travel limits."""
import math
import xml.etree.ElementTree as ET


def position_endpoints(description: str, joint_name: str, *, actuator_bounds: bool = False) -> tuple[float, float]:
    root = ET.fromstring(description)
    joint = next((j for j in root.findall('joint') if j.get('name') == joint_name), None)
    if joint is None or joint.find('limit') is None:
        raise ValueError(f'{joint_name}: missing URDF position limits')
    limit = joint.find('limit')
    lower, upper = float(limit.attrib['lower']), float(limit.attrib['upper'])
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError(f'{joint_name}: nonfinite URDF limits')
    for control_joint in root.findall('ros2_control/joint'):
        if control_joint.get('name') != joint_name:
            continue
        params = {p.get('name'): p.text for p in control_joint.findall('param')}
        if int(params.get('has_position_limits', '0')):
            control_lower = float(params['min_position'])
            control_upper = float(params['max_position'])
            if not (math.isfinite(control_lower) and math.isfinite(control_upper)):
                raise ValueError(f'{joint_name}: nonfinite actuator limits')
            if actuator_bounds:
                lower, upper = control_lower, control_upper
            else:
                lower = max(lower, control_lower)
                upper = min(upper, control_upper)
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError(f'{joint_name}: invalid or disjoint position limits')
    return lower, upper


def velocity_limit(description: str, joint_name: str) -> float:
    root = ET.fromstring(description)
    joint = next(j for j in root.findall('joint') if j.get('name') == joint_name)
    speed = float(joint.find('limit').attrib['velocity'])
    candidates = [speed]
    for control_joint in root.findall('ros2_control/joint'):
        if control_joint.get('name') == joint_name:
            params = {p.get('name'): p.text for p in control_joint.findall('param')}
            if 'max_velocity' in params:
                candidates.append(float(params['max_velocity']))
    if any(not math.isfinite(v) or v <= 0 for v in candidates):
        raise ValueError(f'{joint_name}: invalid velocity limit')
    return min(candidates)
