import json
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from builtin_interfaces.msg import Time
from simlab.controller_configuration import ControllerTrace


class Controller:
    registry_name='PID'
    def __init__(self):self.gain=1.
    def arm_controller(self):pass
    def vehicle_controller(self):pass
    def configuration_snapshot(self):return {'controller':self.registry_name,'parameters':{'gain':self.gain}}


def test_context_distinguishes_selected_and_executing_and_records_gain_changes():
    node=Mock();trace=ControllerTrace(node,'robot_1_');c=Controller()
    robot=SimpleNamespace(prefix='robot_1_',controller_name='PID',control_space='joint_space',
        control_mode='planner',arm_controller_fn=c.arm_controller,vehicle_controller_fn=c.vehicle_controller)
    stamp=Time(sec=12)
    a=trace.context(robot,stamp)
    assert trace.context(robot,stamp)==a
    assert trace.context_pub.publish.call_count==1
    robot.controller_name='CmdReplay'
    b=trace.context(robot,stamp)
    assert b['revision']==a['revision']+1
    payload=json.loads(trace.context_pub.publish.call_args.args[0].data)
    assert payload['selected_controller']=='CmdReplay'
    assert payload['components']['arm']['controller']=='PID'
    c.gain=2.;trace.context(robot,stamp,force=True)
    payload=json.loads(trace.context_pub.publish.call_args.args[0].data)
    assert payload['components']['arm']['parameters']['gain']==2.


def test_kinematics_preserves_inputs_validity_and_context_reference():
    trace=ControllerTrace(Mock(),'robot_1_');robot=SimpleNamespace(prefix='robot_1_')
    stamp=Time(sec=5,nanosec=3)
    trace.kinematics(robot,stamp,{'q':np.arange(4),'feedback_age_sec':np.nan},
                     {'q':np.ones(4)}, {'valid':False,'tool_position_error_m':np.nan})
    payload=json.loads(trace.kinematics_pub.publish.call_args.args[0].data)
    assert payload['stamp_ns']==5000000003
    assert payload['current']['q']==[0,1,2,3]
    assert payload['metrics']['tool_position_error_m'] is None
    assert not payload['metrics']['valid']
    assert payload['context']['revision']==1
