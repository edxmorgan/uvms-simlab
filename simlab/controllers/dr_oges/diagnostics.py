"""Transport existing DR evidence; no estimator or controller computation here."""
from dataclasses import asdict
import json

import numpy as np
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue


def encode(value):
    return json.dumps(value, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                      else v.item(), allow_nan=False, separators=(',', ':'))


class DRDiagnostics:
    def __init__(self, node, prefix):
        self.node = node
        self.prefix = prefix
        self.sequence = 0
        self.publisher = node.create_publisher(
            DiagnosticArray, f'/{prefix}/diagnostics/dr', 100)

    def emit(self, subsystem, event, observation, **values):
        # No serialization cost when diagnostics are not being recorded.
        if self.publisher.get_subscription_count() == 0:
            return
        self.sequence += 1
        values.update(schema_version=1, sequence=self.sequence, event=event,
                      controller='DR-Ours', subsystem=subsystem)
        values['observation'] = (None if observation is None else {
            key: getattr(observation, key)
            for key in ('generation', 'step', 'stamp', 'period')})
        msg = DiagnosticArray()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        status = DiagnosticStatus(name=f'{self.prefix}/DR-Ours/{subsystem}',
                                  hardware_id=self.prefix, message=event)
        status.values = [KeyValue(key=key, value=encode(value))
                         for key, value in values.items()]
        msg.status = [status]
        self.publisher.publish(msg)

    def projection(self, subsystem, observation, result, nominal, dt):
        self.emit(subsystem, 'projection', observation, result=asdict(result),
                  nominal_control=nominal, command_dt=dt)
