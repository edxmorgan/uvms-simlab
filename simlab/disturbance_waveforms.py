"""Waveform definitions and validation shared by the action server and CLI."""
import argparse
import math

from simlab.disturbances import waveform_class, waveform_names
from simlab.disturbances.base import WaveformContext
from types import SimpleNamespace

# Whole-body injection order: Fx,Fy,Fz,Tx,Ty,Tz,tau_e,tau_d,tau_c,tau_b.
DOFS = {name: (i, unit, key) for i, (name, unit, key) in enumerate([
    ('surge', 'N', 'force.x'), ('sway', 'N', 'force.y'), ('heave', 'N', 'force.z'),
    ('roll', 'Nm', 'torque.x'), ('pitch', 'Nm', 'torque.y'), ('yaw', 'Nm', 'torque.z'),
    ('arm_e', 'Nm', 'torque.e'), ('arm_d', 'Nm', 'torque.d'),
    ('arm_c', 'Nm', 'torque.c'), ('arm_b', 'Nm', 'torque.b')])}


TASK_DOFS = {name: (i, 'N' if i < 3 else 'Nm', ('force.' if i < 3 else 'torque.')+'xyz'[i%3])
             for i, name in enumerate(('fx', 'fy', 'fz', 'tx', 'ty', 'tz'))}


def display_channels(options):
    """Task view uses all six tool-axis equivalent wrench components."""
    if options.space == 'generalized':
        return options.channels
    prototype = options.channels[0]
    return [SimpleNamespace(**dict(vars(prototype), dof=name, index=i, unit=unit,
                                  key=key, amplitude=0.))
            for name, (i, unit, key) in TASK_DOFS.items()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot-prefix', default='robot_1_', help='Simulated robot prefix')
    p.add_argument('--space', choices=['generalized', 'task'], default='generalized')
    p.add_argument('--wrench-frame', choices=['tool', 'body', 'world'], default='tool',
                   help='Task wrench axes; body means URDF FLU. Torque is about the IK tool origin.')
    p.add_argument('--dof', choices=list(DOFS)+list(TASK_DOFS), nargs='+', default=None, help='One or more simultaneous DOFs')
    p.add_argument('--amplitude', type=float, nargs='+', help='One amplitude per DOF (N or Nm); Gaussian: standard deviation')
    p.add_argument('--waveform', '--function', choices=waveform_names(),
                   nargs='+', default=['sine'], help='One waveform for all DOFs, or one per DOF')
    p.add_argument('--phase', type=float, nargs='+', default=[0.],
                   help='Phase in degrees: positive leads, negative lags; one or per DOF')
    p.add_argument('--delay', type=float, nargs='+', default=[0.],
                   help='Start delay after baseline in seconds; one or per DOF')
    p.add_argument('--end-frequency', type=float, help='Chirp final frequency in Hz (default: twice the starting frequency)')
    timing = p.add_mutually_exclusive_group()
    timing.add_argument('--period', type=float, help='Period in seconds (default: 20)')
    timing.add_argument('--frequency', type=float, help='Positive frequency in Hz')
    p.add_argument('--cycles', type=int, default=3)
    p.add_argument('--correlation-time', type=float, default=2.,
                   help='Correlated Gaussian e-folding time in seconds (default: 2)')
    p.add_argument('--seed', type=int, default=0, help='Gaussian noise seed (uint32, default: 0)')
    p.add_argument('--ready-fd', type=int, help=argparse.SUPPRESS)
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--close-on-finish', action='store_true', help='Close live plot after injection')
    a = p.parse_args(argv)
    try:
        return validate_options(a)
    except ValueError as error:
        p.error(str(error))


def validate_options(a):
    """Validate/normalize waveform options independently of CLI or ROS transport."""
    def invalid(message):
        raise ValueError(message)

    a.space = getattr(a, 'space', 'generalized')
    a.wrench_frame = getattr(a, 'wrench_frame', 'tool')
    if a.space not in ('generalized', 'task') or a.wrench_frame not in ('tool', 'body', 'world'):
        invalid('Unknown disturbance space or wrench frame')
    dofs = TASK_DOFS if a.space == 'task' else DOFS
    if a.dof is None:
        a.dof = ['fx' if a.space == 'task' else 'surge']
    if not a.dof or any(name not in dofs for name in a.dof):
        invalid('Select at least one supported DOF')
    if not a.waveform or any(kind not in waveform_names() for kind in a.waveform):
        invalid('Select supported waveforms: '+', '.join(waveform_names()))
    if len(set(a.dof)) != len(a.dof):
        invalid('Select each DOF only once')
    if a.amplitude is not None and len(a.amplitude) != len(a.dof):
        invalid('Provide exactly one amplitude per selected DOF, in the same order')
    if len(a.waveform) not in (1, len(a.dof)):
        invalid('Provide one waveform for all DOFs, or one waveform per DOF')
    for option in ('phase', 'delay'):
        values = getattr(a, option)
        if len(values) not in (1, len(a.dof)) or not all(math.isfinite(v) for v in values):
            invalid(f'--{option} needs one finite value or one per DOF')
    if any(v < 0. for v in a.delay):
        invalid('--delay must be nonnegative')
    if not isinstance(a.seed, int) or not 0 <= a.seed <= 2**32-1:
        invalid('Seed must be an integer between 0 and 4294967295')
    a.correlation_time = getattr(a, 'correlation_time', 2.)
    if not math.isfinite(a.correlation_time) or a.correlation_time <= 0.:
        invalid('Correlation time must be finite and positive, in seconds')
    a.channels = []
    for j, name in enumerate(a.dof):
        index, unit, key = dofs[name]
        amplitude = (a.amplitude[j] if a.amplitude is not None else
                     (3. if index < 3 else (.1 if index < 6 else .05)))
        if not math.isfinite(amplitude):
            invalid('Amplitudes must be finite')
        waveform = a.waveform[0] if len(a.waveform) == 1 else a.waveform[j]
        phase = a.phase[0] if len(a.phase) == 1 else a.phase[j]
        delay = a.delay[0] if len(a.delay) == 1 else a.delay[j]
        waveform_class(waveform).validate_channel(amplitude, phase)
        a.channels.append(SimpleNamespace(dof=name,index=index,unit=unit,key=key,
                                          amplitude=amplitude,waveform=waveform,phase_deg=phase,delay_s=delay))
    if a.frequency is not None:
        if not math.isfinite(a.frequency) or a.frequency <= 0:
            invalid('Frequency must be finite and positive, in Hz')
        a.period = 1. / a.frequency
    elif a.period is None:
        a.period = 20.
    if not math.isfinite(a.period) or a.period <= 0 or not 1 <= a.cycles <= 2**32-1:
        invalid('Use a positive finite period and a cycle count between 1 and 4294967295')
    a.frequency = 1. / a.period
    if not math.isfinite(a.frequency) or not math.isfinite(a.period*a.cycles):
        invalid('Frequency and duration must be finite')
    if a.frequency >= 5.:
        invalid('Use a frequency below 5 Hz (20 Hz command stream; no discontinuity smoothing)')
    end_users = [waveform_class(name) for name in a.waveform
                 if waveform_class(name).uses_end_frequency]
    if a.end_frequency is not None and not end_users:
        invalid('--end-frequency requires a waveform supporting end frequency')
    for cls in end_users:
        a.end_frequency = cls.normalize_end_frequency(a.frequency, a.end_frequency)
    a.max_delay = max(c.delay_s for c in a.channels)
    if not math.isfinite(5.+a.period*a.cycles+a.max_delay):
        invalid('Total run duration must be finite')
    return a


def waveform_value(kind, elapsed, frequency, duration, end_frequency=None, phase_deg=0., *, seed=0, channel_index=0, correlation_time=2.):
    """Unit waveform, with zero outside the active interval; chirp phase is integrated."""
    if elapsed < 0. or elapsed >= duration:
        return 0.
    value = float(waveform_class(kind).sample(WaveformContext(
        elapsed, frequency, duration, end_frequency, phase_deg, seed, channel_index, correlation_time)))
    if not math.isfinite(value):
        raise ValueError(f'Waveform {kind} returned a nonfinite value')
    return value


def load_vector(channels, elapsed, frequency, duration, end_frequency=None, *, seed=0, correlation_time=2.):
    values = [0.] * 10
    for channel in channels:
        values[channel.index] = channel.amplitude * waveform_value(
            channel.waveform, elapsed-channel.delay_s, frequency, duration, end_frequency, channel.phase_deg,
            seed=seed, channel_index=channel.index, correlation_time=correlation_time)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Waveform amplitude scaling produced a nonfinite load")
    return values
