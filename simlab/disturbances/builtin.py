"""Built-in stateless unit waveforms."""
import math
from functools import lru_cache
import numpy as np
from .base import DisturbanceWaveform


class Gaussian(DisturbanceWaveform):
    registry_name = 'gaussian'
    plot_style = 'steps-post'
    amplitude_label = 'σ'

    @classmethod
    def timing_label(cls, options):
        return f'20 Hz samples; {options.period*options.cycles:g} s; seed {options.seed}'

    @classmethod
    def metadata(cls):
        return dict(sample_rate_hz=20., numpy_version=np.__version__,
                    generator='PCG64, SeedSequence(seed,dof_index,sample_index)')

    supports_phase = False
    nonnegative_amplitude = True

    @staticmethod
    def sample(c):
        elapsed = c.elapsed
        seed = c.seed
        channel_index = c.channel_index
        sample_index = math.floor(elapsed * 20.0)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, channel_index, sample_index])))
        return float(rng.standard_normal())


@lru_cache(maxsize=8192)
def _correlated_sample(seed, channel, tau, index):
    """Exact stationary OU samples on a 20 Hz grid, independent of call order.

    Powers of two form a Markov chain of anchors. Interior samples use the
    conditional Gaussian midpoint (OU bridge). This provides random access
    without advancing through missed samples or depending on timer jitter.
    The bounded cache only accelerates evaluation; eviction changes no values.
    """
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [seed, channel, index, 0x4F55])))
    z = float(rng.standard_normal())
    if index == 0:
        return z  # Stationary N(0,1), not a zero-initialized warmup.
    if index & (index-1) == 0:
        previous = index//2
        dt = (index-previous)*.05
        rho = math.exp(-dt/tau)
        return (rho*_correlated_sample(seed, channel, tau, previous)
                + math.sqrt(-math.expm1(-2.*dt/tau))*z)
    left = 1 << (index.bit_length()-1)
    right = left*2
    while (left+right)//2 != index:
        middle = (left+right)//2
        if index < middle:
            right = middle
        else:
            left = middle
    half_dt = (right-left)*.025
    rho = math.exp(-half_dt/tau)
    mean = rho/(1.+rho*rho)*(
        _correlated_sample(seed, channel, tau, left)
        + _correlated_sample(seed, channel, tau, right))
    return mean + math.sqrt(math.tanh(half_dt/tau))*z


class CorrelatedGaussian(Gaussian):
    registry_name = 'correlated_gaussian'

    @classmethod
    def timing_label(cls, options):
        return (f'τ={options.correlation_time:g} s; '
                f'{super().timing_label(options)}')

    @classmethod
    def metadata(cls):
        return dict(sample_rate_hz=20., numpy_version=np.__version__,
                    model='stationary Ornstein-Uhlenbeck, exp(-abs(lag)/tau)',
                    generator='PCG64 keyed OU anchors/bridges; version 1')

    @staticmethod
    def sample(c):
        return _correlated_sample(c.seed, c.channel_index, c.correlation_time,
                                  math.floor(c.elapsed*20.))


class Sine(DisturbanceWaveform):
    registry_name = 'sine'

    @staticmethod
    def sample(c):
        elapsed = c.elapsed
        frequency = c.frequency
        phase_deg = c.phase_deg
        offset = math.radians(phase_deg % 360.0)
        phase = 2.0 * math.pi * frequency * elapsed + offset
        return math.sin(phase)


class Step(DisturbanceWaveform):
    registry_name = 'step'

    @classmethod
    def timing_label(cls, options):
        return f'{options.period*options.cycles:g} s hold'

    supports_phase = False

    @staticmethod
    def sample(c):
        return 1.0


class Square(DisturbanceWaveform):
    registry_name = 'square'

    @staticmethod
    def sample(c):
        elapsed = c.elapsed
        frequency = c.frequency
        phase_deg = c.phase_deg
        return 1.0 if (frequency * elapsed + phase_deg % 360.0 / 360.0) % 1.0 < 0.5 else -1.0


class Triangle(DisturbanceWaveform):
    registry_name = 'triangle'

    @staticmethod
    def sample(c):
        elapsed = c.elapsed
        frequency = c.frequency
        phase_deg = c.phase_deg
        offset = math.radians(phase_deg % 360.0)
        phase = 2.0 * math.pi * frequency * elapsed + offset
        return 2.0 / math.pi * math.asin(math.sin(phase))


class Chirp(DisturbanceWaveform):
    registry_name = 'chirp'
    uses_end_frequency = True

    @classmethod
    def normalize_end_frequency(cls, frequency, end_frequency):
        value = 2.*frequency if end_frequency is None else end_frequency
        if not math.isfinite(value) or not 0. < value < 5.:
            raise ValueError('Chirp end frequency must be finite, positive and below 5 Hz')
        return value

    @classmethod
    def timing_label(cls, options):
        return f'{options.frequency:g}→{options.end_frequency:g} Hz'


    @staticmethod
    def sample(c):
        elapsed = c.elapsed
        frequency = c.frequency
        duration = c.duration
        end_frequency = c.end_frequency
        phase_deg = c.phase_deg
        offset = math.radians(phase_deg % 360.0)
        slope = (end_frequency - frequency) / duration
        return math.sin(2.0 * math.pi * (frequency * elapsed + 0.5 * slope * elapsed * elapsed) + offset)


BUILTINS = (Sine, Step, Square, Triangle, Chirp, Gaussian, CorrelatedGaussian)
