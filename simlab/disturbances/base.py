"""Stateless unit-waveform interface; timing, amplitude and transport stay in SimLab."""
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class WaveformContext:
    elapsed: float
    frequency: float
    duration: float
    end_frequency: float | None
    phase_deg: float
    seed: int
    channel_index: int
    correlation_time: float = 2.0


class DisturbanceWaveform(ABC):
    registry_name = ''
    supports_phase = True
    nonnegative_amplitude = False

    version = '1'
    uses_end_frequency = False
    plot_style = None
    amplitude_label = 'amplitude'

    @classmethod
    def validate_channel(cls, amplitude, phase):
        if not cls.supports_phase and phase != 0.:
            raise ValueError(f'{cls.registry_name} has no oscillation phase; use --delay instead')
        if cls.nonnegative_amplitude and amplitude < 0.:
            raise ValueError(f'{cls.registry_name} amplitude must be nonnegative')

    @classmethod
    def normalize_end_frequency(cls, frequency, end_frequency):
        return end_frequency

    @classmethod
    def timing_label(cls, options):
        return f'{options.frequency:g} Hz'

    @classmethod
    def plot_title(cls, channel, options):
        phase = f'; phase {channel.phase_deg:g}°' if cls.supports_phase else ''
        return (f'{channel.dof}: {cls.registry_name}, {cls.amplitude_label} '
                f'{channel.amplitude:g} {channel.unit}\n'
                f'{cls.timing_label(options)}{phase}; delay {channel.delay_s:g} s')

    @classmethod
    def metadata(cls):
        return {}

    @staticmethod
    @abstractmethod
    def sample(context: WaveformContext) -> float:
        """Return a finite unit value. Same context must produce the same value.

        Implementations must be fast, nonblocking, and free of ROS/file side effects.
        """
