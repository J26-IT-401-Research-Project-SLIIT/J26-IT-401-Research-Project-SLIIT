"""Finite, in-memory measurement-nuisance probes; no missing-event repair."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .contracts import Domain, SensorSeries, TapSeries, TaskInput


@dataclass(frozen=True)
class Probe:
    family: str
    amplitude: float
    unit: str
    seed: int
    # Quantization requires a verified device conversion step supplied by caller.


class InvalidProbe(ValueError):
    """A proposed measurement perturbation cannot preserve the raw protocol."""


def apply_probe(raw: TaskInput, probe: Probe, *, quantization_step: float | None = None) -> TaskInput:
    """Return a new observation; preserve event count, sample count and clock schema.

    The nuisance magnitudes are engineering hypotheses, not MPU6050 tolerances.
    A failed BS jitter probe is rejected rather than clipped or sorted. GI timing
    is intentionally unsupported until a continuous reference trajectory exists.
    """
    if not math.isfinite(probe.amplitude) or probe.amplitude < 0:
        raise InvalidProbe('invalid_amplitude')
    rng = random.Random(probe.seed)
    if isinstance(raw, TapSeries):
        if probe.family != 'tap_jitter':
            raise InvalidProbe('unsupported_family')
        times = tuple(t + rng.uniform(-probe.amplitude, probe.amplitude) for t in raw.tap_time_s)
        if any(not 0 <= t < 10 for t in times):
            raise InvalidProbe('tap_out_of_window')
        if any(b <= a for a, b in zip(times, times[1:])):
            raise InvalidProbe('tap_order_collision')
        return TapSeries(times)
    if probe.family == 'quantization':
        if quantization_step is None or not math.isfinite(quantization_step) or quantization_step <= 0:
            raise InvalidProbe('unverified_quantization_step')
        if probe.amplitude != quantization_step:
            raise InvalidProbe('quantization_step_mismatch')
    if probe.family not in ('offset', 'noise', 'gain', 'quantization'):
        raise InvalidProbe('unsupported_family')
    transformed = []
    for xyz in raw.xyz:
        if probe.family == 'offset':
            point = tuple(v + probe.amplitude for v in xyz)
        elif probe.family == 'noise':
            point = tuple(v + rng.gauss(0, probe.amplitude) for v in xyz)
        elif probe.family == 'gain':
            # Signed gain is represented by the seed parity; no change to motor state.
            sign = -1 if probe.seed % 2 else 1
            point = tuple(v * (1 + sign * probe.amplitude) for v in xyz)
        else:
            point = tuple(round(v / quantization_step) * quantization_step for v in xyz)
        if any(not math.isfinite(v) for v in point):
            raise InvalidProbe('nonfinite_transformed_value')
        transformed.append(point)
    return SensorSeries(raw.domain, raw.time_s, tuple(transformed))


def validate_perturbed(raw: TaskInput, domain: Domain) -> None:
    """Check generated raw values without silently repairing any observation."""
    if domain is Domain.BS:
        if not isinstance(raw, TapSeries) or len(raw.tap_time_s) < 2:
            raise InvalidProbe('tap_count_unsupported')
        if any(not math.isfinite(t) or not 0 <= t < 10 for t in raw.tap_time_s):
            raise InvalidProbe('tap_out_of_window')
        if any(b <= a for a, b in zip(raw.tap_time_s, raw.tap_time_s[1:])):
            raise InvalidProbe('tap_order_collision')
        return
    expected = 18000 if domain is Domain.GI else 1000
    if not isinstance(raw, SensorSeries) or raw.domain is not domain or len(raw.time_s) != expected or len(raw.xyz) != expected:
        raise InvalidProbe('recording_length_or_domain')
    if any(not math.isfinite(t) or abs(t - i / 100) > 1e-9 for i, t in enumerate(raw.time_s)):
        raise InvalidProbe('timestamp_grid')
    if any(not math.isfinite(v) for p in raw.xyz for v in p):
        raise InvalidProbe('nonfinite_transformed_value')
