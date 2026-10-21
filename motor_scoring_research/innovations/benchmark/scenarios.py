"""Preregistered in-memory scenario and policy-blind potential repeat bank.

The archived training waveform is a noisy nominal anchor, not a noise-free
human motor state. No code here loads a checkpoint or executes on import.
"""
from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass
from typing import Callable, Mapping

from ..contracts import (AcquisitionContext, AcquisitionStatus, DOMAIN_ORDER,
                         Domain, SensorSeries, TapSeries, TaskInput)
from ..perturbations import InvalidProbe, Probe, apply_probe

SLOTS = ('S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6')
FAULTS = ('timestamp_gap', 'incomplete_capture', 'invalid_unit',
          'declared_failure', 'missing_tap_event', 'duplicated_tap_event')


def derived_seed(master: int, parent_id: str, slot: str, role: str,
                 domain: Domain | str, attempt: int = 0) -> int:
    value = f'{master}|{parent_id}|{slot}|{role}|{Domain(domain).value}|{attempt}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'big')


@dataclass(frozen=True)
class Observation:
    parent_id: str
    split: str
    slot: str
    role: str
    raw: tuple[TaskInput, TaskInput, TaskInput, TaskInput]
    contexts: tuple[AcquisitionContext, AcquisitionContext, AcquisitionContext, AcquisitionContext]
    fault_truth: str | None = None  # evaluator-only; never passed to policies
    eligible: bool = True
    exclusion_reason: str | None = None
    changed_state: bool = False


def _contexts(parent_id: str) -> tuple[AcquisitionContext, ...]:
    return tuple(AcquisitionContext(AcquisitionStatus.CONFIRMED_COMPLETE,
                 's' if d is Domain.BS else 'rad/s' if d is Domain.RS else 'm/s^2',
                 session_id=parent_id) for d in DOMAIN_ORDER)


def _ordinary(raw: TaskInput, domain: Domain, seed: int) -> TaskInput:
    if domain is Domain.BS:
        return apply_probe(raw, Probe('tap_jitter', .001, 's', seed))
    return apply_probe(raw, Probe('noise', .005 if domain is Domain.RS else .01,
                                  'rad/s' if domain is Domain.RS else 'm/s^2', seed))


def _fault(raw: list[TaskInput], contexts: list[AcquisitionContext],
           family: str, domain: Domain, severity: int, *, persistent: bool = False) -> tuple[bool, str | None]:
    index = DOMAIN_ORDER.index(domain)
    item = raw[index]
    if family == 'timestamp_gap':
        if not isinstance(item, SensorSeries):
            raise ValueError('Timestamp gap requires IMU')
        time = list(item.time_s)
        time[len(time)//2] += (.005, .01)[severity]
        raw[index] = SensorSeries(domain, tuple(time), item.xyz)
    elif family == 'incomplete_capture':
        if not isinstance(item, SensorSeries):
            raise ValueError('Truncation requires IMU')
        n = 1000 if domain is Domain.GI else 100
        raw[index] = SensorSeries(domain, item.time_s[:-n], item.xyz[:-n])
    elif family == 'invalid_unit':
        contexts[index] = AcquisitionContext(AcquisitionStatus.CONFIRMED_COMPLETE,
                              'unverified_unit', session_id=contexts[index].session_id)
    elif family == 'declared_failure':
        contexts[index] = AcquisitionContext(AcquisitionStatus.KNOWN_FAILURE,
                             contexts[index].declared_unit, 'simulated acquisition failure',
                             contexts[index].session_id)
    elif family == 'missing_tap_event':
        if not isinstance(item, TapSeries) or len(item.tap_time_s) < 3:
            return False, 'tap_ledger_fewer_than_three_events'
        taps = list(item.tap_time_s)
        del taps[len(taps)//2]
        raw[index] = TapSeries(tuple(taps))
    elif family == 'duplicated_tap_event':
        if not isinstance(item, TapSeries) or len(item.tap_time_s) < 3:
            return False, 'tap_ledger_fewer_than_three_events'
        taps = list(item.tap_time_s)
        gaps = [(b-a, i) for i, (a, b) in enumerate(zip(taps, taps[1:])) if b-a > .002]
        if not gaps:
            return False, 'no_admissible_tap_gap'
        # The inserted event is a distinct spurious detection, not an equal-time CSV row.
        gap, position = gaps[len(gaps)//2]
        taps.insert(position + 1, taps[position] + gap/2)
        raw[index] = TapSeries(tuple(taps))
    else:
        raise ValueError(f'Unregistered fault: {family}')
    return True, None


def _s1(raw: list[TaskInput], rank: int, seed: Callable[[Domain], int],
        spec: dict) -> str:
    assumptions = spec['disturbance_assumptions']
    severity = rank//4 % 2
    family = rank % 4
    if family == 0:
        raw[0] = apply_probe(raw[0], Probe('offset', assumptions['TS_accel_offset_ms2'][severity],
                                           'm/s^2', seed(Domain.TS)))
        return 'TS_accel_offset'
    if family == 1:
        raw[2] = apply_probe(raw[2], Probe('offset', assumptions['RS_gyro_offset_rad_s'][severity],
                                           'rad/s', seed(Domain.RS)))
        return 'RS_gyro_offset'
    if family == 2:
        raw[3] = apply_probe(raw[3], Probe('noise', assumptions['GI_accel_noise_axis_std_ms2'][severity],
                                           'm/s^2', seed(Domain.GI)))
        return 'GI_accel_noise'
    gain = assumptions['TS_GI_shared_gain_fraction'][severity]
    # Shared sign across both accelerometers; independent per-domain seed is
    # otherwise unnecessary for this deterministic gain operator.
    common_seed = seed(Domain.TS)
    if seed(Domain.GI) % 2 != common_seed % 2:
        common_seed += 1
    for index in (0, 3):
        raw[index] = apply_probe(raw[index], Probe('gain', gain, 'fraction', common_seed))
    return 'TS_GI_shared_gain'


def _s6(raw: list[TaskInput], rank: int, parameters: Mapping | None, config: Mapping | None,
        synthesize: Callable | None, seed: int, spec: dict) -> tuple[str, bool]:
    if parameters is None or config is None or synthesize is None:
        raise ValueError('S6 requires archived parameters, frozen V3 config and verified original synthesizer')
    domain = DOMAIN_ORDER[rank % 4]
    key, bound_key = {Domain.TS: ('ts_amp', 'ts_amplitude_ms2'),
                      Domain.BS: ('bs_hz', 'bs_rate_hz'),
                      Domain.RS: ('rs_amp', 'rs_peak_rad_s'),
                      Domain.GI: ('gi_hz', 'gi_step_frequency_hz')}[domain]
    factor = spec['disturbance_assumptions']['S6_changed_state_factors'][f'{domain.value}_{key}'][rank//4 % 2]
    p = copy.deepcopy(parameters)
    lo, hi = config[bound_key]
    old = p[key]
    p[key] = max(lo, min(hi, old * factor))
    p['signal_seed'] = seed
    signals = synthesize(p, config, config['durations_s'])
    for i, d in enumerate(DOMAIN_ORDER):
        values = signals[d.value]
        if d is Domain.BS:
            raw[i] = TapSeries(tuple(float(v) for v in values))
        else:
            raw[i] = SensorSeries(d, tuple(i/100 for i in range(len(values))),
                                  tuple(tuple(float(v) for v in xyz) for xyz in values))
    return f'{domain.value}_{key}_factor_{factor}', p[key] == old


def scenario_pair(parent_id: str, split: str, rank: int, slot: str,
                  anchor: tuple[TaskInput, ...], spec: dict, *,
                  parameters: Mapping | None = None, v3_config: Mapping | None = None,
                  synthesize: Callable | None = None) -> tuple[Observation, dict[Domain, Observation]]:
    """Build one initial and four potential repeats before any policy decision."""
    if split not in ('development', 'audit') or slot not in SLOTS or len(anchor) != 4 or rank < 0:
        raise ValueError('Invalid parent, scenario or complete recording')
    if tuple(Domain.BS if isinstance(x, TapSeries) else x.domain for x in anchor) != DOMAIN_ORDER:
        raise ValueError('Four genuine ordered task inputs required')
    master = spec['master_seed']
    def seed(role: str, d: Domain, attempt: int = 0) -> int:
        return derived_seed(master, parent_id, slot, role, d, attempt)
    raw, contexts = list(anchor), list(_contexts(parent_id))
    family = None
    eligible, reason = True, None
    changed = slot == 'S6'
    if slot == 'S1':
        family = _s1(raw, rank, lambda d: seed('initial', d), spec)
    elif slot == 'S2':
        family = 'BS_timestamp_jitter'
        try:
            raw[1] = apply_probe(raw[1], Probe('tap_jitter', spec['disturbance_assumptions']['BS_jitter_absolute_bound_s'][rank%2], 's', seed('initial', Domain.BS)))
        except InvalidProbe as exc:
            eligible, reason = False, str(exc)
    elif slot in ('S3', 'S4'):
        family = FAULTS[rank % 6]
        domain = Domain.BS if rank%6 >= 4 else (Domain.TS, Domain.RS, Domain.GI)[rank//6 % 3]
        eligible, reason = _fault(raw, contexts, family, domain, rank//6 % 2)
    elif slot == 'S6':
        family, no_change = _s6(raw, rank, parameters, v3_config, synthesize,
                                seed('motor_control', DOMAIN_ORDER[rank%4]), spec)
        if no_change:
            reason = 'changed_parameter_at_bound_no_change'
    initial = Observation(parent_id, split, slot, 'initial', tuple(raw), tuple(contexts),
                          family, eligible, reason, changed)
    bank: dict[Domain, Observation] = {}
    for target in DOMAIN_ORDER:
        candidate = list(raw)
        repeat_context = list(contexts)
        repeat_family = family
        # A potential repeat is independently drawn from the nominal anchor,
        # except S4 persistent faults and S6 changed-state trajectory.
        if slot in ('S0', 'S1', 'S2', 'S3', 'S4'):
            try:
                candidate[DOMAIN_ORDER.index(target)] = _ordinary(anchor[DOMAIN_ORDER.index(target)], target, seed('repeat', target))
            except InvalidProbe as exc:
                bank[target] = Observation(parent_id, split, slot, 'repeat', tuple(candidate), tuple(repeat_context),
                                           family, False, str(exc), changed)
                continue
            if slot == 'S1':
                nuisance = list(anchor)
                _s1(nuisance, rank, lambda d: seed('repeat', d, 1), spec)
                candidate[DOMAIN_ORDER.index(target)] = nuisance[DOMAIN_ORDER.index(target)]
            elif slot == 'S2' and target is Domain.BS:
                try:
                    candidate[1] = apply_probe(candidate[1], Probe('tap_jitter', spec['disturbance_assumptions']['BS_jitter_absolute_bound_s'][rank%2], 's', seed('repeat', Domain.BS, 1)))
                except InvalidProbe as exc:
                    bank[target] = Observation(parent_id, split, slot, 'repeat', tuple(candidate), tuple(repeat_context), family, False, str(exc), changed)
                    continue
            elif slot == 'S3':
                # Only a corrected, observable structural fault is eligible
                # for repair. Latent tap faults have no policy-visible fix log.
                if target is (Domain.BS if rank%6 >= 4 else (Domain.TS, Domain.RS, Domain.GI)[rank//6 % 3]) and rank%6 < 4:
                    repeat_context[DOMAIN_ORDER.index(target)] = _contexts(parent_id)[DOMAIN_ORDER.index(target)]
            elif slot == 'S4':
                fault_domain = Domain.BS if rank%6 >= 4 else (Domain.TS, Domain.RS, Domain.GI)[rank//6 % 3]
                if target is fault_domain:
                    candidate[DOMAIN_ORDER.index(target)] = anchor[DOMAIN_ORDER.index(target)]
                    repeat_context[DOMAIN_ORDER.index(target)] = _contexts(parent_id)[DOMAIN_ORDER.index(target)]
                    eligible, repeat_reason = _fault(candidate, repeat_context, family, fault_domain, rank//6 % 2, persistent=True)
                    if not eligible:
                        bank[target] = Observation(parent_id, split, slot, 'repeat', tuple(candidate), tuple(repeat_context), family, False, repeat_reason, changed)
                        continue
        if slot == 'S5':
            try:
                candidate[DOMAIN_ORDER.index(target)] = _ordinary(anchor[DOMAIN_ORDER.index(target)], target, seed('repeat', target))
            except InvalidProbe as exc:
                bank[target] = Observation(parent_id, split, slot, 'repeat', tuple(candidate), tuple(repeat_context),
                                           family, False, str(exc), changed)
                continue
            fault_domain = DOMAIN_ORDER[rank%4]
            fault = ('timestamp_gap', 'missing_tap_event', 'declared_failure', 'incomplete_capture')[rank%4]
            if target is fault_domain:
                ok, why = _fault(candidate, repeat_context, fault, target, 1)
                if not ok:
                    bank[target] = Observation(parent_id, split, slot, 'repeat', tuple(candidate), tuple(repeat_context), fault, False, why, changed)
                    continue
                repeat_family = fault
        if slot == 'S6':
            repeat_state = list(anchor)
            _s6(repeat_state, rank, parameters, v3_config, synthesize,
                seed('motor_control', DOMAIN_ORDER[rank%4], 1), spec)
            candidate[DOMAIN_ORDER.index(target)] = repeat_state[DOMAIN_ORDER.index(target)]
        bank[target] = Observation(parent_id, split, slot, 'repeat', tuple(candidate), tuple(repeat_context),
                                   repeat_family, True, None, changed)
    return initial, bank
