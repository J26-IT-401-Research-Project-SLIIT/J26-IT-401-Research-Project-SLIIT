"""Protocol validation of existing PP1 CSVs; never repairs or scores inputs."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Mapping

from .contracts import (
    AcquisitionContext, AcquisitionStatus, DURATION_SECONDS, Domain, QualityFinding,
    QualityReport, QualityStatus, RAW_COLUMNS, SensorSeries, TapSeries, ValidationResult,
)
from .contracts import ReasonCode as Reason

SAMPLE_HZ = 100
TIME_TOLERANCE_SECONDS = 1e-9  # Matches the frozen V3 adapter's absolute tolerance.
MAX_REPORTED_FINDINGS = 16
ACCEPTED_UNITS = {
    Domain.TS: frozenset({'m/s^2', 'm/s²'}),
    Domain.BS: frozenset({'s'}),
    Domain.RS: frozenset({'rad/s'}),
    Domain.GI: frozenset({'m/s^2', 'm/s²'}),
}


def _result(domain: Domain, status: QualityStatus, findings: list[QualityFinding],
            count: int, context: AcquisitionContext, input_data=None) -> ValidationResult:
    report = QualityReport(domain, status, tuple(findings[:MAX_REPORTED_FINDINGS]), count, context.status)
    # An ambiguous or invalid task is never forwarded as an inferable raw input.
    return ValidationResult(domain, input_data if status is QualityStatus.VALID else None, report)


def validate_task_csv(domain: Domain, csv_path: Path | str,
                      context: AcquisitionContext | None = None) -> ValidationResult:
    """Read existing CSV shape and timestamps without resampling or adding metadata.

    A missing acquisition context is unknown. Nonempty event lists are structurally
    assessable; empty/one-tap lists are not mapped through the frozen ML imputer.
    """
    domain = Domain(domain)
    context = context or AcquisitionContext()
    path = Path(csv_path)
    if not path.is_file():
        return _result(domain, QualityStatus.INVALID,
                       [QualityFinding(Reason.MISSING_FILE, 'Recording file is unavailable')], 0, context)
    try:
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.reader(stream, strict=True)
            header = tuple(next(reader))
            if header != RAW_COLUMNS[domain]:
                return _result(domain, QualityStatus.INVALID,
                               [QualityFinding(Reason.INVALID_SCHEMA,
                                               f'Expected {RAW_COLUMNS[domain]}; found {header}')], 0, context)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
        return _result(domain, QualityStatus.INVALID,
                       [QualityFinding(Reason.INVALID_SCHEMA, f'Unreadable CSV: {type(exc).__name__}')], 0, context)

    findings: list[QualityFinding] = []
    invalid = False
    if context.status is AcquisitionStatus.KNOWN_FAILURE:
        findings.append(QualityFinding(Reason.DECLARED_ACQUISITION_FAILURE,
                                       context.failure_detail or 'Acquisition reported a failure'))
        invalid = True
    if context.declared_unit is not None and context.declared_unit not in ACCEPTED_UNITS[domain]:
        findings.append(QualityFinding(Reason.UNIT_MISMATCH,
                                       f'Declared unit {context.declared_unit!r} does not match protocol'))
        invalid = True

    expected_width = len(RAW_COLUMNS[domain]); parsed: list[tuple[float, ...]] = []
    for row_number, row in enumerate(rows, start=2):
        if len(row) != expected_width:
            findings.append(QualityFinding(Reason.INVALID_SCHEMA,
                                           f'Expected {expected_width} columns; found {len(row)}', row_number))
            invalid = True
        elif any(not value.strip() for value in row):
            findings.append(QualityFinding(Reason.MISSING_VALUE, 'Blank numeric field', row_number))
            invalid = True
        else:
            try:
                numbers = tuple(float(value) for value in row)
            except ValueError:
                findings.append(QualityFinding(Reason.INVALID_SCHEMA, 'Non-numeric field', row_number))
                invalid = True
            else:
                if not all(math.isfinite(value) for value in numbers):
                    findings.append(QualityFinding(Reason.NONFINITE_VALUE, 'Nonfinite measurement', row_number))
                    invalid = True
                else:
                    parsed.append(numbers)
        if len(findings) >= MAX_REPORTED_FINDINGS:
            invalid = True
            break

    if domain is Domain.BS:
        if not invalid:
            taps = tuple(row[0] for row in parsed)
            for index, value in enumerate(taps):
                if not 0 <= value < 10:
                    findings.append(QualityFinding(Reason.TAP_TIMESTAMP_RANGE,
                                                   'Tap timestamp outside [0,10) seconds', index + 2))
                    invalid = True
                    break
                if index and value <= taps[index - 1]:
                    findings.append(QualityFinding(Reason.TAP_TIMESTAMP_ORDER,
                                                   'Tap timestamps must be strictly increasing', index + 2))
                    invalid = True
                    break
        else:
            taps = ()
        if invalid:
            return _result(domain, QualityStatus.INVALID, findings, len(rows), context)
        if not taps and context.status is AcquisitionStatus.UNKNOWN:
            return _result(domain, QualityStatus.UNKNOWN,
                           [QualityFinding(Reason.UNKNOWN_TAP_ACQUISITION,
                                           'Zero events with unconfirmed capture; no bradykinesia score')], 0, context)
        if len(taps) < 2:
            # Existing engine's interval standard deviation is None, and the
            # frozen ML feature schema requires it. Do not use the imputer here.
            return _result(domain, QualityStatus.SUSPECT,
                           [QualityFinding(Reason.UNSUPPORTED_FEATURE,
                                           'Fewer than two taps; interval feature undefined')], len(taps), context)
        return _result(domain, QualityStatus.VALID, [], len(taps), context, TapSeries(taps))

    expected_rows = DURATION_SECONDS[domain] * SAMPLE_HZ
    if len(rows) != expected_rows:
        findings.append(QualityFinding(Reason.INCOMPLETE_RECORDING,
                                       f'Expected {expected_rows} samples; found {len(rows)}'))
        invalid = True
    if not invalid:
        for index, row in enumerate(parsed):
            if abs(row[0] - index / SAMPLE_HZ) > TIME_TOLERANCE_SECONDS:
                findings.append(QualityFinding(Reason.TIMESTAMP_GRID,
                                               'Timestamp differs from fixed 100-Hz grid', index + 2))
                invalid = True
                break
    if invalid:
        return _result(domain, QualityStatus.INVALID, findings, len(rows), context)
    series = SensorSeries(domain, tuple(row[0] for row in parsed),
                          tuple((row[1], row[2], row[3]) for row in parsed))
    return _result(domain, QualityStatus.VALID, [], len(rows), context, series)


def validate_session(paths: Mapping[Domain, Path | str],
                     contexts: Mapping[Domain, AcquisitionContext] | None = None) -> tuple[ValidationResult, ...]:
    """Return one result per domain; absent tasks remain absent, never padded."""
    contexts = contexts or {}
    known_ids = {context.session_id for context in contexts.values() if context.session_id is not None}
    if len(known_ids) > 1:
        return tuple(_result(domain, QualityStatus.INVALID,
                             [QualityFinding(Reason.SESSION_MISMATCH,
                                             'Declared task session IDs disagree')], 0,
                             contexts.get(domain) or AcquisitionContext())
                     for domain in (Domain.TS, Domain.BS, Domain.RS, Domain.GI))
    results = []
    for domain in (Domain.TS, Domain.BS, Domain.RS, Domain.GI):
        if domain in paths:
            results.append(validate_task_csv(domain, paths[domain], contexts.get(domain)))
        else:
            context = contexts.get(domain) or AcquisitionContext()
            results.append(_result(domain, QualityStatus.INVALID,
                                   [QualityFinding(Reason.MISSING_FILE, 'Domain recording omitted')], 0, context))
    return tuple(results)
