"""Immutable data contracts for future TRUST-MPS and RECAP-MPS layers.

No sensor acquisition, model inference, perturbation or retest policy lives here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias


class Domain(str, Enum):
    TS = 'TS'
    BS = 'BS'
    RS = 'RS'
    GI = 'GI'


DOMAIN_ORDER = (Domain.TS, Domain.BS, Domain.RS, Domain.GI)
DURATION_SECONDS = {Domain.TS: 10, Domain.BS: 10, Domain.RS: 10, Domain.GI: 180}
ALGORITHMS = {Domain.TS: 'SVM', Domain.BS: 'RF', Domain.RS: 'RF', Domain.GI: 'RF'}
FEATURE_ORDER = {
    Domain.TS: ('TS_tremor_rms_ms2',),
    Domain.BS: ('BS_tap_count', 'BS_tap_rate_per_min', 'BS_tap_interval_std_s'),
    Domain.RS: ('RS_mean_gyro_peak_rad_s', 'RS_movement_peaks'),
    Domain.GI: ('GI_step_candidates', 'GI_cadence_steps_min', 'GI_step_interval_std_s'),
}
RAW_COLUMNS = {
    Domain.TS: ('time_s', 'ax_ms2', 'ay_ms2', 'az_ms2'),
    Domain.BS: ('tap_time_s',),
    Domain.RS: ('time_s', 'gx_rad_s', 'gy_rad_s', 'gz_rad_s'),
    Domain.GI: ('time_s', 'ax_ms2', 'ay_ms2', 'az_ms2'),
}


class AssessmentMode(str, Enum):
    SYNTHETIC_DEMO = 'synthetic_demo'
    REAL_DEVICE = 'real_device'


class AcquisitionStatus(str, Enum):
    UNKNOWN = 'unknown'
    CONFIRMED_COMPLETE = 'confirmed_complete'
    KNOWN_FAILURE = 'known_failure'


class QualityStatus(str, Enum):
    VALID = 'valid'
    SUSPECT = 'suspect'
    INVALID = 'invalid'
    UNKNOWN = 'unknown'


class ReasonCode(str, Enum):
    MISSING_FILE = 'missing_file'
    INVALID_SCHEMA = 'invalid_schema'
    MISSING_VALUE = 'missing_value'
    NONFINITE_VALUE = 'nonfinite_value'
    TIMESTAMP_GRID = 'timestamp_grid'
    TAP_TIMESTAMP_ORDER = 'tap_timestamp_order'
    TAP_TIMESTAMP_RANGE = 'tap_timestamp_range'
    INCOMPLETE_RECORDING = 'incomplete_recording'
    DECLARED_ACQUISITION_FAILURE = 'declared_acquisition_failure'
    UNKNOWN_TAP_ACQUISITION = 'unknown_tap_acquisition'
    UNSUPPORTED_FEATURE = 'unsupported_feature'
    UNIT_MISMATCH = 'unit_mismatch'
    SESSION_MISMATCH = 'session_mismatch'


@dataclass(frozen=True)
class AcquisitionContext:
    """Optional verified metadata, external to the unchanged CSV schema."""

    status: AcquisitionStatus = AcquisitionStatus.UNKNOWN
    declared_unit: str | None = None
    # This is an asserted acquisition fact, never inferred from low movement.
    failure_detail: str | None = None
    session_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, AcquisitionStatus):
            raise ValueError('Acquisition status must use a defined enum value')
        if self.session_id is not None and not self.session_id.strip():
            raise ValueError('Provided session ID cannot be blank')


@dataclass(frozen=True)
class SensorSeries:
    domain: Domain
    time_s: tuple[float, ...]
    xyz: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if self.domain not in (Domain.TS, Domain.RS, Domain.GI):
            raise ValueError('SensorSeries requires TS, RS or GI')
        if len(self.time_s) != len(self.xyz):
            raise ValueError('Timestamp/sample lengths disagree')


@dataclass(frozen=True)
class TapSeries:
    tap_time_s: tuple[float, ...]


TaskInput: TypeAlias = SensorSeries | TapSeries


@dataclass(frozen=True)
class QualityFinding:
    code: ReasonCode
    detail: str
    row: int | None = None


@dataclass(frozen=True)
class QualityReport:
    domain: Domain
    status: QualityStatus
    findings: tuple[QualityFinding, ...] = ()
    samples_or_events: int = 0
    acquisition_status: AcquisitionStatus = AcquisitionStatus.UNKNOWN

    def __post_init__(self) -> None:
        if not isinstance(self.domain, Domain) or not isinstance(self.status, QualityStatus):
            raise ValueError('Quality domain/status must use defined enum values')
        if self.samples_or_events < 0:
            raise ValueError('Sample/event count cannot be negative')

    @property
    def permits_demo_scoring(self) -> bool:
        # A suspect status is a recorded caution, never silent imputation.
        return self.status is QualityStatus.VALID


@dataclass(frozen=True)
class ValidationResult:
    domain: Domain
    input: TaskInput | None
    quality: QualityReport

    def __post_init__(self) -> None:
        if self.quality.domain is not self.domain:
            raise ValueError('Quality report belongs to a different domain')
        if self.input is not None:
            actual = Domain.BS if isinstance(self.input, TapSeries) else self.input.domain
            if actual is not self.domain:
                raise ValueError('Validated input belongs to a different domain')


@dataclass(frozen=True)
class FeatureVector:
    domain: Domain
    names: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.names != FEATURE_ORDER[self.domain] or len(self.values) != len(self.names):
            raise ValueError('Frozen feature order/width mismatch')
        if any(not math.isfinite(v) for v in self.values):
            raise ValueError('Invalid or missing feature: do not use pipeline imputation to fabricate raw measurements')


@dataclass(frozen=True)
class FrozenPrediction:
    domain: Domain
    algorithm: str
    score: int
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if self.algorithm != ALGORITHMS[self.domain]:
            raise ValueError('Selected algorithm changed')
        if type(self.score) is not int or self.score not in range(5):
            raise ValueError('Demo score must be an integer 0–4')
        if len(self.checkpoint_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.checkpoint_sha256):
            raise ValueError('A SHA256 checkpoint identity is required')


@dataclass(frozen=True)
class AssessmentResult:
    mode: AssessmentMode
    features: tuple[FeatureVector, ...]
    quality: tuple[QualityReport, ...]
    predictions: tuple[FrozenPrediction, ...] | None
    mps: float | None
    status: str

    def __post_init__(self) -> None:
        if self.mode is AssessmentMode.REAL_DEVICE and (self.predictions is not None or self.mps is not None):
            raise ValueError('Real-device mode is features-only')
        if self.predictions is None and self.mps is not None:
            raise ValueError('MPS cannot exist without four valid scores')
        if self.predictions is not None:
            if tuple(p.domain for p in self.predictions) != DOMAIN_ORDER:
                raise ValueError('A complete ordered four-score bundle is required')
            expected = sum(p.score for p in self.predictions) / 16
            if self.mps != expected:
                raise ValueError('MPS must equal the unchanged four-score formula')


INTERVAL_KIND = 'modeled_perturbation_sensitivity'


@dataclass(frozen=True)
class PerturbationDiagnostics:
    domain: Domain
    scenario_family: str
    attempted: int
    valid: int
    invalid: int
    base_score: int | None
    score_histogram_0_to_4: tuple[int, int, int, int, int] | None
    mean_absolute_ordinal_drift: float | None
    nearest_observed_boundary_amplitude: float | None
    amplitude_unit: str | None
    # Descriptive data container only; no perturbation algorithm in Phase 3B.

    def __post_init__(self) -> None:
        if self.attempted < 0 or self.valid < 0 or self.invalid < 0 or self.valid + self.invalid != self.attempted:
            raise ValueError('Every perturbation attempt must be accounted for')
        if self.base_score is not None and (type(self.base_score) is not int or self.base_score not in range(5)):
            raise ValueError('Invalid base ordinal score')
        if self.score_histogram_0_to_4 is not None and (
                len(self.score_histogram_0_to_4) != 5 or
                any(type(n) is not int or n < 0 for n in self.score_histogram_0_to_4) or
                sum(self.score_histogram_0_to_4) != self.valid):
            raise ValueError('Perturbation score support must match valid probes')


@dataclass(frozen=True)
class MpsSensitivity:
    observed_joint_min: float | None
    observed_joint_max: float | None
    rectangular_outer_min: float | None
    rectangular_outer_max: float | None
    interval_kind: str = INTERVAL_KIND

    def __post_init__(self) -> None:
        if self.interval_kind != INTERVAL_KIND:
            raise ValueError('A modeled interval cannot be labeled as clinical confidence')
        for low, high in ((self.observed_joint_min, self.observed_joint_max),
                          (self.rectangular_outer_min, self.rectangular_outer_max)):
            if (low is None) != (high is None):
                raise ValueError('Sensitivity interval endpoints must be provided together')
            if low is not None and (not math.isfinite(low) or not math.isfinite(high) or
                                    not 0 <= low <= high <= 1):
                raise ValueError('Sensitivity interval is outside the normalized demo index')


@dataclass(frozen=True)
class TrustDiagnostics:
    perturbations: tuple[PerturbationDiagnostics, ...]
    mps_sensitivity: MpsSensitivity | None
    model_manifest_sha256: str | None
    assumptions_id: str | None


class ReassessmentAction(str, Enum):
    NONE = 'none'
    CLARIFY_ACQUISITION = 'clarify_acquisition'
    FIX_ACQUISITION_THEN_REPEAT = 'fix_acquisition_then_repeat'
    REPEAT_RECOMMENDED = 'repeat_recommended'
    NOT_ASSESSABLE = 'not_assessable'


@dataclass(frozen=True)
class ReassessmentAdvice:
    action: ReassessmentAction
    ranked_domains: tuple[Domain, ...]
    reason_codes: tuple[str, ...]
    additional_recording_seconds: int
    assumptions_id: str | None
    automatic_execution: bool = False

    def __post_init__(self) -> None:
        if self.automatic_execution:
            raise ValueError('RECAP may recommend only; it cannot control the device')
        if len(set(self.ranked_domains)) != len(self.ranked_domains):
            raise ValueError('Duplicate task recommendation')
        if self.additional_recording_seconds < 0:
            raise ValueError('Recording time cannot be negative')
        if self.action in (ReassessmentAction.NONE, ReassessmentAction.CLARIFY_ACQUISITION,
                           ReassessmentAction.NOT_ASSESSABLE) and self.additional_recording_seconds:
            raise ValueError('No repeat recording may be counted for this action')
        if self.action in (ReassessmentAction.FIX_ACQUISITION_THEN_REPEAT,
                           ReassessmentAction.REPEAT_RECOMMENDED):
            if not self.ranked_domains or self.additional_recording_seconds != DURATION_SECONDS[self.ranked_domains[0]]:
                raise ValueError('First recommended task must use its fixed protocol duration')


class TrustAnalyzer(Protocol):
    """Phase 3C hook; no implementation or benchmark in Phase 3B."""

    def analyze(self, assessment: AssessmentResult, inputs: tuple[ValidationResult, ...], *,
                assumptions_id: str, seed: int) -> TrustDiagnostics: ...


class RecapPolicy(Protocol):
    """Phase 3C recommendation hook; cannot trigger hardware."""

    def recommend(self, assessment: AssessmentResult, diagnostics: TrustDiagnostics | None,
                  *, time_budget_seconds: int) -> ReassessmentAdvice: ...
