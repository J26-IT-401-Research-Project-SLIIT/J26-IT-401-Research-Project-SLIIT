"""Assemble unchanged demo scoring or real-device feature-only outputs."""
from __future__ import annotations

from .contracts import (
    AssessmentMode, AssessmentResult, DOMAIN_ORDER, FeatureVector, FrozenPrediction,
    QualityStatus, ValidationResult,
)


def assemble_assessment(mode: AssessmentMode,
                        validations: tuple[ValidationResult, ...],
                        features: tuple[FeatureVector, ...] = (),
                        predictions: tuple[FrozenPrediction, ...] | None = None) -> AssessmentResult:
    """No extraction/inference here; refuses invented scores and mixed sessions.

    The caller must supply exactly the frozen engine's features and previously
    verified model predictions. This function cannot initiate an assessment.
    """
    mode = AssessmentMode(mode)
    if tuple(v.domain for v in validations) != DOMAIN_ORDER:
        raise ValueError('One validation result per task in TS/BS/RS/GI order is required')
    if len({f.domain for f in features}) != len(features):
        raise ValueError('Duplicate domain features')
    valid_domains = {v.domain for v in validations if v.quality.status is QualityStatus.VALID and v.input is not None}
    if any(f.domain not in valid_domains for f in features):
        raise ValueError('Features cannot be assigned to invalid/ambiguous raw inputs')
    if mode is AssessmentMode.REAL_DEVICE:
        if predictions is not None:
            raise ValueError('Real-device mode must never receive demo predictions')
        return AssessmentResult(mode, features, tuple(v.quality for v in validations), None, None,
                                'UNCALIBRATED - FEATURES ONLY')
    if predictions is None:
        return AssessmentResult(mode, features, tuple(v.quality for v in validations), None, None,
                                'DEMO PREDICTIONS NOT RUN')
    if valid_domains != set(DOMAIN_ORDER) or {f.domain for f in features} != set(DOMAIN_ORDER):
        raise ValueError('No synthetic demo scores or MPS from an incomplete/invalid session')
    if tuple(p.domain for p in predictions) != DOMAIN_ORDER:
        raise ValueError('Exactly four ordered frozen predictions are required')
    mps = sum(p.score for p in predictions) / 16
    return AssessmentResult(mode, features, tuple(v.quality for v in validations), predictions,
                            mps, 'SYNTHETIC DEMO ONLY - NOT CLINICALLY VALIDATED')
