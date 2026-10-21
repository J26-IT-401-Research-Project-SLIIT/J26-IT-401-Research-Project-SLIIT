"""TRUST-MPS finite perturbation diagnostics for complete PP1 sessions.

This module does not load models or call the original engine by itself. A caller
injects the verified complete-session extractor and frozen predictors in Colab.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .assessment import assemble_assessment
from .contracts import (AssessmentMode, AssessmentResult, DOMAIN_ORDER, Domain,
                        FeatureVector, FrozenPrediction, QualityStatus, ValidationResult)
from .perturbations import InvalidProbe, Probe, apply_probe, validate_perturbed


class CompleteExtractor(Protocol):
    def __call__(self, raw: tuple[object, object, object, object]) -> tuple[FeatureVector, ...]: ...


class DemoPredictor(Protocol):
    def predict(self, feature: FeatureVector) -> FrozenPrediction: ...


@dataclass(frozen=True)
class TrustConfig:
    assumptions_id: str
    domain_probes: Mapping[Domain, tuple[tuple[str, float, str], ...]]
    joint_probes: tuple[tuple[str, float, str], ...]
    max_attempts: int
    sha256: str


def load_config(path: Path | str | None = None) -> TrustConfig:
    path = Path(path) if path is not None else Path(__file__).parent / 'configs' / 'trust_engineering_v1.json'
    payload = path.read_bytes()
    data = json.loads(payload)
    probes = {Domain(k): tuple((v['family'], float(v['amplitude']), v['unit']) for v in items)
              for k, items in data['domain_probes'].items()}
    if set(probes) != set(DOMAIN_ORDER) or not 0 < data['max_attempts'] <= 128:
        raise ValueError('Invalid finite TRUST configuration')
    joint = tuple((p['family'], float(p['amplitude']), p['unit']) for p in data['joint_probes'])
    if sum(map(len, probes.values())) + len(joint) > data['max_attempts']:
        raise ValueError('Configured probes exceed finite budget')
    if any(a < 0 or not math.isfinite(a) for items in (*probes.values(), joint) for _, a, _ in items):
        raise ValueError('Invalid probe amplitude')
    return TrustConfig(data['assumptions_id'], probes, joint, data['max_attempts'], hashlib.sha256(payload).hexdigest())


def _predict_all(raw: tuple[object, ...], extractor: CompleteExtractor,
                 predictor: DemoPredictor) -> tuple[tuple[FeatureVector, ...], tuple[FrozenPrediction, ...]]:
    features = tuple(extractor(raw))
    if tuple(f.domain for f in features) != DOMAIN_ORDER:
        raise ValueError('Extractor must yield four ordered frozen feature vectors')
    predictions = tuple(predictor.predict(f) for f in features)
    if tuple(p.domain for p in predictions) != DOMAIN_ORDER:
        raise ValueError('Predictor must yield four ordered frozen predictions')
    return features, predictions


def _quality_json(validations: tuple[ValidationResult, ...]) -> dict:
    return {v.domain.value: {'status': v.quality.status.value,
                             'reason_codes': [f.code.value for f in v.quality.findings],
                             'acquisition_status': v.quality.acquisition_status.value}
            for v in validations}


def _probe(raw, domain: Domain, family: str, amplitude: float, seed: int):
    if domain is Domain.BS and family != 'tap_jitter':
        # A joint IMU nuisance has no inferred tap-clock counterpart.
        return raw
    if domain is not Domain.BS and family == 'tap_jitter':
        return raw
    if family == 'noise' and domain is Domain.RS:
        amplitude *= 0.5  # Joint dimensionless scenario mapped to domain units.
    if family == 'noise' and domain is Domain.BS:
        return raw
    output = apply_probe(raw, Probe(family, amplitude, '', seed))
    validate_perturbed(output, domain)
    return output


def analyze_trust(validations: tuple[ValidationResult, ...], mode: AssessmentMode,
                  extractor: CompleteExtractor, predictor: DemoPredictor | None = None,
                  *, config: TrustConfig | None = None, seed: int = 0) -> tuple[AssessmentResult, dict]:
    """Return unchanged assessment plus separate JSON-ready diagnostic sidecar.

    Fractions condition on valid probes; invalid attempts are reported separately.
    The 'nearest' boundary is only the smallest observed amplitude in this finite
    library, within its own family/units. No clinical correctness is inferred.
    """
    mode = AssessmentMode(mode)
    config = config or load_config()
    if not 0 < config.max_attempts <= 128:
        raise ValueError('TRUST finite probe budget must be within 1..128')
    if tuple(v.domain for v in validations) != DOMAIN_ORDER:
        raise ValueError('Exactly four ordered task validation results required')
    if not isinstance(seed, int):
        raise ValueError('Seed must be integer')
    sidecar = {'schema_version': 'trust_mps.v1', 'mode': mode.value,
               'assumptions_id': config.assumptions_id, 'config_sha256': config.sha256,
               'seed': seed, 'quality_by_domain': _quality_json(validations),
               'clinical_confidence': None, 'clinical_correctness': None,
               'interval_kind': 'modeled_perturbation_sensitivity',
               'baseline_demo_scores': None, 'baseline_demo_mps': None,
               'domain_diagnostics': {}, 'joint_mps_sensitivity': None,
               'attempted_probes': 0, 'status': None}
    complete = all(v.input is not None and v.quality.status is QualityStatus.VALID for v in validations)
    if not complete:
        assessment = assemble_assessment(mode, validations)
        sidecar['status'] = 'hard_acquisition_failure_or_untestable'
        sidecar['untestable_reason'] = 'complete_valid_four_domain_input_required'
        return assessment, sidecar
    raw = tuple(v.input for v in validations)
    try:
        for d, x in zip(DOMAIN_ORDER, raw):
            validate_perturbed(x, d)  # Reject forged VALID wrappers before extractor.
    except InvalidProbe as exc:
        assessment = assemble_assessment(mode, validations)
        sidecar['status'] = 'hard_acquisition_failure_or_untestable'
        sidecar['untestable_reason'] = 'invalid_raw_input_' + str(exc)
        return assessment, sidecar
    features = tuple(extractor(raw))
    if tuple(f.domain for f in features) != DOMAIN_ORDER:
        raise ValueError('Extractor must yield four ordered frozen feature vectors')
    if mode is AssessmentMode.REAL_DEVICE:
        assessment = assemble_assessment(mode, validations, features)
        sidecar['status'] = 'features_only_untestable_score_sensitivity'
        sidecar['untestable_reason'] = 'real_device_uncalibrated'
        return assessment, sidecar
    if predictor is None:
        raise ValueError('Synthetic demo requires a verified predictor')
    baseline = tuple(predictor.predict(f) for f in features)
    assessment = assemble_assessment(mode, validations, features, baseline)
    scores = tuple(p.score for p in baseline)
    identities = tuple(p.checkpoint_sha256 for p in baseline)
    sidecar['baseline_demo_scores'] = dict(zip((d.value for d in DOMAIN_ORDER), scores))
    sidecar['baseline_demo_mps'] = assessment.mps
    attempted = 0
    observed_by_domain = [[s] for s in scores]
    any_flip = False
    for i, domain in enumerate(DOMAIN_ORDER):
        family_rows: dict[str, dict] = {}
        for j, (family, amplitude, unit) in enumerate(config.domain_probes[domain]):
            attempted += 1
            if attempted > config.max_attempts:
                raise ValueError('TRUST finite probe budget exceeded')
            row = family_rows.setdefault(family, {'unit': unit, 'attempted': 0, 'valid': 0,
                'invalid': 0, 'invalid_reasons': {}, 'score_histogram_0_to_4': [0]*5,
                'absolute_ordinal_drift_sum': 0, 'score_changes': 0,
                'nearest_observed_score_change_amplitude': None})
            if row['unit'] != unit:
                raise ValueError('One family cannot mix units')
            row['attempted'] += 1
            try:
                candidate = list(raw)
                candidate[i] = _probe(raw[i], domain, family, amplitude, seed + 1009*i + 37*j)
                _, preds = _predict_all(tuple(candidate), extractor, predictor)
                if tuple(p.checkpoint_sha256 for p in preds) != identities:
                    raise ValueError('Predictor checkpoint identity changed during probing')
                if any(preds[k].score != scores[k] for k in range(4) if k != i):
                    raise ValueError('One-domain perturbation changed an unchanged domain prediction')
            except InvalidProbe as exc:
                reason = str(exc)
                row['invalid'] += 1
                row['invalid_reasons'][reason] = row['invalid_reasons'].get(reason, 0) + 1
                continue
            current = preds[i].score
            row['valid'] += 1
            row['score_histogram_0_to_4'][current] += 1
            drift = abs(current - scores[i])
            row['absolute_ordinal_drift_sum'] += drift
            observed_by_domain[i].append(current)
            if drift:
                any_flip = True
                row['score_changes'] += 1
                nearest = row['nearest_observed_score_change_amplitude']
                if nearest is None or amplitude < nearest:
                    row['nearest_observed_score_change_amplitude'] = amplitude
        for row in family_rows.values():
            n = row['valid']
            row['score_change_fraction_of_valid'] = row['score_changes']/n if n else None
            row['mean_absolute_ordinal_drift'] = row.pop('absolute_ordinal_drift_sum')/n if n else None
            row['boundary_status'] = ('observed_within_finite_search' if row['nearest_observed_score_change_amplitude'] is not None
                                      else 'not_observed_within_finite_search' if n else 'untestable')
        sidecar['domain_diagnostics'][domain.value] = family_rows
    # Joint outcomes use the same scenario on all four recordings, preserving
    # their pairing. They are never built from marginal extrema.
    joint_values = [assessment.mps]
    joint_rows = []
    for j, (family, amplitude, unit) in enumerate(config.joint_probes):
        attempted += 1
        if attempted > config.max_attempts:
            raise ValueError('TRUST finite probe budget exceeded')
        try:
            candidate = tuple(_probe(x, d, family, amplitude, seed + 100000 + j)
                              for d, x in zip(DOMAIN_ORDER, raw))
            _, preds = _predict_all(candidate, extractor, predictor)
            if tuple(p.checkpoint_sha256 for p in preds) != identities:
                raise ValueError('Predictor checkpoint identity changed during joint probing')
        except InvalidProbe as exc:
            joint_rows.append({'scenario_id': j, 'family': family, 'valid': False, 'reason': str(exc)})
            continue
        mps = sum(p.score for p in preds)/16
        joint_values.append(mps)
        joint_rows.append({'scenario_id': j, 'family': family, 'amplitude': amplitude,
                           'unit': unit, 'valid': True, 'scores': [p.score for p in preds], 'demo_mps': mps})
    low, high = min(joint_values), max(joint_values)
    sidecar['joint_mps_sensitivity'] = {'observed_min': low, 'observed_max': high,
        'observed_width': high-low, 'included_unperturbed_baseline': True,
        'valid_joint_probe_count': len(joint_values)-1, 'scenarios': joint_rows,
        'rectangular_outer_min': sum(min(v) for v in observed_by_domain)/16,
        'rectangular_outer_max': sum(max(v) for v in observed_by_domain)/16,
        'rectangular_is_jointly_observed': False}
    sidecar['attempted_probes'] = attempted
    sidecar['status'] = ('untestable_no_valid_perturbations' if not any(
        row['valid'] for families in sidecar['domain_diagnostics'].values() for row in families.values())
        and len(joint_values) == 1 else
        'valid_disturbance_sensitive' if any_flip or high > low else
        'valid_stable_for_tested_disturbances')
    return assessment, sidecar
