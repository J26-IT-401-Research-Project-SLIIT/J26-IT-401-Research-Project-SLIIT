"""RECAP-MPS: bounded, recommendation-only policy on quality and TRUST reports."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .contracts import (AcquisitionStatus, AssessmentMode, AssessmentResult,
                        DOMAIN_ORDER, DURATION_SECONDS, Domain, QualityStatus,
                        ReasonCode, ReassessmentAction, ReassessmentAdvice)


@dataclass(frozen=True)
class RecapConfig:
    assumptions_id: str
    max_recommendations: int
    recording_time_budget_seconds: int
    max_repeats_per_domain: int
    max_rounds_per_session: int
    minimum_valid_probes_per_domain: int
    minimum_observed_score_changes: int
    tie_break_order: tuple[Domain, ...]
    sha256: str

    def __post_init__(self) -> None:
        if not 1 <= self.max_recommendations <= 4 or not 0 <= self.recording_time_budget_seconds <= 840:
            raise ValueError('RECAP recommendation/time budget outside finite limits')
        if self.max_repeats_per_domain != 1 or not 1 <= self.max_rounds_per_session <= 4:
            raise ValueError('RECAP repeat/round limit outside finite limits')
        if not 1 <= self.minimum_valid_probes_per_domain <= 128 or not 1 <= self.minimum_observed_score_changes <= 128:
            raise ValueError('RECAP evidence requirement outside finite limits')
        if set(self.tie_break_order) != set(DOMAIN_ORDER) or len(self.tie_break_order) != 4:
            raise ValueError('Tie order must contain every domain exactly once')


def load_recap_config(path: Path | str | None = None) -> RecapConfig:
    path = Path(path) if path is not None else Path(__file__).parent / 'configs' / 'recap_policy_v1.json'
    payload = path.read_bytes()
    data = json.loads(payload)
    return RecapConfig(data['assumptions_id'], data['max_recommendations'],
                       data['recording_time_budget_seconds'], data['max_repeats_per_domain'],
                       data['max_rounds_per_session'], data['minimum_valid_probes_per_domain'],
                       data['minimum_observed_score_changes'],
                       tuple(Domain(d) for d in data['tie_break_order']),
                       hashlib.sha256(payload).hexdigest())


_REPAIR_CODES = frozenset({ReasonCode.DECLARED_ACQUISITION_FAILURE,
                           ReasonCode.UNIT_MISMATCH, ReasonCode.SESSION_MISMATCH})
_AMBIGUOUS_CODES = frozenset({ReasonCode.UNKNOWN_TAP_ACQUISITION,
                              ReasonCode.UNSUPPORTED_FEATURE})


def _validated_trust(assessment: AssessmentResult, trust: dict | None) -> tuple[dict | None, str]:
    """Fail closed on malformed, stale or incompatible synthetic sidecars."""
    if trust is None:
        return None, 'trust_unavailable'
    if not isinstance(trust, dict) or trust.get('schema_version') != 'trust_mps.v1':
        return None, 'trust_schema_invalid'
    if trust.get('mode') != AssessmentMode.SYNTHETIC_DEMO.value:
        return None, 'trust_mode_mismatch'
    if trust.get('status') not in ('valid_disturbance_sensitive',
                                   'valid_stable_for_tested_disturbances',
                                   'untestable_no_valid_perturbations'):
        return None, 'trust_status_unusable'
    if assessment.predictions is None or assessment.mps is None:
        return None, 'assessment_has_no_demo_scores'
    expected = {p.domain.value: p.score for p in assessment.predictions}
    if trust.get('baseline_demo_scores') != expected or trust.get('baseline_demo_mps') != assessment.mps:
        return None, 'trust_baseline_mismatch'
    quality = trust.get('quality_by_domain')
    if not isinstance(quality, dict):
        return None, 'trust_quality_missing'
    for report in assessment.quality:
        item = quality.get(report.domain.value)
        if not isinstance(item, dict) or item.get('status') != report.status.value or item.get('reason_codes') != [f.code.value for f in report.findings]:
            return None, 'trust_quality_mismatch'
    if not isinstance(trust.get('domain_diagnostics'), dict):
        return None, 'trust_diagnostics_missing'
    return trust, 'trust_accepted'


def _evidence(families: object, baseline_score: int) -> tuple[int, int, float] | None:
    if not isinstance(families, dict) or not families:
        return None
    valid_total = changes_total = 0
    drift_sum = 0.0
    for row in families.values():
        if not isinstance(row, dict):
            return None
        valid, changes, drift = row.get('valid'), row.get('score_changes'), row.get('mean_absolute_ordinal_drift')
        attempted, invalid = row.get('attempted'), row.get('invalid')
        if any(type(x) is not int or x < 0 for x in (valid, changes, attempted, invalid)):
            return None
        if valid + invalid != attempted or changes > valid:
            return None
        if valid:
            if type(drift) not in (int, float) or not math.isfinite(drift) or not 0 <= drift <= 4:
                return None
            histogram = row.get('score_histogram_0_to_4')
            if not isinstance(histogram, list) or len(histogram) != 5 or any(type(v) is not int or v < 0 for v in histogram) or sum(histogram) != valid:
                return None
            observed_changes = valid - histogram[baseline_score]
            observed_drift = sum(abs(score - baseline_score) * count
                                 for score, count in enumerate(histogram)) / valid
            if changes != observed_changes or abs(drift - observed_drift) > 1e-12:
                return None
            fraction = row.get('score_change_fraction_of_valid')
            if type(fraction) not in (int, float) or not math.isfinite(fraction) or abs(fraction - changes/valid) > 1e-12:
                return None
            if changes == 0 and drift != 0 or changes > 0 and drift <= 0:
                return None
            drift_sum += valid * drift
        elif drift is not None:
            return None
        valid_total += valid
        changes_total += changes
    return valid_total, changes_total, drift_sum/valid_total if valid_total else 0.0


def recommend_reassessment(
    assessment: AssessmentResult, trust: dict | None = None, *,
    config: RecapConfig | None = None,
    remediation: Mapping[Domain, str] | None = None,
    prior_repeats: Mapping[Domain, int] | None = None,
    completed_rounds: int = 0,
    recording_time_budget_seconds: int | None = None,
    operator_permits_repeat: bool = True,
) -> tuple[ReassessmentAdvice, dict]:
    """Recommend only; never invokes hardware, extractor, model or score mutation.

    remediation values: ``corrected`` (verified fix), ``unresolved`` (persistent
    fault), or ``unknown``. Absence means unknown. A corrected quality failure
    takes precedence over all score-sensitive candidates regardless of duration.
    """
    config = config or load_recap_config()
    remediation, prior_repeats = remediation or {}, prior_repeats or {}
    budget = config.recording_time_budget_seconds if recording_time_budget_seconds is None else recording_time_budget_seconds
    if type(budget) is not int or not 0 <= budget <= 840:
        raise ValueError('Recording-time budget must be an integer within 0..840 seconds')
    if type(completed_rounds) is not int or completed_rounds < 0:
        raise ValueError('Invalid completed-round count')
    if set(remediation) - set(DOMAIN_ORDER) or any(v not in ('corrected', 'unresolved', 'unknown') for v in remediation.values()):
        raise ValueError('Invalid remediation state')
    if set(prior_repeats) - set(DOMAIN_ORDER) or any(type(v) is not int or v < 0 for v in prior_repeats.values()):
        raise ValueError('Invalid repeat history')
    if tuple(q.domain for q in assessment.quality) != DOMAIN_ORDER:
        raise ValueError('Complete ordered quality reports are required')
    if assessment.mode is AssessmentMode.REAL_DEVICE and (assessment.predictions is not None or assessment.mps is not None):
        raise ValueError('Real-device assessment cannot carry demo scores')
    sidecar = {'schema_version': 'recap_mps.v1', 'mode': assessment.mode.value,
               'assumptions_id': config.assumptions_id, 'config_sha256': config.sha256,
               'decision_status': None, 'recommended_tasks': [], 'per_domain': {},
               'modeled_instability_ranking': [], 'total_estimated_recording_seconds': 0,
               'recording_time_budget_seconds': budget, 'stopping_reason': None,
               'automatic_execution': False, 'retest_outcomes': None,
               'score_replacement': False,
               'limitations': ['Recommendation only; recording time excludes setup, recovery and computation.',
                               'No expected accuracy improvement or clinical benefit is estimated.',
                               'No acquisition is triggered and no existing scores or MPS are changed.']}
    if assessment.mode is AssessmentMode.REAL_DEVICE:
        sidecar['limitations'].append('Real-device mode has no validated ordinal scores or MPS; score-dependent ranking is disabled.')
    def finish(action: ReassessmentAction, domains: tuple[Domain, ...], reason: str, status: str):
        seconds = sum(DURATION_SECONDS[d] for d in domains)
        advice = ReassessmentAdvice(action, domains, (reason,), seconds, config.assumptions_id)
        sidecar['decision_status'] = status
        sidecar['recommended_tasks'] = [{'domain': d.value, 'duration_seconds': DURATION_SECONDS[d]} for d in domains]
        sidecar['total_estimated_recording_seconds'] = seconds
        sidecar['stopping_reason'] = reason
        return advice, sidecar
    priority = {d: i for i, d in enumerate(config.tie_break_order)}
    corrected, blocking = [], []
    for q in assessment.quality:
        d = q.domain
        codes = {f.code for f in q.findings}
        state = remediation.get(d, 'unknown')
        row = {'quality_status': q.status.value, 'acquisition_status': q.acquisition_status.value,
               'quality_reason_codes': [f.code.value for f in q.findings],
               'remediation_status': state if q.status is not QualityStatus.VALID else None,
               'eligible': False, 'eligibility_reason': None,
               'duration_seconds': DURATION_SECONDS[d], 'prior_repeats': prior_repeats.get(d, 0),
               'modeled_mean_absolute_score_drift': None, 'modeled_sensitivity_per_second': None}
        sidecar['per_domain'][d.value] = row
        if q.status is QualityStatus.VALID:
            row['eligibility_reason'] = 'quality_valid_pending_trust'
        elif ReasonCode.UNKNOWN_TAP_ACQUISITION in codes or q.status is QualityStatus.UNKNOWN:
            row['eligibility_reason'] = 'clarify_acquisition_first'
            blocking.append(d)
        elif codes & _AMBIGUOUS_CODES:
            row['eligibility_reason'] = 'unsupported_feature_or_ambiguous_acquisition'
            blocking.append(d)
        elif state == 'unresolved' or codes & _REPAIR_CODES and state != 'corrected':
            row['eligibility_reason'] = 'repair_persistent_fault_first'
            blocking.append(d)
        elif state != 'corrected' or q.acquisition_status is AcquisitionStatus.UNKNOWN and not codes:
            row['eligibility_reason'] = 'verify_and_correct_acquisition_first'
            blocking.append(d)
        else:
            row['eligible'] = True
            row['eligibility_reason'] = 'confirmed_failure_corrected'
            corrected.append(d)
    if blocking:
        return finish(ReassessmentAction.CLARIFY_ACQUISITION, (),
                      'clarify_or_repair_acquisition_before_repeat', 'acquisition_action_required')
    if completed_rounds >= config.max_rounds_per_session or not operator_permits_repeat:
        return finish(ReassessmentAction.NONE, (),
                      'round_limit_or_operator_stop', 'stopped')
    def select(candidates: list[Domain]) -> tuple[Domain, ...]:
        selected = []
        remaining = budget
        for d in candidates:
            row = sidecar['per_domain'][d.value]
            if prior_repeats.get(d, 0) >= config.max_repeats_per_domain:
                row['eligible'] = False
                row['eligibility_reason'] = 'per_domain_repeat_limit'
            elif len(selected) >= config.max_recommendations:
                row['eligible'] = False
                row['eligibility_reason'] = 'recommendation_count_limit'
            elif DURATION_SECONDS[d] > remaining:
                row['eligible'] = False
                row['eligibility_reason'] = 'recording_time_budget'
            else:
                selected.append(d)
                remaining -= DURATION_SECONDS[d]
        return tuple(selected)
    if corrected:
        selected = select(sorted(corrected, key=priority.get))
        if selected:
            return finish(ReassessmentAction.REPEAT_RECOMMENDED, selected,
                          'confirmed_corrected_acquisition_failure', 'repeat_recommended')
        return finish(ReassessmentAction.NONE, (), 'corrected_failure_blocked_by_limits_or_budget', 'stopped')
    if assessment.mode is AssessmentMode.REAL_DEVICE:
        for row in sidecar['per_domain'].values():
            if row['quality_status'] == QualityStatus.VALID.value:
                row['eligibility_reason'] = 'real_device_quality_valid_no_score_based_retest'
        return finish(ReassessmentAction.NONE, (), 'no_confirmed_correctable_acquisition_failure', 'no_repeat')
    accepted, trust_reason = _validated_trust(assessment, trust)
    if accepted is None:
        return finish(ReassessmentAction.NOT_ASSESSABLE, (), trust_reason, 'insufficient_evidence')
    ranked = []
    total_valid = total_changes = 0
    for d in DOMAIN_ORDER:
        row = sidecar['per_domain'][d.value]
        evidence = _evidence(accepted['domain_diagnostics'].get(d.value),
                             accepted['baseline_demo_scores'][d.value])
        if evidence is None:
            row['eligibility_reason'] = 'unsupported_or_invalid_trust_diagnostics'
            return finish(ReassessmentAction.NOT_ASSESSABLE, (),
                          'unsupported_or_invalid_trust_diagnostics', 'insufficient_evidence')
        valid, changes, drift = evidence
        total_valid += valid
        total_changes += changes
        row['valid_probes'] = valid
        row['observed_score_changes'] = changes
        if valid < config.minimum_valid_probes_per_domain:
            row['eligibility_reason'] = 'insufficient_valid_probes'
            return finish(ReassessmentAction.NOT_ASSESSABLE, (),
                          'insufficient_valid_probes', 'insufficient_evidence')
        row['modeled_mean_absolute_score_drift'] = drift
        if changes >= config.minimum_observed_score_changes and drift > 0:
            value = drift/(16*DURATION_SECONDS[d])
            row['modeled_sensitivity_per_second'] = value
            row['eligible'] = True
            row['eligibility_reason'] = 'modeled_disturbance_sensitive'
            ranked.append((d, value, drift))
        else:
            row['eligibility_reason'] = 'no_tested_instability_meets_minimum_evidence'
    if accepted['status'] == 'valid_stable_for_tested_disturbances' and total_changes:
        return finish(ReassessmentAction.NOT_ASSESSABLE, (),
                      'trust_status_conflicts_with_score_changes', 'insufficient_evidence')
    if accepted['status'] == 'untestable_no_valid_perturbations' and total_valid:
        return finish(ReassessmentAction.NOT_ASSESSABLE, (),
                      'trust_status_conflicts_with_valid_probes', 'insufficient_evidence')
    ranked.sort(key=lambda item: (-item[1], priority[item[0]]))
    sidecar['modeled_instability_ranking'] = [
        {'domain': d.value, 'modeled_mean_absolute_score_drift': drift,
         'modeled_sensitivity_per_second': value, 'duration_seconds': DURATION_SECONDS[d]}
        for d, value, drift in ranked]
    if not ranked:
        return finish(ReassessmentAction.NONE, (), 'no_tested_instability_warrants_repeat', 'no_repeat')
    selected = select([d for d, _, _ in ranked])
    if not selected:
        return finish(ReassessmentAction.NONE, (), 'sensitivity_candidates_blocked_by_limits_or_budget', 'stopped')
    return finish(ReassessmentAction.REPEAT_RECOMMENDED, selected,
                  'modeled_sensitivity_per_second_heuristic', 'optional_retest_candidate')
