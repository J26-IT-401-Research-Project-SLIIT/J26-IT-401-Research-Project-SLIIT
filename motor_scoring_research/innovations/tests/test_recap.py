"""Invented quality and TRUST JSON only; no data, checkpoints or inference."""
import copy
import json
import unittest

from motor_scoring_research.innovations.contracts import (
    AcquisitionStatus, AssessmentMode, AssessmentResult, DOMAIN_ORDER, Domain,
    FrozenPrediction, QualityFinding, QualityReport, QualityStatus, ReasonCode,
    ReassessmentAction,
)
from motor_scoring_research.innovations.recap import (
    RecapConfig, load_recap_config, recommend_reassessment,
)


def assessment(*, mode=AssessmentMode.SYNTHETIC_DEMO, quality=None):
    quality = quality or tuple(QualityReport(d, QualityStatus.VALID) for d in DOMAIN_ORDER)
    predictions = None if mode is AssessmentMode.REAL_DEVICE or any(q.status is not QualityStatus.VALID for q in quality) else tuple(
        FrozenPrediction(d, 'SVM' if d is Domain.TS else 'RF', 2, 'a'*64) for d in DOMAIN_ORDER)
    mps = .5 if predictions else None
    return AssessmentResult(mode, (), quality, predictions, mps, 'invented_fixture')


def trust_for(result, drifts=None, valid=10):
    drifts = drifts or {}
    diag = {}
    for d in DOMAIN_ORDER:
        drift = drifts.get(d, 0.0)
        # Invented but arithmetically consistent ordinal outcomes around score 2.
        total_drift = round(drift * valid)
        if (valid and abs(total_drift/valid - drift) > 1e-12) or total_drift > 2 * valid:
            raise ValueError('Fixture drift is not realizable')
        twos = total_drift // 2
        ones = total_drift % 2
        changes = twos + ones
        histogram = [0, 0, valid-changes, ones, twos]
        diag[d.value] = {'noise': {'attempted': valid, 'valid': valid, 'invalid': 0,
            'score_changes': changes, 'score_change_fraction_of_valid': changes/valid if valid else None,
            'mean_absolute_ordinal_drift': drift if valid else None,
            'score_histogram_0_to_4': histogram}}
    return {'schema_version': 'trust_mps.v1', 'mode': 'synthetic_demo',
            'status': 'valid_disturbance_sensitive' if drifts else 'valid_stable_for_tested_disturbances',
            'baseline_demo_scores': {d.value: 2 for d in DOMAIN_ORDER},
            'baseline_demo_mps': result.mps,
            'quality_by_domain': {q.domain.value: {'status': q.status.value,
                                  'reason_codes': [f.code.value for f in q.findings]}
                                  for q in result.quality},
            'domain_diagnostics': diag}


def changed_quality(domain, code, status=QualityStatus.INVALID,
                    acquisition=AcquisitionStatus.UNKNOWN):
    return tuple(QualityReport(d, status, (QualityFinding(code, 'invented'),),
                               acquisition_status=acquisition) if d is domain else
                 QualityReport(d, QualityStatus.VALID) for d in DOMAIN_ORDER)


def configured(**changes):
    old = load_recap_config()
    data = {field: getattr(old, field) for field in old.__dataclass_fields__}
    data.update(changes)
    return RecapConfig(**data)


class RecapTests(unittest.TestCase):
    def test_confirmed_corrected_failure_precedes_sensitivity_and_gi_cost(self):
        result = assessment(quality=changed_quality(Domain.GI, ReasonCode.INCOMPLETE_RECORDING))
        advice, report = recommend_reassessment(result, remediation={Domain.GI: 'corrected'})
        self.assertEqual(advice.action, ReassessmentAction.REPEAT_RECOMMENDED)
        self.assertEqual(advice.ranked_domains, (Domain.GI,))
        self.assertEqual(report['total_estimated_recording_seconds'], 180)
        self.assertEqual(report['modeled_instability_ranking'], [])

    def test_persistent_fault_requires_repair(self):
        result = assessment(quality=changed_quality(Domain.RS, ReasonCode.DECLARED_ACQUISITION_FAILURE,
                                                   acquisition=AcquisitionStatus.KNOWN_FAILURE))
        advice, report = recommend_reassessment(result, remediation={Domain.RS: 'unresolved'})
        self.assertEqual(advice.action, ReassessmentAction.CLARIFY_ACQUISITION)
        self.assertEqual(report['total_estimated_recording_seconds'], 0)
        self.assertEqual(report['per_domain']['RS']['eligibility_reason'], 'repair_persistent_fault_first')

    def test_unknown_empty_taps_clarification_not_bradykinesia(self):
        result = assessment(quality=changed_quality(Domain.BS, ReasonCode.UNKNOWN_TAP_ACQUISITION,
                                                    QualityStatus.UNKNOWN))
        advice, report = recommend_reassessment(result)
        self.assertEqual(advice.action, ReassessmentAction.CLARIFY_ACQUISITION)
        self.assertEqual(report['recommended_tasks'], [])
        self.assertEqual(report['per_domain']['BS']['eligibility_reason'], 'clarify_acquisition_first')

    def test_stable_no_repeat_and_immutable_scores(self):
        result = assessment()
        before = copy.deepcopy(result)
        advice, report = recommend_reassessment(result, trust_for(result))
        self.assertEqual(advice.action, ReassessmentAction.NONE)
        self.assertEqual(result, before)
        self.assertFalse(report['score_replacement'])
        self.assertIsNone(report['retest_outcomes'])
        self.assertFalse(report['automatic_execution'])

    def test_unstable_priority_uses_duration_and_drift(self):
        result = assessment()
        trust = trust_for(result, {Domain.TS: .2, Domain.GI: 1.5, Domain.BS: .1})
        advice, report = recommend_reassessment(result, trust)
        self.assertEqual(advice.ranked_domains, (Domain.TS,))
        self.assertEqual([row['domain'] for row in report['modeled_instability_ranking']],
                         ['TS', 'BS', 'GI'])
        self.assertAlmostEqual(report['modeled_instability_ranking'][0]['modeled_sensitivity_per_second'], .2/160)
        self.assertEqual(report['decision_status'], 'optional_retest_candidate')

    def test_budget_count_and_one_repeat_limit(self):
        result = assessment()
        trust = trust_for(result, {Domain.TS: .2, Domain.BS: .1, Domain.GI: 1.5})
        advice, report = recommend_reassessment(result, trust,
            config=configured(max_recommendations=2, recording_time_budget_seconds=20))
        self.assertEqual(advice.ranked_domains, (Domain.TS, Domain.BS))
        self.assertEqual(report['total_estimated_recording_seconds'], 20)
        advice, _ = recommend_reassessment(result, trust, prior_repeats={Domain.TS: 1})
        self.assertEqual(advice.ranked_domains, (Domain.BS,))
        advice, report = recommend_reassessment(result, trust, recording_time_budget_seconds=0)
        self.assertEqual(advice.action, ReassessmentAction.NONE)
        self.assertEqual(report['stopping_reason'], 'sensitivity_candidates_blocked_by_limits_or_budget')

    def test_equal_priority_tie_break(self):
        result = assessment()
        trust = trust_for(result, {Domain.TS: .2, Domain.BS: .2})
        advice, _ = recommend_reassessment(result, trust)
        self.assertEqual(advice.ranked_domains, (Domain.TS,))
        advice, _ = recommend_reassessment(result, trust,
                                            config=configured(tie_break_order=(Domain.BS, Domain.TS, Domain.RS, Domain.GI)))
        self.assertEqual(advice.ranked_domains, (Domain.BS,))

    def test_missing_invalid_and_zero_probe_trust_are_not_stable(self):
        result = assessment()
        for trust in (None, {}, trust_for(result, valid=0)):
            advice, report = recommend_reassessment(result, trust)
            self.assertEqual(advice.action, ReassessmentAction.NOT_ASSESSABLE)
            self.assertEqual(report['decision_status'], 'insufficient_evidence')
        bad = trust_for(result)
        bad['baseline_demo_mps'] = 0.0
        advice, report = recommend_reassessment(result, bad)
        self.assertEqual(report['stopping_reason'], 'trust_baseline_mismatch')
        bad = trust_for(result)
        bad['domain_diagnostics']['TS']['noise']['score_changes'] = 1
        advice, report = recommend_reassessment(result, bad)
        self.assertEqual(report['stopping_reason'], 'unsupported_or_invalid_trust_diagnostics')
        bad = trust_for(result, {Domain.TS: .1})
        bad['status'] = 'valid_stable_for_tested_disturbances'
        advice, report = recommend_reassessment(result, bad)
        self.assertEqual(report['stopping_reason'], 'trust_status_conflicts_with_score_changes')

    def test_real_device_never_uses_trust_or_exposes_scores(self):
        result = assessment(mode=AssessmentMode.REAL_DEVICE)
        advice, report = recommend_reassessment(result, {'baseline_demo_scores': {'TS': 4}})
        self.assertEqual(advice.action, ReassessmentAction.NONE)
        self.assertEqual(report['modeled_instability_ranking'], [])
        self.assertNotIn('mps', report)
        self.assertNotIn('demo_scores', report)
        self.assertIsNone(result.predictions)
        self.assertIsNone(result.mps)

    def test_repair_first_and_stop_conditions(self):
        result = assessment(quality=changed_quality(Domain.TS, ReasonCode.INCOMPLETE_RECORDING))
        advice, _ = recommend_reassessment(result, remediation={Domain.TS: 'unknown'})
        self.assertEqual(advice.action, ReassessmentAction.CLARIFY_ACQUISITION)
        advice, report = recommend_reassessment(result, remediation={Domain.TS: 'corrected'},
                                                recording_time_budget_seconds=0)
        self.assertEqual(advice.action, ReassessmentAction.NONE)
        self.assertEqual(report['per_domain']['TS']['eligibility_reason'], 'recording_time_budget')
        result = assessment()
        advice, _ = recommend_reassessment(result, trust_for(result, {Domain.TS: 1.}),
                                            operator_permits_repeat=False)
        self.assertEqual(advice.action, ReassessmentAction.NONE)

    def test_json_and_determinism(self):
        result = assessment()
        trust = trust_for(result, {Domain.TS: 1.})
        first = recommend_reassessment(result, trust)
        second = recommend_reassessment(result, trust)
        self.assertEqual(first, second)
        json.dumps(first[1], sort_keys=True)
        self.assertEqual(first[1]['schema_version'], 'recap_mps.v1')
        self.assertEqual(first[1]['recommended_tasks'][0]['duration_seconds'], 10)


if __name__ == '__main__':
    unittest.main()
