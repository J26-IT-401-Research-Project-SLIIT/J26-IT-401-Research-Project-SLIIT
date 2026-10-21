"""Tiny invented observations and fake predictors only; no real model inference."""
import json
import unittest

from motor_scoring_research.innovations.contracts import (
    AssessmentMode, DOMAIN_ORDER, Domain, FEATURE_ORDER, FeatureVector,
    FrozenPrediction, QualityReport, QualityStatus, SensorSeries, TapSeries,
    ValidationResult,
)
from motor_scoring_research.innovations.perturbations import (
    InvalidProbe, Probe, apply_probe, validate_perturbed,
)
from motor_scoring_research.innovations.trust import TrustConfig, analyze_trust, load_config


HASH = 'a' * 64


def complete():
    result = []
    for domain in DOMAIN_ORDER:
        if domain is Domain.BS:
            raw = TapSeries((1.0, 2.0, 3.0))
        else:
            count = 18000 if domain is Domain.GI else 1000
            raw = SensorSeries(domain, tuple(i/100 for i in range(count)),
                               ((0.0, 0.0, 9.81),) * count)
        result.append(ValidationResult(domain, raw, QualityReport(domain, QualityStatus.VALID,
                                                                  samples_or_events=3 if domain is Domain.BS else count)))
    return tuple(result)


def fake_extract(raw):
    return tuple(FeatureVector(d, FEATURE_ORDER[d],
                               (raw[i].xyz[0][0],) * len(FEATURE_ORDER[d])
                               if d is not Domain.BS else (3.0, 18.0, 0.0))
                 for i, d in enumerate(DOMAIN_ORDER))


class FakePredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, feature):
        self.calls += 1
        score = 2 if feature.domain is Domain.TS and feature.values[0] >= 0.005 else 1
        algorithm = 'SVM' if feature.domain is Domain.TS else 'RF'
        return FrozenPrediction(feature.domain, algorithm, score, HASH)


class TrustTests(unittest.TestCase):
    def test_tap_count_order_and_schema(self):
        source = TapSeries((1., 2., 3.))
        new = apply_probe(source, Probe('tap_jitter', .005, 's', 4))
        self.assertEqual(len(new.tap_time_s), 3)
        self.assertTrue(all(b > a for a, b in zip(new.tap_time_s, new.tap_time_s[1:])))
        self.assertEqual(new, apply_probe(source, Probe('tap_jitter', .005, 's', 4)))
        self.assertEqual(source.tap_time_s, (1., 2., 3.))
        with self.assertRaises(InvalidProbe):
            apply_probe(TapSeries((1., 1.0001)), Probe('tap_jitter', 2., 's', 4))

    def test_sensor_shape_and_conditional_quantization(self):
        source = SensorSeries(Domain.TS, (0., .01), ((0., 0., 9.81), (1., 0., 9.81)))
        new = apply_probe(source, Probe('offset', .01, 'm/s^2', 2))
        self.assertEqual(new.time_s, source.time_s)
        self.assertEqual(len(new.xyz), len(source.xyz))
        self.assertEqual(new.domain, source.domain)
        with self.assertRaisesRegex(InvalidProbe, 'unverified_quantization_step'):
            apply_probe(source, Probe('quantization', .01, 'm/s^2', 2))
        with self.assertRaisesRegex(InvalidProbe, 'recording_length'):
            validate_perturbed(source, Domain.TS)
        bad_clock = SensorSeries(Domain.TS, (float('nan'),) + (0.,) * 999,
                                 ((0., 0., 9.81),) * 1000)
        with self.assertRaisesRegex(InvalidProbe, 'timestamp_grid'):
            validate_perturbed(bad_clock, Domain.TS)

    def test_bounded_drift_joint_arithmetic_and_reproducibility(self):
        base = load_config()
        config = TrustConfig('test', {d: ((('offset', .01, 'm/s^2'),) if d is Domain.TS else ())
                                       for d in DOMAIN_ORDER},
                             (('offset', .01, 'm/s^2'),), 2, base.sha256)
        a, report = analyze_trust(complete(), AssessmentMode.SYNTHETIC_DEMO,
                                  fake_extract, FakePredictor(), config=config, seed=8)
        _, again = analyze_trust(complete(), AssessmentMode.SYNTHETIC_DEMO,
                                 fake_extract, FakePredictor(), config=config, seed=8)
        self.assertEqual(report, again)
        self.assertEqual(report['attempted_probes'], 2)
        self.assertEqual(a.mps, .25)
        row = report['domain_diagnostics']['TS']['offset']
        self.assertEqual(row['score_change_fraction_of_valid'], 1)
        self.assertEqual(row['mean_absolute_ordinal_drift'], 1)
        self.assertEqual(row['score_histogram_0_to_4'], [0, 0, 1, 0, 0])
        self.assertEqual(report['joint_mps_sensitivity']['observed_width'], 1/16)
        self.assertEqual(report['joint_mps_sensitivity']['observed_max'], 5/16)
        self.assertFalse(report['joint_mps_sensitivity']['rectangular_is_jointly_observed'])
        json.dumps(report)

    def test_real_mode_never_calls_predictor(self):
        predictor = FakePredictor()
        assessment, report = analyze_trust(complete(), AssessmentMode.REAL_DEVICE,
                                            fake_extract, predictor)
        self.assertIsNone(assessment.predictions)
        self.assertIsNone(assessment.mps)
        self.assertIsNone(report['baseline_demo_scores'])
        self.assertEqual(predictor.calls, 0)
        self.assertEqual(report['status'], 'features_only_untestable_score_sensitivity')

    def test_partial_input_never_calls_extractor_or_predictor(self):
        validations = list(complete())
        validations[3] = ValidationResult(Domain.GI, None,
                                          QualityReport(Domain.GI, QualityStatus.INVALID))
        predictor = FakePredictor()
        def forbidden(_):
            self.fail('Partial session reached extractor')
        assessment, report = analyze_trust(tuple(validations), AssessmentMode.SYNTHETIC_DEMO,
                                            forbidden, predictor)
        self.assertIsNone(assessment.mps)
        self.assertIsNone(assessment.predictions)
        self.assertEqual(predictor.calls, 0)
        self.assertEqual(report['status'], 'hard_acquisition_failure_or_untestable')

    def test_budget_and_stability_not_correctness(self):
        original = load_config()
        config = TrustConfig('test', original.domain_probes, original.joint_probes, 1, original.sha256)
        with self.assertRaisesRegex(ValueError, 'budget exceeded'):
            analyze_trust(complete(), AssessmentMode.SYNTHETIC_DEMO,
                          fake_extract, FakePredictor(), config=config)
        config = TrustConfig('test', {d: () for d in DOMAIN_ORDER}, (), 1, original.sha256)
        _, report = analyze_trust(complete(), AssessmentMode.SYNTHETIC_DEMO,
                                  fake_extract, FakePredictor(), config=config)
        self.assertEqual(report['status'], 'untestable_no_valid_perturbations')
        self.assertIsNone(report['clinical_correctness'])
        self.assertIsNone(report['clinical_confidence'])
        config = TrustConfig('test', {d: ((('offset', 0., 'm/s^2'),) if d is Domain.TS else ())
                                       for d in DOMAIN_ORDER}, (), 1, original.sha256)
        _, report = analyze_trust(complete(), AssessmentMode.SYNTHETIC_DEMO,
                                  fake_extract, FakePredictor(), config=config)
        self.assertEqual(report['status'], 'valid_stable_for_tested_disturbances')
        self.assertEqual(report['domain_diagnostics']['TS']['offset']['boundary_status'],
                         'not_observed_within_finite_search')
        self.assertIsNone(report['clinical_correctness'])

    def test_forged_valid_raw_is_rejected_without_prediction(self):
        validations = list(complete())
        bad = SensorSeries(Domain.TS, (0.,), ((0., 0., 9.81),))
        validations[0] = ValidationResult(Domain.TS, bad,
                                          QualityReport(Domain.TS, QualityStatus.VALID))
        predictor = FakePredictor()
        def forbidden(_):
            self.fail('Invalid raw input reached extractor')
        assessment, report = analyze_trust(tuple(validations), AssessmentMode.SYNTHETIC_DEMO,
                                            forbidden, predictor)
        self.assertIsNone(assessment.predictions)
        self.assertEqual(predictor.calls, 0)
        self.assertIn('invalid_raw_input_', report['untestable_reason'])


if __name__ == '__main__':
    unittest.main()
