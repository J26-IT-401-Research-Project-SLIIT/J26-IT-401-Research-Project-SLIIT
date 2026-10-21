"""Invented CSV/metadata fixtures only; never load a saved ML checkpoint."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from motor_scoring_research.innovations.assessment import assemble_assessment
from motor_scoring_research.innovations.contracts import (
    ALGORITHMS, DOMAIN_ORDER, DURATION_SECONDS, FEATURE_ORDER, AcquisitionContext,
    AcquisitionStatus, AssessmentMode, Domain, FeatureVector, FrozenPrediction,
    MpsSensitivity, QualityReport, QualityStatus, RAW_COLUMNS, ReassessmentAction,
    ReassessmentAdvice, ReasonCode, SensorSeries, TapSeries, ValidationResult,
)
from motor_scoring_research.innovations.frozen_adapter import (
    decode_ordinal, inspect_frozen_run, load_verified_models,
)
from motor_scoring_research.innovations.quality import validate_session, validate_task_csv

HEX = 'a' * 64


def write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def tiny_sensor(path: Path, domain: Domain, count: int, *, bad_time: int | None = None,
                nonfinite: int | None = None, blank: int | None = None) -> None:
    rows = []
    for i in range(count):
        t = i / 100
        if i == bad_time: t += .02
        axes = ('0.0', '0.0', '9.81') if domain is not Domain.RS else ('0.1', '0.0', '0.0')
        if i == nonfinite: axes = ('nan', axes[1], axes[2])
        if i == blank: axes = ('', axes[1], axes[2])
        rows.append((t, *axes))
    write_csv(path, RAW_COLUMNS[domain], rows)


def fake_validations() -> tuple[ValidationResult, ...]:
    results = []
    for domain in DOMAIN_ORDER:
        if domain is Domain.BS:
            raw = TapSeries((.1, .4))
        else:
            raw = SensorSeries(domain, (0.0,), ((0.0, 0.0, 9.81),))
        results.append(ValidationResult(domain, raw, QualityReport(domain, QualityStatus.VALID)))
    return tuple(results)


def fake_features() -> tuple[FeatureVector, ...]:
    return tuple(FeatureVector(domain, FEATURE_ORDER[domain], tuple(1.0 for _ in FEATURE_ORDER[domain]))
                 for domain in DOMAIN_ORDER)


def fake_predictions() -> tuple[FrozenPrediction, ...]:
    return tuple(FrozenPrediction(domain, ALGORITHMS[domain], score, HEX)
                 for domain, score in zip(DOMAIN_ORDER, (0, 1, 2, 3)))


def fake_frozen_run(root: Path) -> None:
    files: dict[str, str] = {}
    selected: dict[str, object] = {}
    selection = {'selected': {d.value: ALGORITHMS[d] for d in DOMAIN_ORDER}}
    def save(relative: str, data: bytes):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[relative] = hashlib.sha256(data).hexdigest()
    for domain in DOMAIN_ORDER:
        algorithm = ALGORITHMS[domain]
        pipeline = f'training/{domain.value}/{algorithm}/pipeline.joblib'
        metadata = f'training/{domain.value}/{algorithm}/metadata.json'
        save(pipeline, b'FAKE BUNDLE; NEVER DESERIALIZE')
        meta = {'algorithm': algorithm, 'domain': domain.value, 'target': domain.value,
                'features': list(FEATURE_ORDER[domain]), 'ordinal_classes': [0, 1, 2, 3, 4]}
        save(metadata, json.dumps(meta).encode())
        selected[domain.value] = {'algorithm': algorithm, 'pipeline': pipeline,
                                  'pipeline_sha256': files[pipeline], 'features': list(FEATURE_ORDER[domain])}
    save('selected_models.json', json.dumps(selected).encode())
    save('validation/selection.json', json.dumps(selection).encode())
    save('training/environment.json', b'{"python":"fixture"}')
    (root / 'FROZEN.json').write_text(json.dumps({'files': files, 'selected': selection['selected'],
                                                  'candidate_count': 16, 'test_predictions_accessed': False}))


class QualityTests(unittest.TestCase):
    def test_valid_exact_sensor_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'TS.csv'; tiny_sensor(path, Domain.TS, 1000)
            result = validate_task_csv(Domain.TS, path)
            self.assertEqual(result.quality.status, QualityStatus.VALID)
            self.assertEqual(len(result.input.xyz), 1000)
            self.assertEqual(result.input.time_s[-1], 9.99)

    def test_incomplete_and_timestamp_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'TS.csv'; tiny_sensor(path, Domain.TS, 999)
            result = validate_task_csv(Domain.TS, path)
            self.assertEqual(result.quality.findings[0].code, ReasonCode.INCOMPLETE_RECORDING)
            self.assertIsNone(result.input)
            path.unlink(); tiny_sensor(path, Domain.TS, 1000, bad_time=20)
            result = validate_task_csv(Domain.TS, path)
            self.assertEqual(result.quality.findings[0].code, ReasonCode.TIMESTAMP_GRID)

    def test_invalid_schema_missing_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'x.csv'
            write_csv(path, ('wrong',), [(1,)])
            self.assertEqual(validate_task_csv(Domain.BS, path).quality.findings[0].code,
                             ReasonCode.INVALID_SCHEMA)
            path.unlink(); tiny_sensor(path, Domain.RS, 1000, blank=3)
            self.assertEqual(validate_task_csv(Domain.RS, path).quality.findings[0].code,
                             ReasonCode.MISSING_VALUE)
            path.unlink(); tiny_sensor(path, Domain.RS, 1000, nonfinite=4)
            self.assertEqual(validate_task_csv(Domain.RS, path).quality.findings[0].code,
                             ReasonCode.NONFINITE_VALUE)

    def test_tap_order_range_and_no_fabrication(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'BS.csv'; write_csv(path, RAW_COLUMNS[Domain.BS], [])
            unknown = validate_task_csv(Domain.BS, path)
            self.assertEqual(unknown.quality.status, QualityStatus.UNKNOWN)
            self.assertEqual(unknown.quality.findings[0].code, ReasonCode.UNKNOWN_TAP_ACQUISITION)
            self.assertIsNone(unknown.input)
            verified = validate_task_csv(Domain.BS, path, AcquisitionContext(AcquisitionStatus.CONFIRMED_COMPLETE))
            self.assertEqual(verified.quality.status, QualityStatus.SUSPECT)
            self.assertEqual(verified.quality.findings[0].code, ReasonCode.UNSUPPORTED_FEATURE)
            self.assertIsNone(verified.input)
            path.unlink(); write_csv(path, RAW_COLUMNS[Domain.BS], [(.1,), (.1,)])
            self.assertEqual(validate_task_csv(Domain.BS, path).quality.findings[0].code,
                             ReasonCode.TAP_TIMESTAMP_ORDER)
            path.unlink(); write_csv(path, RAW_COLUMNS[Domain.BS], [(.1,), (10.0,)])
            self.assertEqual(validate_task_csv(Domain.BS, path).quality.findings[0].code,
                             ReasonCode.TAP_TIMESTAMP_RANGE)

    def test_valid_slow_irregular_taps_are_not_quality_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'BS.csv'
            write_csv(path, RAW_COLUMNS[Domain.BS], [(0.2,), (2.7,), (9.8,)])
            result = validate_task_csv(Domain.BS, path)
            self.assertEqual(result.quality.status, QualityStatus.VALID)
            self.assertEqual(result.input.tap_time_s, (.2, 2.7, 9.8))

    def test_declared_failure_and_wrong_unit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'BS.csv'
            write_csv(path, RAW_COLUMNS[Domain.BS], [(.1,), (.5,)])
            failed = validate_task_csv(Domain.BS, path, AcquisitionContext(AcquisitionStatus.KNOWN_FAILURE))
            self.assertEqual(failed.quality.findings[0].code, ReasonCode.DECLARED_ACQUISITION_FAILURE)
            wrong = validate_task_csv(Domain.BS, path, AcquisitionContext(declared_unit='ms'))
            self.assertEqual(wrong.quality.findings[0].code, ReasonCode.UNIT_MISMATCH)
            self.assertIsNone(wrong.input)

    def test_partial_session_has_no_dummy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'BS.csv'; write_csv(path, RAW_COLUMNS[Domain.BS], [(.1,), (.3,)])
            results = validate_session({Domain.BS: path})
            self.assertEqual(tuple(r.domain for r in results), DOMAIN_ORDER)
            self.assertEqual(sum(r.input is not None for r in results), 1)
            self.assertTrue(all(r.quality.findings[0].code is ReasonCode.MISSING_FILE
                                for r in results if r.domain is not Domain.BS))

    def test_declared_session_mismatch_prevents_mixing_tasks(self):
        contexts = {Domain.TS: AcquisitionContext(session_id='A'),
                    Domain.BS: AcquisitionContext(session_id='B')}
        result = validate_session({}, contexts)
        self.assertTrue(all(r.input is None and r.quality.status is QualityStatus.INVALID for r in result))
        self.assertTrue(all(r.quality.findings[0].code is ReasonCode.SESSION_MISMATCH for r in result))


class ContractTests(unittest.TestCase):
    def test_ordinal_decoder(self):
        self.assertEqual(decode_ordinal((0, 2, 4), 1.0), 2)
        for bad in (True, 1.2, 3, float('nan'), -1):
            with self.assertRaises(ValueError): decode_ordinal((0, 2, 4), bad)
        with self.assertRaises(ValueError):
            FeatureVector(Domain.BS, tuple(reversed(FEATURE_ORDER[Domain.BS])), (1.0, 2.0, 3.0))

    def test_mps_exact_and_immutable_sources(self):
        validations = fake_validations(); features = fake_features(); predictions = fake_predictions()
        before = (validations, features, predictions)
        result = assemble_assessment(AssessmentMode.SYNTHETIC_DEMO, validations, features, predictions)
        self.assertEqual(result.mps, (0 + 1 + 2 + 3) / 16)
        self.assertEqual(before, (validations, features, predictions))
        with self.assertRaises(FrozenInstanceError):
            result.mps = 1.0
        with self.assertRaises(FrozenInstanceError):
            predictions[0].score = 4

    def test_real_device_mode_never_scores(self):
        validations = fake_validations()
        result = assemble_assessment(AssessmentMode.REAL_DEVICE, validations,
                                     (fake_features()[0],))
        self.assertIsNone(result.predictions)
        self.assertIsNone(result.mps)
        with self.assertRaises(ValueError):
            assemble_assessment(AssessmentMode.REAL_DEVICE, validations, fake_features(), fake_predictions())

    def test_invalid_session_cannot_have_demo_score(self):
        validations = list(fake_validations())
        validations[1] = ValidationResult(Domain.BS, None, QualityReport(Domain.BS, QualityStatus.UNKNOWN))
        with self.assertRaises(ValueError):
            assemble_assessment(AssessmentMode.SYNTHETIC_DEMO, tuple(validations),
                                fake_features(), fake_predictions())
        with self.assertRaises(ValueError):
            assemble_assessment(AssessmentMode.SYNTHETIC_DEMO, fake_validations(),
                                fake_features(), fake_predictions()[:3])

    def test_fixed_retest_durations_and_no_device_execution(self):
        advice = ReassessmentAdvice(ReassessmentAction.REPEAT_RECOMMENDED, (Domain.GI,),
                                    ('modeled_instability',), DURATION_SECONDS[Domain.GI], 'fixture')
        self.assertFalse(advice.automatic_execution)
        with self.assertRaises(ValueError):
            ReassessmentAdvice(ReassessmentAction.REPEAT_RECOMMENDED, (Domain.GI,), (), 10, None)
        with self.assertRaises(ValueError):
            ReassessmentAdvice(ReassessmentAction.REPEAT_RECOMMENDED, (Domain.TS,), (), 10, None, True)
        with self.assertRaises(ValueError):
            MpsSensitivity(0.2, 0.3, 0.1, 0.4, 'clinical_confidence_interval')


class FrozenAdapterTests(unittest.TestCase):
    def test_read_only_manifest_inspection_and_load_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); fake_frozen_run(root)
            # A final-test file is deliberately outside FROZEN.files and is not read.
            (root / 'test').mkdir()
            (root / 'test' / 'final_report.json').write_text('not a valid JSON document')
            verified = inspect_frozen_run(root)
            self.assertEqual([spec.algorithm for spec in verified.models], ['SVM','RF','RF','RF'])
            with self.assertRaises(PermissionError):
                load_verified_models(verified)
            with self.assertRaises(RuntimeError):
                load_verified_models(verified, enable_inference=True)

    def test_checkpoint_change_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); fake_frozen_run(root)
            path = root / 'training/TS/SVM/pipeline.joblib'
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                inspect_frozen_run(root)

    def test_feature_schema_or_selection_change_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); fake_frozen_run(root)
            path = root / 'selected_models.json'
            data = json.loads(path.read_text()); data['TS']['features'] = ['wrong']
            path.write_text(json.dumps(data))
            frozen = json.loads((root / 'FROZEN.json').read_text())
            frozen['files']['selected_models.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / 'FROZEN.json').write_text(json.dumps(frozen))
            with self.assertRaisesRegex(ValueError, 'schema/decoder mismatch'):
                inspect_frozen_run(root)


if __name__ == '__main__':
    unittest.main()
