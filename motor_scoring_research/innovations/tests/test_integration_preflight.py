"""Static and invented-data contract checks; no model deserialization."""
import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path

from motor_scoring_research.innovations.contracts import Domain, SensorSeries, TapSeries
from motor_scoring_research.innovations.integration.build_code_bundle import source_paths
from motor_scoring_research.innovations.integration.build_notebook import notebook
from motor_scoring_research.innovations.integration.feature_bridge import CompleteSessionFeatureBridge
from motor_scoring_research.innovations.integration.preflight import (
    discover_run, report_status, verify_engine_source, verify_frozen,
)


WORKSPACE = Path(__file__).resolve().parents[3]


class FakeEngine:
    def __init__(self):
        self.calls = []

    def extract_features(self, ts, bs, rs, gi, fs=100):
        self.calls.append((len(ts), len(bs), len(rs), len(gi), fs))
        return {'TS': {'tremor_rms_ms2': .1},
                'BS': {'tap_count': 3, 'tap_rate_per_min': 18., 'tap_interval_std_s': 0.},
                'RS': {'mean_gyro_peak_rad_s': 1., 'movement_peaks': 2},
                'GI': {'step_candidates': 100, 'cadence_steps_min': 33.333,
                       'step_interval_std_s': .01}}


def invented_raw():
    return (SensorSeries(Domain.TS, tuple(i/100 for i in range(1000)),
                         ((0., 0., 9.81),)*1000),
            TapSeries((1., 2., 3.)),
            SensorSeries(Domain.RS, tuple(i/100 for i in range(1000)),
                         ((0., 0., 0.),)*1000),
            SensorSeries(Domain.GI, tuple(i/100 for i in range(18000)),
                         ((0., 0., 9.81),)*18000))


class IntegrationPreflightTests(unittest.TestCase):
    def test_exact_four_recording_bridge_with_fake_engine(self):
        engine = FakeEngine()
        bridge = CompleteSessionFeatureBridge(engine)
        original = invented_raw()
        preserved = copy.deepcopy(original)
        values = bridge(original)
        self.assertEqual(engine.calls, [(1000, 3, 1000, 18000, 100)])
        self.assertEqual([f.domain for f in values], list(Domain))
        self.assertEqual(values[1].values, (3., 18., 0.))
        self.assertEqual(original, preserved)
        with self.assertRaisesRegex(ValueError, 'four recordings'):
            bridge(original[:3])
        self.assertEqual(len(engine.calls), 1)

    def test_undefined_feature_is_not_imputed(self):
        class Undefined(FakeEngine):
            def extract_features(self, *args, **kwargs):
                values = super().extract_features(*args, **kwargs)
                values['RS']['mean_gyro_peak_rad_s'] = None
                return values
        with self.assertRaisesRegex(ValueError, 'undefined RS feature'):
            CompleteSessionFeatureBridge(Undefined())(invented_raw())

    def test_engine_training_static_contract_and_no_remote_run(self):
        engine = WORKSPACE / 'motor_scoring_research/reference_engine/pp1_motor_engine.py'
        training = WORKSPACE / 'motor_scoring_research/v3/training/src/pp1_training.py'
        self.assertEqual(verify_engine_source(engine, training)['status'], 'PASS')
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(discover_run(Path(temp) / 'missing')['status'], 'BLOCKED')
            self.assertEqual(verify_frozen(Path(temp) / 'missing')['status'], 'BLOCKED')
        self.assertEqual(report_status({'a': {'status':'PASS'}, 'b': {'status':'BLOCKED'}}), 'BLOCKED')

    def test_code_only_manifest_and_notebook_are_static(self):
        paths = source_paths(WORKSPACE)
        self.assertTrue(paths)
        self.assertTrue(all(p.suffix in ('.py', '.json', '.txt') for p in paths))
        self.assertFalse(any('tests' in p.parts or p.suffix == '.joblib' for p in paths))
        cells = notebook()['cells']
        self.assertEqual(len(cells), 9)
        for cell in cells:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'], [])
        self.assertIn('RUN_READ_ONLY_PREFLIGHT = False', ''.join(cells[4]['source']))
        json.dumps(notebook())


if __name__ == '__main__':
    unittest.main()
