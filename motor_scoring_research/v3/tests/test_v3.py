"""Tiny generated fixtures only. Never read V2 recordings or generate sessions."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pp1_v3 as v3
import numpy as np


class TestV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = v3.read_config()
        cls.engine = v3.load_engine(v3.DEFAULT_ENGINE, cls.c)

    def test_original_engine_hash(self):
        self.assertEqual(v3.sha256(v3.DEFAULT_ENGINE), self.c['engine_sha256'])

    def test_high_boundary_inclusive_low_boundary_exclusive(self):
        for i, threshold in enumerate(self.engine.DEMO_THRESHOLDS['TS']):
            self.assertEqual(self.engine.score_high(threshold, self.engine.DEMO_THRESHOLDS['TS']), i+1)
            self.assertEqual(self.engine.score_high(np.nextafter(threshold, -np.inf), self.engine.DEMO_THRESHOLDS['TS']), i)
        for d in ['BS','RS','GI']:
            for i, threshold in enumerate(self.engine.DEMO_THRESHOLDS[d]):
                self.assertEqual(self.engine.score_low(threshold, self.engine.DEMO_THRESHOLDS[d]), i)
                self.assertEqual(self.engine.score_low(np.nextafter(threshold, -np.inf), self.engine.DEMO_THRESHOLDS[d]), i+1)

    def test_ts_four_physical_feasibility_fixture(self):
        # A 0.25-second / 25-sample mathematical fixture, not an assessment session.
        t = np.arange(25)/100
        acceleration = np.column_stack([np.zeros(25),np.zeros(25),9.81+1.2*np.sin(2*np.pi*4*t)])
        rms = np.sqrt(np.mean((np.linalg.norm(acceleration,axis=1)-9.81)**2))
        self.assertAlmostEqual(rms, 1.2/np.sqrt(2))
        self.assertEqual(self.engine.score_high(rms, self.engine.DEMO_THRESHOLDS['TS']), 4)
        self.assertLess(1.2/(2*np.pi*4)**2, .002)  # displacement <2 mm

    def test_rotation_preserves_magnitude(self):
        x = np.array([[1.,2.,9.81],[-1.,0.,9.81]])
        out = v3.orient(x,np.array([0.,.01]),[.4,.8,-.2],.15)
        np.testing.assert_allclose(np.linalg.norm(out,axis=1),np.linalg.norm(x,axis=1))

    def test_plan_reproducible_and_disjoint(self):
        a = v3.session_plan(self.c)
        self.assertEqual(a,v3.session_plan(self.c))
        self.assertEqual(len(a),240)
        self.assertEqual(len({r['session_id'] for r in a}),240)
        self.assertEqual(len({r['seed'] for r in a}),240)
        self.assertEqual({s:sum(r['split']==s for r in a) for s in ['train','validation','test']},
                         {'train':168,'validation':36,'test':36})

    def test_stress_disjoint(self):
        base = v3.session_plan(self.c); stress = v3.session_plan(self.c,True)
        self.assertEqual(len(stress),40)
        self.assertFalse({r['seed'] for r in base}&{r['seed'] for r in stress})
        self.assertTrue(all(r['split']=='stress' for r in stress))

    def test_parameter_reproducibility(self):
        row = v3.session_plan(self.c)[0]
        p = v3.sample_parameters(self.c,row)
        self.assertEqual(p,v3.sample_parameters(self.c,row))
        self.assertGreaterEqual(p['ts_amp'],.035)
        self.assertLessEqual(p['ts_amp'],2.6)
        self.assertNotIn('scores',p)

    def test_tap_fixture(self):
        p = {'bs_cv':.3,'bs_hz':4.5,'bs_hesitation_probability':.1,'bs_fatigue_fraction':.2}
        a = v3.tap_events(p,self.c,np.random.default_rng(42),duration=1)
        self.assertTrue(np.all(np.diff(a)>0))
        self.assertTrue(np.all((a>=0)&(a<1)))

    def test_tiny_waveforms_finite_and_deterministic(self):
        row = v3.session_plan(self.c)[0]
        for family in self.c['families'] + [self.c['stress_family']]:
            row = {**row, 'generator_family': family}
            p = v3.sample_parameters(self.c,row)
            durations = dict(TS=.2,BS=.2,RS=.2,GI=.2)
            a = v3._synthesize(p,self.c,durations)
            b = v3._synthesize(p,self.c,durations)
            for d in ['TS','RS','GI']:
                self.assertEqual(a[d].shape,(20,3))
                self.assertTrue(np.isfinite(a[d]).all())
                np.testing.assert_array_equal(a[d],b[d])

    def test_real_mode_has_no_scores_with_tiny_feature_stub(self):
        with patch.object(self.engine,'extract_features',return_value={'fixture':True}):
            result = self.engine.assess_motor(None,None,None,None,mode='real')
            self.assertIsNone(result['scores'])
            self.assertIsNone(result['mps'])

    def test_missing_rs_peak_keeps_mps_null(self):
        fake = {'TS':{'tremor_rms_ms2':.1},'BS':{'tap_rate_per_min':200},
                'RS':{'mean_gyro_peak_rad_s':None},'GI':{'cadence_steps_min':100}}
        with patch.object(self.engine,'extract_features',return_value=fake):
            result=self.engine.assess_motor(None,None,None,None)
            self.assertIsNone(result['scores']['RS'])
            self.assertIsNone(result['mps'])

    def test_writes_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.json'
            v3.write_json(path,{'x':1})
            with self.assertRaises(FileExistsError):
                v3.write_json(path,{'x':2})

    def test_generation_approval_gate(self):
        with self.assertRaises(PermissionError):
            v3.generate(v3.DEFAULT_CONFIG,v3.DEFAULT_ENGINE,'must_not_be_created')

    def test_generation_runtime_gate(self):
        with patch.object(v3,'require_colab_cpu',side_effect=RuntimeError('blocked')):
            with self.assertRaises(RuntimeError):
                v3.generate(v3.DEFAULT_CONFIG,v3.DEFAULT_ENGINE,'must_not_be_created',True)

    def test_hash_rejects_changed_engine(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'engine.py'; path.write_text('raise RuntimeError("must not execute")')
            with self.assertRaises(ValueError):
                v3.load_engine(path,self.c)

    def test_feature_allowlist(self):
        self.assertEqual(sum(map(len,v3.FEATURES.values())),9)
        for fields in v3.FEATURES.values():
            self.assertFalse({'profile','seed','split','demo_MPS','scores'}&set(fields))


if __name__ == '__main__':
    unittest.main()
