"""Tiny invented fixtures only: no model import, inference or real V3 raw data."""
import unittest
from pathlib import Path

from motor_scoring_research.innovations.benchmark.locks import claim_audit, eligible_ids, verify_design
from motor_scoring_research.innovations.benchmark.metrics import (cluster_interval, mps,
                                                                  paired_effect, success_checks)
from motor_scoring_research.innovations.benchmark.policies import decide, reveal_selected
from motor_scoring_research.innovations.benchmark.scenarios import derived_seed, scenario_pair
from motor_scoring_research.innovations.contracts import (AcquisitionStatus, AssessmentMode,
    AssessmentResult, DOMAIN_ORDER, Domain, QualityReport, QualityStatus, SensorSeries, TapSeries)
from motor_scoring_research.innovations.recap import load_recap_config

ROOT = Path(__file__).resolve().parents[3]


def tiny_anchor():
    # Scenario logic works on invented tiny arrays; protocol validation is
    # tested separately and rejects these as actual full recordings.
    return (SensorSeries(Domain.TS, (0., .01, .02), ((0., 0., 9.8),)*3),
            TapSeries((.2, .5, .8, 1.1)),
            SensorSeries(Domain.RS, (0., .01, .02), ((.1, .2, .3),)*3),
            SensorSeries(Domain.GI, (0., .01, .02), ((0., 0., 9.8),)*3))


def fake_assessment(statuses=(QualityStatus.VALID,)*4):
    return AssessmentResult(AssessmentMode.SYNTHETIC_DEMO, (),
        tuple(QualityReport(d, s, (), 2, AcquisitionStatus.CONFIRMED_COMPLETE)
              for d, s in zip(DOMAIN_ORDER, statuses)), None, None, 'invented')


class BenchmarkRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec, cls.parents = verify_design(ROOT)

    def test_split_sealed_and_no_crossover(self):
        dev = eligible_ids(self.parents, 'development')
        self.assertEqual(len(dev), 84)
        with self.assertRaises(PermissionError):
            eligible_ids(self.parents, 'audit')
        self.assertTrue(all(self.parents[x]['innovation_split'] == 'development' for x in dev))

    def test_seed_and_repeat_bank_determinism(self):
        self.assertEqual(derived_seed(7, 'invented', 'S1', 'initial', Domain.TS),
                         derived_seed(7, 'invented', 'S1', 'initial', Domain.TS))
        a = scenario_pair('invented', 'development', 0, 'S1', tiny_anchor(), self.spec)
        b = scenario_pair('invented', 'development', 0, 'S1', tiny_anchor(), self.spec)
        self.assertEqual(a, b)
        self.assertEqual(set(a[1]), set(DOMAIN_ORDER))
        self.assertEqual(a[0].split, 'development')
        self.assertTrue(all(v.parent_id == 'invented' and v.split == 'development' for v in a[1].values()))

    def test_bs_jitter_preserves_order_count(self):
        initial, bank = scenario_pair('invented', 'development', 0, 'S2', tiny_anchor(), self.spec)
        self.assertEqual(len(initial.raw[1].tap_time_s), 4)
        self.assertEqual(len(bank[Domain.BS].raw[1].tap_time_s), 4)
        self.assertEqual(sorted(initial.raw[1].tap_time_s), list(initial.raw[1].tap_time_s))

    def test_budget_and_no_outcome_peeking(self):
        assessment = fake_assessment()
        config = load_recap_config()
        decision = decide('C_random_eligible', assessment, None, {}, 10,
                          parent_id='invented', slot='S0', random_seed=7, recap_config=config)
        self.assertLessEqual(decision.additional_seconds, 10)
        self.assertNotIn(Domain.GI, decision.tasks)
        repeat_all = decide('B_repeat_all', assessment, None, {}, 180,
                            parent_id='invented', slot='S0', random_seed=7, recap_config=config)
        self.assertFalse(repeat_all.feasible)
        initial, bank = scenario_pair('invented', 'development', 0, 'S0', tiny_anchor(), self.spec)
        chosen = reveal_selected(initial.raw, {d: bank[d].raw for d in DOMAIN_ORDER}, decision)
        self.assertEqual(initial.raw[3], chosen[3])

    def test_metrics_and_parent_cluster_unit(self):
        self.assertEqual(mps((0, 1, 2, 3)), .375)
        self.assertEqual(paired_effect(.5, .375, .375), .125)
        rows = [{'parent_id': 'a', 'x': 0}, {'parent_id': 'a', 'x': 1},
                {'parent_id': 'b', 'x': 1}]
        result = cluster_interval(rows, {'a': 'healthy_like', 'b': 'symptom_like'}, 'x')
        self.assertEqual(result['supporting_parent_clusters'], 2)
        self.assertEqual(result['point'], .75)
        self.assertEqual(success_checks(*([{'status': 'indeterminate'}]*4))['overall'], 'indeterminate')

    def test_immutable_score_and_real_mode(self):
        assessment = fake_assessment()
        config = load_recap_config()
        before = repr(assessment)
        decide('F_full_trust_recap', assessment, None, {}, 180,
               parent_id='invented', slot='S0', random_seed=7, recap_config=config)
        self.assertEqual(repr(assessment), before)
        real = AssessmentResult(AssessmentMode.REAL_DEVICE, (), assessment.quality,
                                None, None, 'invented features only')
        chosen = decide('F_full_trust_recap', real, None, {}, 180,
                        parent_id='invented', slot='S0', random_seed=7, recap_config=config)
        self.assertEqual(chosen.tasks, ())
        self.assertIsNone(real.mps)

    def test_audit_claim_refuses_without_approval_or_drive_registry(self):
        with self.assertRaises(PermissionError):
            claim_audit(Path('/content/drive/MyDrive/PP1_V3_innovation_audit_registry'), {},
                        approved=False, frozen_development_sha256='a'*64)


if __name__ == '__main__':
    unittest.main()
