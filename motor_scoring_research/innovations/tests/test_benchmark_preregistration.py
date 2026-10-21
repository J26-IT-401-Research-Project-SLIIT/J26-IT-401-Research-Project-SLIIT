"""Metadata-only preregistration checks; no signals or model execution."""
import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from motor_scoring_research.innovations.benchmark.freeze_parent_split import (
    SOURCE_SHA256, assign_from_bytes,
)


ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / 'motor_scoring_research/innovations/benchmark'
SOURCE = ROOT / 'motor_scoring_research/v3/training/configs/frozen_session_plan.csv'


class BenchmarkPreregistrationTests(unittest.TestCase):
    def test_parent_manifest_is_exact_and_excludes_old_holdouts(self):
        source = SOURCE.read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), SOURCE_SHA256)
        assigned, expected = assign_from_bytes(source)
        manifest = (BENCH / 'frozen_parent_split_v1.csv').read_bytes()
        lock = json.loads((BENCH / 'frozen_parent_split_v1.lock.json').read_text())
        self.assertEqual(hashlib.sha256(manifest).hexdigest(), lock['manifest_sha256'])
        self.assertEqual(lock['counts'], {'development': 84, 'audit': 84})
        self.assertEqual(lock['profiles_by_split']['audit'], {'healthy_like': 42, 'symptom_like': 42})
        self.assertEqual(expected['original_validation_test_denylist_sha256'],
                         lock['original_validation_test_denylist_sha256'])
        rows = list(csv.DictReader(manifest.decode().splitlines()))
        self.assertEqual(rows, assigned)
        old = list(csv.DictReader(source.decode().splitlines()))
        forbidden = {row['session_id'] for row in old if row['split'] != 'train'}
        self.assertFalse({row['parent_session_id'] for row in rows} & forbidden)
        self.assertEqual(len({row['parent_session_id'] for row in rows}), 168)

    def test_preregistered_budgets_scenarios_and_policy_hashes(self):
        spec = json.loads((BENCH / 'preregistration_v1.json').read_text())
        lock = json.loads((BENCH / 'frozen_parent_split_v1.lock.json').read_text())
        self.assertEqual(spec['parent_split_manifest_sha256'], lock['manifest_sha256'])
        self.assertEqual(spec['original_validation_test_denylist_sha256'],
                         lock['original_validation_test_denylist_sha256'])
        self.assertEqual(spec['parent_counts'], {'development': 84, 'audit': 84})
        self.assertEqual(len(spec['scenario_slots_per_parent']), 7)
        self.assertEqual(spec['budgets_additional_recording_seconds'], [10, 20, 180, 210])
        self.assertFalse(spec['benchmark_generation_approved'])
        self.assertFalse(spec['benchmark_evaluation_approved'])
        config_root = ROOT / 'motor_scoring_research/innovations/configs'
        for name, expected in spec['frozen_policy_files_sha256'].items():
            self.assertEqual(hashlib.sha256((config_root / name).read_bytes()).hexdigest(), expected)

    def test_original_training_plan_families_are_shared(self):
        with (BENCH / 'frozen_parent_split_v1.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        families = {split: {row['generator_family'] for row in rows if row['innovation_split'] == split}
                    for split in ('development', 'audit')}
        self.assertEqual(families['development'], families['audit'])
        self.assertEqual({row['source_reference_id'] for row in rows}, {'SIM_ASSUMPTIONS_V3'})

    def test_design_lock_matches_every_registered_local_file(self):
        lock = json.loads((BENCH / 'PHASE_3F_A_DESIGN_LOCK.json').read_text())
        self.assertFalse(lock['audit_generation_approved'])
        self.assertIsNone(lock['remote_selected_models_manifest_sha256'])
        for relative, expected in lock['files_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                             expected, relative)


if __name__ == '__main__':
    unittest.main()
