"""Small invented environment manifests; no package installation or model load."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from motor_scoring_research.innovations.integration.environment_compatibility import (
    PACKAGES, compare_environment, read_recorded_environment,
)


class EnvironmentCompatibilityTests(unittest.TestCase):
    def test_exact_python_and_all_packages_required(self):
        versions = {name: '1.2.3' for name in PACKAGES}
        recorded = {'python': '3.13.16', 'packages': versions}
        current = {'python': '3.13.16', 'packages': versions.copy()}
        self.assertEqual(compare_environment(recorded, current)['status'], 'PASS')
        current['python'] = '3.13.15'
        self.assertEqual(compare_environment(recorded, current)['reason'], 'exact_python_mismatch')
        current['python'] = '3.13.16'
        current['packages']['joblib'] = None
        self.assertEqual(compare_environment(recorded, current)['reason'], 'package_versions_mismatch')

    def test_recorded_python_must_be_read_from_hashed_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / 'training/environment.json'
            target.parent.mkdir()
            payload = json.dumps({'python': '3.12.9', 'packages': {name:'1.0' for name in PACKAGES},
                                  'threads': 1}).encode()
            target.write_bytes(payload)
            (root / 'FROZEN.json').write_text(json.dumps({'files': {
                'training/environment.json': hashlib.sha256(payload).hexdigest()}}))
            result = read_recorded_environment(root)
            self.assertEqual(result['python'], '3.12.9')
            target.write_bytes(payload + b' ')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                read_recorded_environment(root)


if __name__ == '__main__':
    unittest.main()
