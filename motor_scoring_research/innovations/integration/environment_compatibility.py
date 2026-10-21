"""Read-only compatibility checks for the frozen training environment.

No dependency import, installation or model deserialization occurs here.
"""
from __future__ import annotations

import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path

PACKAGES = ('numpy', 'scipy', 'scikit-learn', 'xgboost', 'catboost', 'joblib')


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_recorded_environment(run_root: Path | str) -> dict:
    """Check environment.json against the existing pre-test FROZEN.json lock.

    This is one integrity check, not proof that FROZEN.json itself is trusted.
    The full inspect_frozen_run check must be repeated before any future load.
    """
    root = Path(run_root)
    frozen = json.loads((root / 'FROZEN.json').read_text(encoding='utf-8'))
    locked = frozen.get('files')
    relative = 'training/environment.json'
    if not isinstance(locked, dict) or relative not in locked:
        raise ValueError('Frozen environment lock missing')
    payload = (root / relative).read_bytes()
    if sha256_bytes(payload) != locked[relative]:
        raise ValueError('Frozen environment hash mismatch')
    result = json.loads(payload)
    if not isinstance(result, dict) or not isinstance(result.get('python'), str) or not result['python']:
        raise ValueError('Recorded Python version is missing')
    versions = result.get('packages')
    if not isinstance(versions, dict) or set(versions) != set(PACKAGES):
        raise ValueError('Recorded dependency schema differs from training source')
    if any(not isinstance(versions[name], str) or not versions[name] for name in PACKAGES):
        raise ValueError('Recorded dependency version is missing')
    if result.get('threads') != 1:
        raise ValueError('Recorded training thread setting differs')
    return {'python': result['python'], 'packages': {name: versions[name] for name in PACKAGES},
            'threads': 1, 'environment_sha256': sha256_bytes(payload)}


def current_environment() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'packages': versions}


def compare_environment(recorded: dict, current: dict) -> dict:
    """Fail closed on unknown Python or any mismatched package version."""
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        raise ValueError('Environment records must be mappings')
    python_recorded, python_current = recorded.get('python'), current.get('python')
    if not isinstance(python_recorded, str) or not python_recorded:
        return {'status': 'BLOCKED', 'reason': 'recorded_python_unknown'}
    if python_recorded != python_current:
        return {'status': 'BLOCKED', 'reason': 'exact_python_mismatch',
                'recorded_python': python_recorded, 'current_python': python_current}
    a, b = recorded.get('packages'), current.get('packages')
    if not isinstance(a, dict) or not isinstance(b, dict) or set(a) != set(PACKAGES):
        return {'status': 'BLOCKED', 'reason': 'dependency_provenance_incomplete'}
    mismatches = {name: {'recorded': a[name], 'current': b.get(name)}
                  for name in PACKAGES if not a[name] or a[name] != b.get(name)}
    if mismatches:
        return {'status': 'BLOCKED', 'reason': 'package_versions_mismatch', 'mismatches': mismatches}
    return {'status': 'PASS', 'reason': 'exact_python_and_package_versions_match'}
