"""Create a small, deterministic code-only Colab input ZIP, refusing overwrite."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from .feature_bridge import sha256


def source_paths(workspace: Path) -> tuple[Path, ...]:
    base = Path('motor_scoring_research')
    package = workspace / base / 'innovations'
    relative = [p.relative_to(workspace) for p in package.glob('*.py')]
    relative += [p.relative_to(workspace) for p in (package / 'integration').glob('*.py')
                 if p.name in ('__init__.py', 'feature_bridge.py', 'preflight.py')]
    relative += [p.relative_to(workspace) for p in (package / 'configs').glob('*.json')]
    relative += [base / 'reference_engine' / 'pp1_motor_engine.py',
                 base / 'v3' / 'training' / 'src' / 'pp1_training.py',
                 base / 'v3' / 'training' / 'requirements-colab.txt']
    result = tuple(sorted((workspace / path for path in relative), key=lambda p: p.as_posix()))
    if not result or any(not path.is_file() for path in result):
        raise FileNotFoundError('Required code-only bundle source is missing')
    return result


def build_bundle(workspace: Path | str, destination: Path | str) -> dict:
    workspace, destination = Path(workspace).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Refuse archive overwrite: {destination}')
    paths = source_paths(workspace)
    manifest = {'schema_version': 'pp1_preflight_code_bundle.v1',
                'purpose': 'read_only_colab_preflight_code',
                'files_sha256': {path.relative_to(workspace).as_posix(): sha256(path) for path in paths}}
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9, allowZip64=False) as archive:
        for path in paths:
            archive.write(path, path.relative_to(workspace).as_posix())
        archive.writestr('preflight_bundle_manifest.json',
                         json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC integrity failed')
        expected = set(manifest['files_sha256']) | {'preflight_bundle_manifest.json'}
        if set(archive.namelist()) != expected:
            raise ValueError('Unexpected archive members')
        for name, expected_sha in manifest['files_sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected_sha:
                raise ValueError(f'Archive content hash mismatch: {name}')
    return {'path': str(destination), 'size_bytes': destination.stat().st_size,
            'file_count': len(paths), 'archive_sha256': sha256(destination), 'manifest': manifest}
