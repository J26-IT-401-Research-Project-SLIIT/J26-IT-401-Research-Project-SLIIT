"""Fail-closed preregistration, lineage and trusted checkpoint checks."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ..frozen_adapter import inspect_frozen_run


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def verify_design(workspace: Path) -> tuple[dict, dict[str, dict]]:
    root = workspace.resolve(strict=True)
    folder = root / 'motor_scoring_research/innovations/benchmark'
    design = read_json(folder / 'PHASE_3F_A_DESIGN_LOCK.json')
    if design.get('audit_generation_approved') is not False or design.get('status') != 'design_only_no_benchmark_generation_or_evaluation':
        raise ValueError('Phase 3F-A design gate changed')
    for name, expected in design['files_sha256'].items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError(f'Design-lock mismatch: {name}')
    spec = read_json(folder / 'preregistration_v1.json')
    split_lock = read_json(folder / 'frozen_parent_split_v1.lock.json')
    if (spec['benchmark_generation_approved'] is not False or
            spec['benchmark_evaluation_approved'] is not False or
            spec['parent_source_sha256'] != split_lock['source_sha256'] or
            spec['parent_split_manifest_sha256'] != split_lock['manifest_sha256'] or
            sha256(folder / spec['parent_split_manifest']) != split_lock['manifest_sha256'] or
            sha256(root / spec['parent_source']) != split_lock['source_sha256']):
        raise ValueError('Preregistration or source lock mismatch')
    for name, expected in spec['frozen_policy_files_sha256'].items():
        if sha256(root / 'motor_scoring_research/innovations/configs' / name) != expected:
            raise ValueError(f'Policy configuration changed: {name}')
    with (root / spec['parent_source']).open(newline='', encoding='utf-8') as stream:
        source = list(csv.DictReader(stream))
    with (folder / spec['parent_split_manifest']).open(newline='', encoding='utf-8') as stream:
        split = list(csv.DictReader(stream))
    by_id = {row['session_id']: row for row in source}
    if len(by_id) != 240 or len(split) != 168:
        raise ValueError('Frozen V3 parent count changed')
    denied = sorted(sid for sid, row in by_id.items() if row['split'] in ('validation', 'test'))
    deny_hash = hashlib.sha256(('\n'.join(denied) + '\n').encode()).hexdigest()
    if len(denied) != 72 or deny_hash != split_lock['original_validation_test_denylist_sha256']:
        raise ValueError('Validation/test denylist mismatch')
    parents = {}
    counts = {'development': 0, 'audit': 0}
    for row in split:
        sid = row['parent_session_id']
        if sid in parents or sid not in by_id or by_id[sid]['split'] != 'train' or sid in denied:
            raise ValueError('Parent split leaked or duplicated')
        if any(row[k] != by_id[sid][k] for k in ('profile', 'generator_family', 'source_reference_id')):
            raise ValueError('Parent source metadata mismatch')
        label = row['innovation_split']
        if label not in counts:
            raise ValueError('Unknown innovation split')
        parents[sid] = row
        counts[label] += 1
    if counts != {'development': 84, 'audit': 84} or set(parents) != {sid for sid, row in by_id.items() if row['split'] == 'train'}:
        raise ValueError('Parent assignment mismatch')
    return spec, parents


def verify_frozen_models(run_root: Path, expected_selected_sha256: str):
    """Hash-only verification; no checkpoint deserialization."""
    if not isinstance(expected_selected_sha256, str) or len(expected_selected_sha256) != 64:
        raise ValueError('Explicit selected-model manifest hash required from verified preflight')
    manifest = inspect_frozen_run(run_root)
    if manifest.selected_models_sha256 != expected_selected_sha256:
        raise ValueError('Selected-model manifest hash differs from trusted preflight')
    return manifest


def eligible_ids(parents: dict[str, dict], split: str, *, audit_enabled: bool = False) -> tuple[str, ...]:
    if split not in ('development', 'audit') or (split == 'audit' and not audit_enabled):
        raise PermissionError('Audit parents are sealed')
    rows = (row for row in parents.values() if row['innovation_split'] == split)
    return tuple(row['parent_session_id'] for row in sorted(rows, key=lambda r: (r['split_key_sha256'], r['parent_session_id'])))


def claim_audit(registry: Path, claim: dict, *, approved: bool, frozen_development_sha256: str) -> Path:
    """Claim before reading any audit raw data. Existing marker is never removed."""
    if not approved or not frozen_development_sha256 or len(frozen_development_sha256) != 64:
        raise PermissionError('Audit requires explicit approval and frozen development lock')
    if not registry.is_dir():
        raise FileNotFoundError('Persistent Google Drive registry directory must already exist')
    expected = Path('/content/drive/MyDrive/PP1_V3_innovation_audit_registry')
    if registry.resolve() != expected:
        raise ValueError('Audit registry must be the single fixed Google Drive path')
    marker = registry / 'AUDIT_STARTED.json'
    payload = {**claim, 'frozen_development_sha256': frozen_development_sha256}
    with marker.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
    return marker
