"""Metadata-only innovation parent assignment; no signals, labels or models read."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

SOURCE_SHA256 = '134ddcd16393f5122818f3eabb14d6661d443d1efc1d33f13a267fbf9a1c76bf'
SALT = 'PP1_INNOVATION_PARENT_SPLIT_V1|20261022|'
FIELDS = ('parent_session_id', 'profile', 'generator_family', 'source_reference_id',
          'innovation_split', 'split_key_sha256')


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def assign_from_bytes(source: bytes, *, require_pinned_source: bool = True) -> tuple[list[dict], dict]:
    """Stratify only on profile; every descendant inherits this parent split."""
    source_hash = sha256(source)
    if require_pinned_source and source_hash != SOURCE_SHA256:
        raise ValueError('Frozen V3 split metadata hash mismatch')
    rows = list(csv.DictReader(io.StringIO(source.decode('utf-8-sig'), newline='')))
    expected_columns = {'session_id', 'split', 'profile', 'seed',
                        'generator_family', 'source_reference_id'}
    if not rows or set(rows[0]) != expected_columns or len(rows) != 240:
        raise ValueError('Unexpected frozen V3 session-plan schema or count')
    ids = [row['session_id'] for row in rows]
    if len(ids) != len(set(ids)) or Counter(row['split'] for row in rows) != {
            'train': 168, 'validation': 36, 'test': 36}:
        raise ValueError('Frozen V3 IDs or split counts changed')
    training = [row for row in rows if row['split'] == 'train']
    if Counter(row['profile'] for row in training) != {'healthy_like': 84, 'symptom_like': 84}:
        raise ValueError('Expected two 84-parent synthetic profiles')
    assigned = []
    for profile in ('healthy_like', 'symptom_like'):
        members = []
        for row in training:
            if row['profile'] == profile:
                key = sha256((SALT + row['session_id']).encode('utf-8'))
                members.append((key, row['session_id'], row))
        members.sort(key=lambda item: (item[0], item[1]))
        for position, (key, _, row) in enumerate(members):
            assigned.append({'parent_session_id': row['session_id'], 'profile': profile,
                             'generator_family': row['generator_family'],
                             'source_reference_id': row['source_reference_id'],
                             'innovation_split': 'development' if position < 42 else 'audit',
                             'split_key_sha256': key})
    assigned.sort(key=lambda row: row['parent_session_id'])
    denylist = sorted(row['session_id'] for row in rows if row['split'] != 'train')
    if {row['parent_session_id'] for row in assigned} & set(denylist):
        raise ValueError('Original validation/test parent entered innovation split')
    metadata = {'schema_version': 'pp1_innovation_parent_split.v1',
                'source_sha256': source_hash, 'salt': SALT,
                'original_validation_test_denylist_sha256': sha256(('\n'.join(denylist) + '\n').encode()),
                'original_validation_test_denylist_count': len(denylist),
                'counts': dict(Counter(row['innovation_split'] for row in assigned)),
                'profiles_by_split': {split: dict(Counter(row['profile'] for row in assigned
                                                         if row['innovation_split'] == split))
                                      for split in ('development', 'audit')},
                'generator_family_by_split': {split: dict(Counter(row['generator_family'] for row in assigned
                                                                 if row['innovation_split'] == split))
                                              for split in ('development', 'audit')},
                'source_independence_claim': False}
    return assigned, metadata


def freeze(source_path: Path | str, manifest_path: Path | str, lock_path: Path | str) -> dict:
    source_path, manifest_path, lock_path = Path(source_path), Path(manifest_path), Path(lock_path)
    if manifest_path.exists() or lock_path.exists():
        raise FileExistsError('Refuse to overwrite frozen innovation split')
    rows, metadata = assign_from_bytes(source_path.read_bytes())
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue().encode('utf-8')
    metadata['manifest_sha256'] = sha256(payload)
    metadata['manifest_rows'] = len(rows)
    with manifest_path.open('xb') as stream:
        stream.write(payload)
    with lock_path.open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, sort_keys=True, indent=2)
        stream.write('\n')
    return metadata
