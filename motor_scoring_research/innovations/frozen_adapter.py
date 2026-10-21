"""Read-only manifest verification and future Colab-only frozen-model adapter.

Inspecting a manifest reads hashes/JSON only. Model deserialization/prediction
requires an explicit Colab CPU call and is never used by Phase 3B local tests.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ALGORITHMS, DOMAIN_ORDER, FEATURE_ORDER, Domain, FeatureVector, FrozenPrediction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f'Expected JSON object: {path.name}')
    return result


def _within(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or '\\' in relative or ':' in relative or any(p in ('', '.', '..') for p in relative.split('/')):
        raise ValueError(f'Unsafe checkpoint path: {relative}')
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f'Checkpoint path leaves run root: {relative}')
    return target


def decode_ordinal(classes: tuple[int, ...], encoded: object) -> int:
    """Decode the training-side integer index; never use it as a raw score."""
    if isinstance(encoded, bool):
        raise ValueError('Boolean prediction code is invalid')
    try:
        code = float(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError('Non-numeric prediction code') from exc
    if not math.isfinite(code) or not code.is_integer() or int(code) not in range(len(classes)):
        raise ValueError('Prediction code is outside the frozen ordinal decoder')
    score = classes[int(code)]
    if type(score) is not int or score not in range(5):
        raise ValueError('Decoded score is outside 0–4')
    return score


@dataclass(frozen=True)
class ModelSpec:
    domain: Domain
    algorithm: str
    pipeline_path: Path
    checkpoint_sha256: str
    feature_names: tuple[str, ...]
    ordinal_classes: tuple[int, ...]


@dataclass(frozen=True)
class VerifiedFrozenManifest:
    run_root: Path
    frozen_sha256: str
    selected_models_sha256: str
    models: tuple[ModelSpec, ...]

    def model_for(self, domain: Domain) -> ModelSpec:
        return next(spec for spec in self.models if spec.domain is Domain(domain))


def inspect_frozen_run(run_root: Path | str) -> VerifiedFrozenManifest:
    """Read-only: verify pre-test freeze and selected pipelines; no joblib.load.

    FROZEN.json is authored before test in V3. Its file list is the permitted
    verification scope, so final-test result files are never read here.
    """
    root = Path(run_root).resolve(strict=True)
    frozen_path = _within(root, 'FROZEN.json')
    frozen = _read_json(frozen_path)
    files = frozen.get('files')
    if not isinstance(files, dict) or frozen.get('candidate_count') != 16 or frozen.get('test_predictions_accessed') is not False:
        raise ValueError('Run is not a valid pre-test frozen 16-candidate checkpoint')
    if any(relative.startswith('test/') for relative in files):
        raise ValueError('Test results must not enter the pre-test frozen manifest')
    required = {'selected_models.json', 'validation/selection.json', 'training/environment.json'}
    if not required.issubset(files):
        raise ValueError('Frozen manifest lacks selected-model or environment metadata')
    # Check all frozen artifacts, including trained preprocessing and decoder files.
    for relative, expected_hash in files.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError('Malformed frozen hash entry')
        if _sha256(_within(root, relative)) != expected_hash:
            raise ValueError(f'Frozen artifact hash mismatch: {relative}')
    selected_path = _within(root, 'selected_models.json')
    selected = _read_json(selected_path)
    selection = _read_json(_within(root, 'validation/selection.json'))
    if set(selected) != {d.value for d in DOMAIN_ORDER} or frozen.get('selected') != selection.get('selected'):
        raise ValueError('Selected algorithms disagree with frozen validation decision')
    specs: list[ModelSpec] = []
    for domain in DOMAIN_ORDER:
        value = selected[domain.value]
        algorithm = ALGORITHMS[domain]
        relative = f'training/{domain.value}/{algorithm}/pipeline.joblib'
        metadata_relative = f'training/{domain.value}/{algorithm}/metadata.json'
        if relative not in files or metadata_relative not in files:
            raise ValueError(f'Selected {domain.value} artifacts are not frozen')
        meta = _read_json(_within(root, metadata_relative))
        classes = meta.get('ordinal_classes')
        if (value.get('algorithm') != algorithm or value.get('pipeline') != relative or
                value.get('pipeline_sha256') != files[relative] or
                selection['selected'].get(domain.value) != algorithm or
                meta.get('algorithm') != algorithm or meta.get('domain') != domain.value or
                meta.get('target') != domain.value or
                meta.get('features') != list(FEATURE_ORDER[domain]) or
                value.get('features') != list(FEATURE_ORDER[domain]) or
                not isinstance(classes, list) or not classes or
                any(type(c) is not int or c not in range(5) for c in classes) or
                classes != sorted(set(classes))):
            raise ValueError(f'Frozen selected model schema/decoder mismatch: {domain.value}')
        specs.append(ModelSpec(domain, algorithm, _within(root, relative), files[relative],
                               FEATURE_ORDER[domain], tuple(classes)))
    return VerifiedFrozenManifest(root, _sha256(frozen_path), _sha256(selected_path), tuple(specs))


def _require_colab_cpu() -> None:
    if not Path('/content').is_dir() or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Checkpoint deserialization and inference require Colab CPU')
    import google.colab  # noqa: F401
    if Path('/dev/nvidia0').exists() or os.environ.get('COLAB_TPU_ADDR'):
        raise RuntimeError('Select a CPU runtime')


class FrozenModelAdapter:
    """Prediction-only interface; holds full fitted pipelines and class decoders."""

    def __init__(self, manifest: VerifiedFrozenManifest, bundles: tuple[object, ...]):
        if len(bundles) != len(DOMAIN_ORDER):
            raise ValueError('Exactly four verified model bundles are required')
        self._manifest = manifest
        self._bundles = bundles

    @property
    def manifest(self) -> VerifiedFrozenManifest:
        return self._manifest

    def predict(self, feature: FeatureVector) -> FrozenPrediction:
        _require_colab_cpu()
        import numpy as np
        spec = self._manifest.model_for(feature.domain)
        if feature.names != spec.feature_names:
            raise ValueError('Feature order differs from frozen model metadata')
        index = DOMAIN_ORDER.index(feature.domain)
        bundle = self._bundles[index]
        codes = np.asarray(bundle['pipeline'].predict(np.asarray([feature.values], dtype=float))).reshape(-1)
        if len(codes) != 1:
            raise ValueError('Frozen pipeline returned an unexpected batch width')
        return FrozenPrediction(feature.domain, spec.algorithm,
                                decode_ordinal(spec.ordinal_classes, codes[0]), spec.checkpoint_sha256)


def load_verified_models(manifest: VerifiedFrozenManifest, *, enable_inference: bool = False) -> FrozenModelAdapter:
    """Future Colab entry point. Never invoked on the remote artifacts in Phase 3B."""
    if not enable_inference:
        raise PermissionError('Explicit inference enablement is required')
    _require_colab_cpu()
    if inspect_frozen_run(manifest.run_root) != manifest:
        raise ValueError('Frozen run changed since inspection')
    recorded = _read_json(_within(manifest.run_root, 'training/environment.json'))
    if not isinstance(recorded.get('packages'), dict):
        raise ValueError('Missing dependency provenance for frozen models')
    for package in ('numpy', 'scipy', 'scikit-learn', 'joblib'):
        if importlib.metadata.version(package) != recorded['packages'].get(package):
            raise RuntimeError(f'{package} version differs from the training environment')
    if recorded.get('python') != '.'.join(str(n) for n in __import__('sys').version_info[:3]):
        raise RuntimeError('Python version differs from the training environment')
    import joblib
    bundles = []
    for spec in manifest.models:
        if _sha256(spec.pipeline_path) != spec.checkpoint_sha256:
            raise ValueError('Checkpoint changed before read-only load')
        bundle = joblib.load(spec.pipeline_path)  # Trusted, hash-verified artifact only.
        if not isinstance(bundle, dict) or set(bundle) != {'pipeline', 'ordinal_classes'}:
            raise ValueError('Unexpected model bundle shape')
        if tuple(bundle['ordinal_classes']) != spec.ordinal_classes:
            raise ValueError('Ordinal decoder mismatch')
        pipeline = bundle['pipeline']
        if not hasattr(pipeline, 'named_steps') or tuple(pipeline.named_steps) != (
                ('imputer', 'scaler', 'model') if spec.domain is Domain.TS else ('imputer', 'model')):
            raise ValueError('Fitted preprocessing pipeline differs from frozen design')
        expected_class = 'SVC' if spec.domain is Domain.TS else 'RandomForestClassifier'
        if type(pipeline.named_steps['model']).__name__ != expected_class:
            raise ValueError('Model class differs from selected algorithm')
        bundles.append(bundle)
    return FrozenModelAdapter(manifest, tuple(bundles))
