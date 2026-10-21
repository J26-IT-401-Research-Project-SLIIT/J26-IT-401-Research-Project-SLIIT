"""Exact four-recording adapter to the unchanged PP1 feature extractor.

Merely importing this module performs no extraction or dependency import.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from ..contracts import DOMAIN_ORDER, FEATURE_ORDER, Domain, FeatureVector, SensorSeries, TapSeries
from ..perturbations import validate_perturbed
from ..frozen_adapter import _require_colab_cpu

ENGINE_SHA256 = 'c39620fe8738ca9217b1b7f9d75141e0f2ab9a7b49d22ebd63b6234202c437e4'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_reference_engine(engine_path: Path | str, *, enable_extraction: bool = False):
    """Separate explicit gate; do not call in the read-only preflight notebook."""
    if not enable_extraction:
        raise PermissionError('Reference-engine import/extraction needs later approval')
    _require_colab_cpu()
    path = Path(engine_path)
    if sha256(path) != ENGINE_SHA256:
        raise ValueError('Reference engine SHA256 mismatch')
    spec = importlib.util.spec_from_file_location('pp1_innovations_reference_engine', path)
    if spec is None or spec.loader is None:
        raise ValueError('Reference engine cannot be imported')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CompleteSessionFeatureBridge:
    """Calls engine.extract_features once with four genuine, validated inputs.

    No partial extraction, dummy task, threshold scoring or imputation is used.
    Any undefined original-engine feature remains unavailable, not zero-filled.
    """

    def __init__(self, engine):
        if not callable(getattr(engine, 'extract_features', None)):
            raise ValueError('Original engine must provide extract_features')
        self._engine = engine

    def __call__(self, raw: tuple[object, ...]) -> tuple[FeatureVector, ...]:
        if len(raw) != 4:
            raise ValueError('Original extractor requires all four recordings')
        for domain, recording in zip(DOMAIN_ORDER, raw):
            validate_perturbed(recording, domain)
        ts, bs, rs, gi = raw
        if not (isinstance(ts, SensorSeries) and isinstance(bs, TapSeries) and
                isinstance(rs, SensorSeries) and isinstance(gi, SensorSeries)):
            raise ValueError('Raw recording types differ from PP1 protocol')
        original = self._engine.extract_features(ts.xyz, bs.tap_time_s, rs.xyz, gi.xyz, fs=100)
        if not isinstance(original, dict) or set(original) != {d.value for d in DOMAIN_ORDER}:
            raise ValueError('Original engine feature domains changed')
        output = []
        for domain in DOMAIN_ORDER:
            values = original[domain.value]
            names = FEATURE_ORDER[domain]
            keys = tuple(name.removeprefix(domain.value + '_') for name in names)
            if not isinstance(values, dict) or set(values) != set(keys):
                raise ValueError(f'Original engine feature keys changed for {domain.value}')
            if any(values[key] is None for key in keys):
                raise ValueError(f'Original engine returned an undefined {domain.value} feature')
            output.append(FeatureVector(domain, names, tuple(float(values[key]) for key in keys)))
        return tuple(output)
