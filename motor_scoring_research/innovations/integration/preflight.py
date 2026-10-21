"""Read-only frozen-run preflight: JSON, SHA256, AST and package metadata only."""
from __future__ import annotations

import ast
import importlib.metadata
import json
import platform
from pathlib import Path

from ..contracts import ALGORITHMS, DOMAIN_ORDER, FEATURE_ORDER
from ..frozen_adapter import inspect_frozen_run
from .feature_bridge import ENGINE_SHA256, sha256

TRAINING_SHA256 = 'c46281d18c9cc4b7306c241cfd625c88e6df1de040c5f9e545511be37613edf2'
EXPECTED_ENGINE_ARGS = ('ts_acc', 'bs_taps', 'rs_gyro', 'gi_acc', 'fs')
PACKAGE_NAMES = ('numpy', 'scipy', 'scikit-learn', 'xgboost', 'catboost', 'joblib')


def check(status: str, detail: str, **facts) -> dict:
    if status not in ('PASS', 'FAIL', 'BLOCKED'):
        raise ValueError('Unknown preflight status')
    return {'status': status, 'detail': detail, **facts}


def runtime_information() -> dict:
    """No package import, installation, model load or device activity."""
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'packages': packages}


def discover_run(run_root: Path | str) -> dict:
    """Check only expected top-level control files; never walk test results."""
    root = Path(run_root)
    if not root.is_dir():
        return check('BLOCKED', 'Frozen Drive run is unavailable', run_root=str(root))
    expected = ('FROZEN.json', 'selected_models.json', 'validation/selection.json',
                'training/environment.json', 'training/TRAINING_COMPLETE.json')
    missing = [name for name in expected if not (root / name).is_file()]
    if missing:
        return check('BLOCKED', 'Required manifest or environment file is absent', missing=missing)
    return check('PASS', 'Frozen-run control files found', files=list(expected))


def verify_frozen(run_root: Path | str) -> dict:
    """Hash frozen bytes and read metadata; no joblib deserialization."""
    if discover_run(run_root)['status'] != 'PASS':
        return check('BLOCKED', 'Frozen manifest discovery incomplete')
    try:
        manifest = inspect_frozen_run(run_root)
    except FileNotFoundError as exc:
        return check('BLOCKED', 'Frozen referenced file missing', path=str(exc.filename))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return check('FAIL', f'Frozen manifest or hash verification failed: {type(exc).__name__}: {exc}')
    return check('PASS', 'Pre-test frozen bytes, selection, metadata, features and ordinal decoder verified',
                 frozen_sha256=manifest.frozen_sha256,
                 selected_models_sha256=manifest.selected_models_sha256,
                 selected={spec.domain.value: {'algorithm': spec.algorithm,
                   'pipeline': spec.pipeline_path.relative_to(manifest.run_root).as_posix(),
                   'pipeline_sha256': spec.checkpoint_sha256,
                   'features': list(spec.feature_names),
                   'ordinal_classes': list(spec.ordinal_classes)} for spec in manifest.models},
                 deserialized=False)


def verify_dependencies(run_root: Path | str) -> dict:
    path = Path(run_root) / 'training/environment.json'
    if not path.is_file():
        return check('BLOCKED', 'Training environment manifest unavailable')
    try:
        recorded = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return check('FAIL', f'Unreadable training environment: {exc}')
    if not isinstance(recorded, dict) or not isinstance(recorded.get('packages'), dict):
        return check('BLOCKED', 'Training package-version provenance missing')
    current = runtime_information()
    mismatches = {name: {'recorded': recorded['packages'].get(name), 'current': current['packages'][name]}
                  for name in PACKAGE_NAMES if not recorded['packages'].get(name) or
                  recorded['packages'][name] != current['packages'][name]}
    if not recorded.get('python') or recorded['python'] != current['python']:
        mismatches['python'] = {'recorded': recorded.get('python'), 'current': current['python']}
    if mismatches:
        return check('BLOCKED', 'Runtime differs from recorded training environment; do not deserialize',
                     mismatches=mismatches)
    return check('PASS', 'Recorded Python and package versions match', runtime=current)


def verify_engine_source(engine_path: Path | str, training_source: Path | str) -> dict:
    """Static hash/AST check; neither source module is imported or executed."""
    engine_path, training_source = Path(engine_path), Path(training_source)
    missing = [str(path) for path in (engine_path, training_source) if not path.is_file()]
    if missing:
        return check('BLOCKED', 'Engine or training source missing from code bundle', missing=missing)
    hashes = {'engine': sha256(engine_path), 'training': sha256(training_source)}
    if hashes != {'engine': ENGINE_SHA256, 'training': TRAINING_SHA256}:
        return check('FAIL', 'Pinned source SHA256 mismatch', hashes=hashes)
    try:
        tree = ast.parse(engine_path.read_text(encoding='utf-8'))
        training_tree = ast.parse(training_source.read_text(encoding='utf-8'))
    except (SyntaxError, UnicodeError) as exc:
        return check('FAIL', f'Source cannot be statically parsed: {exc}')
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    feature = functions.get('extract_features')
    if feature is None or tuple(arg.arg for arg in feature.args.args) != EXPECTED_ENGINE_ARGS:
        return check('FAIL', 'Original complete-session extractor signature differs')
    if not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
               node.func.id == 'extract_features' for node in ast.walk(functions['assess_motor'])):
        return check('FAIL', 'Original assessment no longer calls extractor')
    training_assignments = {target.id: node.value for node in training_tree.body
                            if isinstance(node, ast.Assign) for target in node.targets
                            if isinstance(target, ast.Name)}
    try:
        declared = ast.literal_eval(training_assignments['FEATURES'])
    except (KeyError, ValueError, TypeError) as exc:
        return check('FAIL', f'Training feature schema is not statically readable: {exc}')
    if declared != {d.value: list(FEATURE_ORDER[d]) for d in DOMAIN_ORDER}:
        return check('FAIL', 'Training feature order differs from innovation contract')
    return check('PASS', 'Pinned original engine and training source match complete-session signature and feature order',
                 sha256=hashes, engine_api='extract_features(ts_acc, bs_taps, rs_gyro, gi_acc, fs=100)',
                 source_executed=False)


def contract_limits() -> dict:
    return {
        'selected_algorithms': check('PASS', 'Expected selection is fixed in the adapter',
                                     algorithms={d.value: ALGORITHMS[d] for d in DOMAIN_ORDER}),
        'preprocessing_design': check('PASS', 'Training source specifies median imputer; TS SVM also StandardScaler; pipelines save encoder bundle'),
        'fitted_preprocessing_and_decoder': check('BLOCKED', 'Fitted joblib internals require later explicitly approved deserialize'),
        'fusion_transport': check('BLOCKED', 'No golden numeric fusion transport fixture is available'),
        'real_device_contract': check('PASS', 'Real-device scores and MPS remain null; preflight performs no scoring'),
    }


def report_status(checks: dict[str, dict]) -> str:
    statuses = {entry['status'] for entry in checks.values()}
    return 'FAIL' if 'FAIL' in statuses else 'BLOCKED' if 'BLOCKED' in statuses else 'PASS'
