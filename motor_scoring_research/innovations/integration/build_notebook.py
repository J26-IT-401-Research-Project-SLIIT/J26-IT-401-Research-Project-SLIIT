"""Generate the manually executed, read-only Colab preflight notebook."""
from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': source.splitlines(keepends=True)}


def notebook() -> dict:
    cells = [
        markdown('''# PP1 TRUST-MPS / RECAP-MPS: read-only Colab CPU integration preflight

Run cells manually in order. This notebook **does not** deserialize models,
extract signal features, infer scores, run TRUST probes or call RECAP.
The frozen Drive run is read only. Do not open its historical `test/` results.
Keep `RUN_READ_ONLY_PREFLIGHT = False` until explicitly approved to run this preflight.
'''),
        code('''# 1. Runtime information and CPU verification (no package installation).
import os, platform
from importlib import metadata
from pathlib import Path
print('Python:', platform.python_version())
print('Colab:', bool(os.environ.get('COLAB_RELEASE_TAG')))
print('GPU device:', Path('/dev/nvidia0').exists(), 'TPU:', bool(os.environ.get('COLAB_TPU_ADDR')))
assert os.environ.get('COLAB_RELEASE_TAG'), 'Select a Google Colab runtime'
assert not Path('/dev/nvidia0').exists() and not os.environ.get('COLAB_TPU_ADDR'), 'Select CPU runtime'
for name in ('numpy','scipy','scikit-learn','xgboost','catboost','joblib'):
    try: print(name, metadata.version(name))
    except metadata.PackageNotFoundError: print(name, 'MISSING')
'''),
        code('''# 2. Mount Google Drive; the frozen run remains untouched.
from google.colab import drive
drive.mount('/content/drive')
'''),
        code('''# 3. Upload PP1_Innovations_Preflight_Code_v1.zip, verify and restore code only.
# Extraction refuses absolute/traversal/symlink paths, duplicates and overwrites.
import hashlib, json, zipfile
from pathlib import Path, PurePosixPath
from google.colab import files
uploaded = files.upload()
assert set(uploaded) == {'PP1_Innovations_Preflight_Code_v1.zip'}, 'Upload exactly the named code ZIP'
bundle = Path('/content/PP1_Innovations_Preflight_Code_v1.zip')
assert bundle.is_file()
restore_root = Path('/content/PP1_preflight_workspace')
assert not restore_root.exists(), 'Refuse to overwrite an existing restore directory'
with zipfile.ZipFile(bundle) as archive:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    assert len(names) == len(set(names)) and len(names) <= 50, 'Duplicate or excessive ZIP entries'
    assert sum(info.file_size for info in infos) <= 2_000_000, 'Code ZIP unexpectedly large'
    assert archive.testzip() is None, 'ZIP CRC failure'
    assert 'preflight_bundle_manifest.json' in names
    manifest = json.loads(archive.read('preflight_bundle_manifest.json'))
    expected = set(manifest['files_sha256']) | {'preflight_bundle_manifest.json'}
    assert manifest['schema_version'] == 'pp1_preflight_code_bundle.v1' and set(names) == expected
    for info in infos:
        name = info.filename
        path = PurePosixPath(name)
        assert not info.is_dir() and not name.startswith('/') and '\\\\' not in name and ':' not in name
        assert all(part not in ('', '.', '..') for part in name.split('/'))
        assert (info.external_attr >> 16) & 0o170000 != 0o120000, 'Symlink forbidden'
        assert name == 'preflight_bundle_manifest.json' or name.startswith('motor_scoring_research/innovations/') or name in {
            'motor_scoring_research/reference_engine/pp1_motor_engine.py',
            'motor_scoring_research/v3/training/src/pp1_training.py',
            'motor_scoring_research/v3/training/requirements-colab.txt'}
        if name != 'preflight_bundle_manifest.json':
            assert hashlib.sha256(archive.read(info)).hexdigest() == manifest['files_sha256'][name], name
    restore_root.mkdir(exist_ok=False)
    for info in infos:
        target = restore_root.joinpath(*PurePosixPath(info.filename).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream: stream.write(archive.read(info))
print('Verified code files:', len(manifest['files_sha256']), 'restored at', restore_root)
'''),
        code('''# 4. Explicit read-only preflight gate. Change to True only after approval.
RUN_READ_ONLY_PREFLIGHT = False
assert RUN_READ_ONLY_PREFLIGHT, 'Stop: read-only Colab preflight has not been approved'
import sys
sys.path.insert(0, str(restore_root))
from motor_scoring_research.innovations.integration import preflight
run_root = Path('/content/drive/MyDrive/PP1_V3_training_runs/PP1_V3_training_20261020T214151Z_961db068_final')
engine_path = restore_root / 'motor_scoring_research/reference_engine/pp1_motor_engine.py'
training_source = restore_root / 'motor_scoring_research/v3/training/src/pp1_training.py'
checks = {}
'''),
        code('''# 5. Read-only checkpoint discovery: control files only, no test results.
checks['checkpoint_discovery'] = preflight.discover_run(run_root)
print(json.dumps(checks['checkpoint_discovery'], indent=2))
'''),
        code('''# 6. SHA256 verification of frozen files, selected algorithms and metadata.
# inspect_frozen_run reads/hashes frozen bytes; it never calls joblib.load.
checks['frozen_selection_and_hashes'] = preflight.verify_frozen(run_root)
print(json.dumps(checks['frozen_selection_and_hashes'], indent=2))
'''),
        code('''# 7. Recorded dependencies; static feature/preprocessing/decoder design checks.
checks['dependencies'] = preflight.verify_dependencies(run_root)
checks['engine_and_training_source'] = preflight.verify_engine_source(engine_path, training_source)
checks.update(preflight.contract_limits())
print(json.dumps({k: v['status'] for k, v in checks.items()}, indent=2))
'''),
        code('''# 8. Final PASS/FAIL/BLOCKED report, held only in Colab memory.
# Fitted preprocessing/decoder internals and fusion transport remain BLOCKED by design.
report = {'schema_version': 'pp1_innovations_read_only_preflight.v1',
          'overall_status': preflight.report_status(checks),
          'frozen_run': str(run_root), 'checks': checks,
          'checkpoint_deserialized': False, 'model_inference_run': False,
          'raw_recordings_read': False, 'historical_test_results_read': False}
print(json.dumps(report, indent=2, sort_keys=True))
'''),
    ]
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
        'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}


def write_notebook(path: Path | str) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refuse notebook overwrite: {path}')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(notebook(), stream, indent=1, ensure_ascii=False)
        stream.write('\n')
