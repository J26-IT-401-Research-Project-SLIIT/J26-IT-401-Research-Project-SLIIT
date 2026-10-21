"""Build a code-only Colab bundle and manually executed notebook; never runs ML."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(__file__).resolve().parent
NOTEBOOK = BENCH / 'PP1_3F_B_TRUST_RECAP_Benchmark_CPU.ipynb'
ARCHIVE = ROOT / 'PP1_3F_B_Code_Inputs.zip'


def cell(kind: str, lines: str):
    return {'cell_type': kind, 'metadata': {}, 'source': lines.splitlines(keepends=True),
            **({'execution_count': None, 'outputs': []} if kind == 'code' else {})}


def build_notebook():
    cells = [
        cell('markdown', '# PP1 TRUST-MPS + RECAP-MPS benchmark (Colab CPU)\n'
             'Phase 3F-B implementation. Execute cells manually. Development only unless a separately approved, '
             'single-use audit is explicitly authorized. Synthetic demo-rule outcomes are not clinical validity.'),
        cell('code', "# 1. Fresh Google Colab CPU runtime: information only; no inference.\n"
             "import os, platform, importlib.metadata, importlib.util, pathlib\n"
             "assert pathlib.Path('/content').is_dir() and os.environ.get('COLAB_RELEASE_TAG')\n"
             "assert not pathlib.Path('/dev/nvidia0').exists() and not os.environ.get('COLAB_TPU_ADDR')\n"
             "print('Python:', platform.python_version())\n"
             "print({p: importlib.metadata.version(p) if importlib.util.find_spec(p.replace('-', '_')) else 'missing' for p in ('numpy','scipy','sklearn','joblib')})"),
        cell('code', "# 2. Mount Google Drive. Source V3 data and frozen models remain read-only.\n"
             "from google.colab import drive\n"
             "drive.mount('/content/drive')"),
        cell('code', "# 3. Restore the code-only ZIP into a fresh workspace; refuse unsafe paths and overwrites.\n"
             "from pathlib import Path\nimport zipfile\n"
             "BUNDLE = Path('/content/drive/MyDrive/PP1_3F_B_Code_Inputs.zip')\n"
             "ROOT = Path('/content/PP1_3F_B_workspace')\n"
             "assert BUNDLE.is_file() and not ROOT.exists()\n"
             "with zipfile.ZipFile(BUNDLE) as z:\n"
             "    assert z.testzip() is None\n"
             "    names = z.namelist()\n"
             "    assert len(names) == len(set(names))\n"
             "    for item in z.infolist():\n"
             "        name = item.filename\n"
             "        assert not item.is_dir() and not name.startswith(('/', '\\\\')) and '\\\\' not in name\n"
             "        assert all(part not in ('', '.', '..') for part in name.split('/')) and ':' not in name\n"
             "        assert (item.external_attr >> 16) & 0o170000 not in (0o120000,)\n"
             "    ROOT.mkdir()\n"
             "    for item in z.infolist():\n"
             "        target = ROOT / item.filename\n"
             "        target.parent.mkdir(parents=True, exist_ok=True)\n"
             "        with z.open(item) as source, target.open('xb') as sink:\n"
             "            import shutil; shutil.copyfileobj(source, sink)\n"
             "print('Restored', len(names), 'code/lock files to', ROOT)"),
        cell('code', "# 4. Runtime compatibility: inspect the frozen environment BEFORE installation/loading.\n"
             "import json, platform\n"
             "FROZEN = Path('/content/drive/MyDrive/PP1_V3_training_runs/PP1_V3_training_20261020T214151Z_961db068_final')\n"
             "DATASET = Path('/content/drive/MyDrive/PP1_V3_runs/PP1_Synthetic_Motor_Dataset_V3_20261020T210904Z_b8c13141')\n"
             "recorded = json.loads((FROZEN/'training/environment.json').read_text())\n"
             "print('Recorded Python:', recorded['python'], 'Current:', platform.python_version())\n"
             "print('Recorded packages:', recorded['packages'])\n"
             "assert recorded['python'] == platform.python_version(), 'STOP: use a compatible fresh Colab runtime'"),
        cell('code', "# 5. Install ONLY in Colab; pin to recorded training packages. Restart if pip requests it.\n"
             "import subprocess, sys\n"
             "wanted = {k: recorded['packages'][k] for k in ('numpy','scipy','scikit-learn','xgboost','catboost','joblib')}\n"
             "subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *[f'{k}=={v}' for k,v in wanted.items()]], check=True)\n"
             "print('Restart runtime if required, then rerun setup and integrity cells.')"),
        cell('code', "# 6. Fail-closed design, split, dependency and checkpoint hash preflight. No joblib load.\n"
             "import sys, importlib.metadata\nsys.path.insert(0, str(ROOT))\n"
             "from motor_scoring_research.innovations.benchmark.locks import verify_design, verify_frozen_models\n"
             "spec, parents = verify_design(ROOT)\n"
             "assert len(parents) == 168\n"
             "for name, version in wanted.items():\n"
             "    assert importlib.metadata.version(name) == version, f'Package mismatch: {name}'\n"
             "# Enter the SHA256 recorded by the prior trusted read-only preflight, never guess it.\n"
             "EXPECTED_SELECTED_MODELS_SHA256 = ''\n"
             "assert len(EXPECTED_SELECTED_MODELS_SHA256) == 64, 'STOP: supply trusted selected_models.json SHA256'\n"
             "frozen_manifest = verify_frozen_models(FROZEN, EXPECTED_SELECTED_MODELS_SHA256)\n"
             "print('PASS: design, 168-parent split, packages and frozen checkpoint hashes')"),
        cell('markdown', '## Development generation and evaluation\n'
             'The next cell is the **first inference/generation gate**. It defaults to FALSE. '
             'Only the 84 development parents are accessible through this cell.'),
        cell('code', "# 7. Separate explicit approval for development-only benchmark generation/evaluation.\n"
             "APPROVE_DEVELOPMENT_RUN = False\n"
             "assert APPROVE_DEVELOPMENT_RUN is True, 'STOP: development run is not approved'\n"
             "from datetime import datetime, timezone\n"
             "from motor_scoring_research.innovations.benchmark.runner import run\n"
             "stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')\n"
             "OUT = Path('/content/drive/MyDrive/PP1_V3_innovation_benchmark_runs') / f'development_{stamp}'\n"
             "OUT.parent.mkdir(parents=True, exist_ok=True)\n"
             "assert not OUT.exists()\n"
             "run(workspace=ROOT, dataset=DATASET, frozen_run=FROZEN, output=OUT,\n"
             "    selected_models_sha256=EXPECTED_SELECTED_MODELS_SHA256, approved_development=True, split='development')\n"
             "print('Development report:', OUT/'summary.json')"),
        cell('markdown', '## Single-use innovation audit: sealed\n'
             'Audit access requires a separately approved, frozen development/code decision lock, '
             'a fixed persistent Drive registry, and a fresh run. A claim is permanent even if execution fails. '
             'The audit cell deliberately remains disabled.'),
        cell('code', "# 8. Audit approval gate. DO NOT enable in Phase 3F-B.\n"
             "APPROVE_SINGLE_USE_AUDIT = False\n"
             "assert APPROVE_SINGLE_USE_AUDIT is True, 'STOP: audit set is sealed'\n"
             "# Only after independent review: copy the signed/reviewed DEVELOPMENT_LOCK.json\n"
             "# to the fixed registry and enter its SHA256 here. Never delete AUDIT_STARTED.json.\n"
             "DEVELOPMENT_LOCK_SHA256 = ''\n"
             "assert len(DEVELOPMENT_LOCK_SHA256) == 64\n"
             "from datetime import datetime, timezone\n"
             "from motor_scoring_research.innovations.benchmark.runner import run\n"
             "REGISTRY = Path('/content/drive/MyDrive/PP1_V3_innovation_audit_registry')\n"
             "assert REGISTRY.is_dir() and not (REGISTRY/'AUDIT_STARTED.json').exists()\n"
             "audit_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')\n"
             "AUDIT_OUT = Path('/content/drive/MyDrive/PP1_V3_innovation_benchmark_runs') / f'audit_{audit_stamp}'\n"
             "assert not AUDIT_OUT.exists()\n"
             "run(workspace=ROOT, dataset=DATASET, frozen_run=FROZEN, output=AUDIT_OUT,\n"
             "    selected_models_sha256=EXPECTED_SELECTED_MODELS_SHA256, split='audit',\n"
             "    approved_audit=True, registry=REGISTRY, development_lock_sha256=DEVELOPMENT_LOCK_SHA256)"),
    ]
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
        'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 5}
    with NOTEBOOK.open('x', encoding='utf-8') as stream:
        json.dump(notebook, stream, indent=1, ensure_ascii=False)
    return NOTEBOOK


def build_archive():
    innovations = ROOT / 'motor_scoring_research/innovations'
    include = list(innovations.glob('*.py'))
    include += list((innovations / 'configs').glob('*.json'))
    include += [innovations / 'integration/__init__.py',
                innovations / 'integration/feature_bridge.py']
    include += [path for path in (innovations / 'benchmark').iterdir()
                if path.suffix in ('.py', '.json', '.md', '.csv', '.ipynb') and path.is_file()]
    include += [ROOT / name for name in (
        'motor_scoring_research/reference_engine/pp1_motor_engine.py',
        'motor_scoring_research/v3/pp1_v3.py',
        'motor_scoring_research/v3/configs/simulation_v3.json',
        'motor_scoring_research/v3/manifests/preparation_lock.json',
        'motor_scoring_research/v3/training/configs/frozen_session_plan.csv',
        'motor_scoring_research/v3/training/src/pp1_training.py')]
    with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(include)):
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise ValueError('Code archive CRC failed')
        names = set(archive.namelist())
        required = {'motor_scoring_research/innovations/benchmark/preregistration_v1.json',
                    'motor_scoring_research/innovations/benchmark/PHASE_3F_A_DESIGN_LOCK.json',
                    'motor_scoring_research/reference_engine/pp1_motor_engine.py'}
        if not required.issubset(names) or any('/cases/' in n or n.endswith('.joblib') for n in names):
            raise ValueError('Code archive contains missing or forbidden payload')
    return ARCHIVE


if __name__ == '__main__':
    build_notebook()
    print(build_archive())
