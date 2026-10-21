# Phase 3E-B: frozen-model compatibility and integration plan

Status: **prepared only**. No package was installed, checkpoint deserialized,
model queried, signal extracted, dataset opened or benchmark executed in this
phase. The user-reported Phase 3E-A Colab preflight passed frozen discovery,
hashes, selection and source contracts. Its reported dependency check is BLOCKED.

## 1. Exact recorded-environment evidence

The unchanged [training source](../../v3/training/src/pp1_training.py) writes
`training/environment.json` in `train()` with this schema:

```json
{"python":"<platform.python_version()>",
 "packages":{"numpy":"...","scipy":"...","scikit-learn":"...",
             "xgboost":"...","catboost":"...","joblib":"..."},
 "threads":1}
```

`FROZEN.json.files["training/environment.json"]` locks its SHA256. The source
does **not** reveal the value of `python`; the exact training interpreter must
be read from the frozen Drive file. [environment_compatibility.py](environment_compatibility.py)
checks that lock and all seven version fields without importing a heavy package.
Its environment comparison blocks on any Python or package mismatch.

The six reported training pins match
[requirements-colab.txt](../../v3/training/requirements-colab.txt):

| Package | Recorded training | Reported current Colab | Result |
|---|---:|---:|---|
| NumPy | 2.2.6 | 2.1.3 | mismatch |
| SciPy | 1.15.3 | 1.16.3 | mismatch |
| scikit-learn | 1.6.1 | 1.6.1 | version match |
| XGBoost | 3.0.5 | 3.4.1 | mismatch |
| CatBoost | 1.2.8 | missing | missing |
| joblib | 1.4.2 | 1.6.0 | mismatch |
| Python | **not yet read** | 3.13.16 | BLOCKED |

The current runtime must not deserialize a checkpoint. The selected SVM/RF
pipelines especially require matching scikit-learn, NumPy, SciPy and joblib.
The training manifest also records XGBoost and CatBoost, so the proposed
reproduction gate requires **all six** exact versions. Matching package
versions cannot compensate for a different recorded Python patch version under
the existing adapter's exact-version policy.

Release evidence for CPython 3.13 Linux x86-64: the pinned
[NumPy 2.2.6](https://pypi.org/project/numpy/2.2.6/),
[SciPy 1.15.3](https://pypi.org/project/scipy/1.15.3/),
[scikit-learn 1.6.1](https://pypi.org/project/scikit-learn/1.6.1/) and
[CatBoost 1.2.8](https://pypi.org/project/catboost/1.2.8/) publish cp313
manylinux x86-64 wheels. [XGBoost 3.0.5](https://pypi.org/project/xgboost/3.0.5/)
lists Python 3.13 and a py3 manylinux x86-64 wheel.
[joblib 1.4.2](https://pypi.org/project/joblib/1.4.2/) has a universal
`py3-none-any` wheel and `Requires-Python >=3.8`, but its published classifiers
stop at 3.12; this is not affirmative tested-support evidence for 3.13.
Wheel availability establishes installability, **not** exact checkpoint
portability. The recorded Python value and a fresh-runtime compatibility check
remain decisive. The package index and Colab runtime may change; re-check
availability at setup time using binary wheels only, without building source.

Scikit-learn's [model persistence guidance](https://scikit-learn.org/stable/model_persistence.html)
says loading across different scikit-learn versions is unsupported and that
pickle/joblib loading can execute code. The original frozen checkpoint source
must be trusted in addition to matching SHA256; a hash alone proves identity
against the frozen manifest, not benign contents.

## 2. Proposed manually executed Colab CPU cells

These cells are a **future notebook design**, not executed or installed now.
All state-changing or executable-code-loading cells default to `False`.
Start from a **fresh Colab CPU runtime**, not the already verified session.
Do not use `Run all`.

1. **CPU and provenance read (read-only).** Mount Drive; set the user-reported
   final-run root; verify it lies under `/content/drive/MyDrive/PP1_V3_training_runs/`.
   Re-run the existing Phase 3E-A `inspect_frozen_run` hash/selection check.
   Independently read `training/environment.json` and compare its SHA256 to
   `FROZEN.json.files`. Print `python`, six package versions and `threads`.
   Do not read `test/` or `validation/` results other than frozen control JSON.

   ```python
   # Read-only cell, after Drive mount and restoring the verified code bundle.
   from motor_scoring_research.innovations.frozen_adapter import inspect_frozen_run
   import hashlib, json, platform
   from motor_scoring_research.innovations.integration import preflight
   manifest = inspect_frozen_run(FROZEN_RUN)  # hashes only; no joblib.load
   raw = (FROZEN_RUN / 'training/environment.json').read_bytes()
   frozen = json.loads((FROZEN_RUN / 'FROZEN.json').read_text())
   assert hashlib.sha256(raw).hexdigest() == frozen['files']['training/environment.json']
   recorded = json.loads(raw)
   print(recorded)  # Reveals the exact training Python patch version.
   print(preflight.verify_dependencies(FROZEN_RUN))
   ```

2. **Python and binary-wheel feasibility (read-only).** If recorded Python is
   not exactly the runtime's `platform.python_version()`, report BLOCKED and
   stop. Do not try to change Python by `pip`; select a matching Colab CPU
   runtime/image if available. Before any install, confirm a compatible binary
   wheel for every pin and Linux architecture; refuse source builds. If no
   matching environment is available, stop without deserializing.

3. **Fresh-runtime environment setup (future approval gate).** Set
   `APPROVE_ENV_INSTALL = False` and `FRESH_RUNTIME_CONFIRMED = False` by default.
   Require both to be changed deliberately after approval. Ensure no NumPy,
   SciPy, sklearn, XGBoost, CatBoost or joblib module has already been imported.
   Require exact Python equality **before** the install. Install the six exact
   recorded pins with `sys.executable -m pip install --only-binary=:all:`;
   fail if resolution would require a source build. Do not run inference.

   ```python
   APPROVE_ENV_INSTALL = False
   FRESH_RUNTIME_CONFIRMED = False
   assert APPROVE_ENV_INSTALL and FRESH_RUNTIME_CONFIRMED
   assert platform.python_version() == recorded['python']
   assert not any(name in sys.modules for name in
                  ('numpy','scipy','sklearn','xgboost','catboost','joblib'))
   pins = [f'{name}=={recorded["packages"][name]}' for name in
           ('numpy','scipy','scikit-learn','xgboost','catboost','joblib')]
   subprocess.run([sys.executable, '-m', 'pip', 'install',
                   '--only-binary=:all:', '--no-cache-dir', *pins], check=True)
   ```

   Restart the runtime after setup. Re-mount Drive, restore the verified code
   bundle, re-run all read-only manifest/hash checks and demand exact Python
   and six-package equality. Do not bypass any mismatch warning.

4. **Trusted local checkpoint staging (future approval gate).** Set
   `APPROVE_LOCAL_CHECKPOINT_STAGING = False`. After full pre-test manifest
   verification, copy **only** the four selected `pipeline.joblib` files to a
   new exclusive local `/content/PP1_verified_selected_<run-id>/` directory.
   Resolve each relative path from the already verified `selected_models.json`
   and restrict it to the frozen run. Compare Drive source and local-copy
   SHA256 to both `selected_models.json.pipeline_sha256` and `FROZEN.json.files`.
   Refuse overwrite, symlinks, duplicate paths, extra files or a hash mismatch.
   Keep the Drive originals read only. A new local copy requires a fresh hash
   check; do not reuse a stale copy across runtime restarts.

5. **Trusted deserialization and fitted-contract inspection (separate future
   approval gate).** Set `APPROVE_TRUSTED_DESERIALIZATION = False`. Re-check
   exact runtime versions, all frozen-control hashes and local SHA256 **just
   before** each `joblib.load(local_copy)`. Load no other paths. This is
   executable-code loading, so approval applies only to the known frozen run
   and the four locally verified selected copies. Then inspect without fitting
   or calling `predict`:

   - bundle keys exactly `pipeline`, `ordinal_classes`;
   - decoder equals frozen metadata; unique sorted integer labels in 0–4;
   - steps exactly `imputer, scaler, model` for TS/SVM and `imputer, model`
     for BS/RS/GI/RF;
   - fitted `n_features_in_` and imputer statistics width equal the frozen
     feature count; TS scaler fitted arrays have the same width;
   - estimator class is SVC for TS and RandomForestClassifier otherwise;
     fitted encoded `classes_` are integers `0..len(ordinal_classes)-1`;
   - recorded feature order exactly matches `FEATURE_ORDER` and metadata.

   Reject any undefined attribute or mismatch. Inspect structure only; no
   predictions or score claims. Do not refit, mutate or resave the bundles.

6. **Invented-input integration smoke (later, separate inference approval).**
   Set `APPROVE_SYNTHETIC_SMOKE_INFERENCE = False`. Construct exactly 1,000
   TS acceleration rows, at least two actual invented BS timestamps, 1,000 RS
   gyro rows and 18,000 GI acceleration rows on the 100-Hz nominal grid.
   Use small deterministic sinusoids and a fixed seed, not a V3 session.
   Validate all four raw objects, run the pinned original
   `extract_features(..., fs=100)` through `CompleteSessionFeatureBridge`,
   demand finite frozen-order features, then call the four verified pipelines
   once. Decode encoded outputs through each saved `ordinal_classes`; require
   four integers 0–4 and MPS `sum(scores)/16`. Run the same invented complete
   input in real-device mode with **no predictor call**, asserting scores and
   MPS are null. This is a software smoke test, not model evaluation. No
   TRUST perturbation loop or RECAP benchmark runs here.

7. **Report.** Emit PASS/FAIL/BLOCKED per gate and explicit code, environment,
   checkpoint and input identities. Keep the existing fusion numerical output
   unchanged. Do not access the historical test sessions or compare smoke
   scores with clinical or rule-generated labels.

## 3. Remaining blockers

- Exact `training/environment.json.python` value is unknown until the read-only
  Drive cell runs. This can invalidate the current Python 3.13.16 runtime even
  if all pinned wheels are available.
- The current Colab package versions are not the recorded environment. No
  checkpoint load is permitted there. A fresh compatible CPU runtime and an
  explicitly approved install are still required.
- Fitted preprocessing and decoder internals remain BLOCKED until trusted,
  locally copied and freshly hash-verified joblib deserialization is approved.
- Fusion transport remains BLOCKED: obtain an independently agreed golden JSON
  fixture for the teammate's numeric input before integration. Do not derive
  it from test results or insert TRUST/RECAP sidecars into the numeric vector.
- The historical final test has already been inspected and is excluded from
  compatibility tuning, smoke inputs and innovation benchmarks.

**Exact next step after approval:** start a fresh Colab CPU runtime, mount the
frozen Drive run read only, re-run the Phase 3E-A frozen hash check and print
the hash-verified `training/environment.json` `python` field. Stop immediately
if it differs from that runtime's exact Python version.
