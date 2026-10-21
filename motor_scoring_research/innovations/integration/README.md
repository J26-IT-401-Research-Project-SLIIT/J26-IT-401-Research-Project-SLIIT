# Phase 3E-A: read-only Colab integration preflight

Use [PP1_TRUST_RECAP_Read_Only_Colab_Preflight.ipynb](../notebooks/PP1_TRUST_RECAP_Read_Only_Colab_Preflight.ipynb)
in a Google Colab **CPU** runtime through VS Code's Colab extension. Upload
the code-only ZIP built for this phase, mount Drive and manually run the cells.
The `RUN_READ_ONLY_PREFLIGHT` switch defaults to `False`; this preparation
phase does not execute the Colab notebook.

The archive holds only the innovations Python modules/configs, the unchanged
reference engine source, the frozen training source and its Colab requirements.
It contains no case CSVs, dataset, joblib file, model checkpoint, final-test
record or evaluation report. Each member's SHA256 is locked in the archive's
`preflight_bundle_manifest.json`. Extraction verifies the lock and refuses
unsafe paths or an existing restore directory.

## Source-derived frozen artifact schema

The training module's `train()` writes 16 paths
`training/<domain>/<algorithm>/pipeline.joblib`, companion
`preprocessor.joblib`, `estimator.joblib`, `metadata.json` and `cv.json`.
Each `pipeline.joblib` is a dictionary with `pipeline` and `ordinal_classes`.
The pipeline owns the median imputer; SVM also owns StandardScaler. Its
estimator predicts encoded indices, which are decoded through
`ordinal_classes`. No such bundle is deserialized in preflight.

`freeze()` writes `selected_models.json` mapping domain to `algorithm`,
`pipeline`, `pipeline_sha256`, `features`, `target` and `metadata`. It writes
`FROZEN.json` with `files` (relative-path SHA256 lock), `selected`,
`candidate_count=16`, `test_predictions_accessed=false`,
`refit_after_validation=false` and the synthetic-imitation claim. The
pre-test lock includes `validation/selection.json`,
`training/environment.json` and all 16 candidate model files. The expected
selection is TS/SVM, BS/RF, RS/RF and GI/RF, giving selected paths:

```
training/TS/SVM/pipeline.joblib
training/BS/RF/pipeline.joblib
training/RS/RF/pipeline.joblib
training/GI/RF/pipeline.joblib
```

The [frozen adapter](../frozen_adapter.py) checks these paths, frozen hashes,
selected algorithms, metadata feature order and ordinal classes from JSON
and bytes only. The preflight also compares recorded Python/package versions
and checks the pinned original engine/training source hashes and AST signatures.
It does not import the original engine, open the `test/` directory or invoke
the final-test entry point. A missing file or package provenance is BLOCKED;
a mismatched hash or incompatible frozen metadata is FAIL.

## Deliberate limits

Fitted preprocessing, estimator classes and bundle decoder values cannot be
verified without deserializing trusted joblib bytes. They remain BLOCKED until
a later explicit approval and matching Colab environment. Likewise, a golden
transport fixture for the teammate's numeric fusion input has not been found;
fusion-transport compatibility is BLOCKED. The preflight's overall status will
therefore normally be BLOCKED even when every read-only check passes.

The [complete-session bridge](feature_bridge.py) is prepared for later approved
use. It validates TS/BS/RS/GI together, passes their original raw arrays and
tap times to `extract_features(..., fs=100)` and maps exact existing output keys
to frozen feature order. It never pads missing tasks or turns undefined original
features into values. The read-only notebook does not call this bridge or any
predictor. Real-device scores and MPS remain null.

The old final test and validation results are not thresholds or design inputs.
The notebook mismatch noted in the Phase 3A audit remains untouched; the
preflight neither rewrites that lock nor claims it passes.
