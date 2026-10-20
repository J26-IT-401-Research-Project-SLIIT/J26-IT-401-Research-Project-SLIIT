# PP1 V3 Phase 2B — preparation only

**No local fitting, full-data inference, or data generation. No clinical claims.**

Open [PP1_V3_Train_Compare_16_Models.ipynb](notebooks/PP1_V3_Train_Compare_16_Models.ipynb)
in VS Code and select the official Google Colab extension's CPU runtime. The
notebook is unexecuted; both training and final-test approval switches default FALSE.

## Existing data, immutable preparation

The user reports 240 complete V3 sessions and a passing Colab audit, stored at:

`/content/drive/MyDrive/PP1_V3_runs/PP1_Synthetic_Motor_Dataset_V3_20261020T210904Z_b8c13141`

This remote backup is not available in the local Windows workspace. We inspected
the existing generator and frozen protocol, not the remote integrity report. The
Colab preflight reads that actual report and rejects any mismatch before fitting.
No dataset is generated, copied back over, relabeled or resplit by this package.

The separate training package has its own [hash lock](package_lock.json). Original
V3 preparation files, preparation lock, generator configuration, source manifests,
engine and V2 remain unchanged. The [protocol copy](configs/frozen_evaluation_protocol.json)
is preserved verbatim, including its historical `training_implemented: false` field;
this new package implements that design without rewriting the historical document.

Local preservation check found that the old generation notebook is now named
`Notebook_google_colab.ipynb`, while its old lock retains the previous filename.
Its content hash still matches. This existing path mismatch is left untouched;
training does not call that old local preparation verifier. Colab checks the
generated dataset's archived inputs and the new training package separately.

The actual writer creates targets `TS`, `BS`, `RS`, `GI`, `demo_MPS`, rather than
V2's longer `_demo` names. [Input contract](configs/input_contract.json) and runtime
mapping checks enforce the real schema. Each domain only uses its own engine
features; identifiers, targets and provenance never enter the predictor matrix.

## Setup and execution after approval

1. Upload the new `PP1_V3_Training_Code.zip` to Google Drive My Drive. This is code
   only; do not upload/regenerate 240 sessions again. Open the local notebook in VS Code.
2. Choose Colab CPU and run Cell 1. It verifies the runtime before installing the
   six pinned packages in [requirements-colab.txt](requirements-colab.txt).
3. Mount Drive and restore code to a fresh `/content/PP1_V3_training_code/` directory.
   The notebook validates ZIP paths, regular-file types, duplicates, CRC and package
   hashes. It uses the existing data folder directly, read-only. Only output runs
   and separate backup/test-registry directories are written.
4. Run preflight: verify COMPLETE/integrity status, exact 240 session membership,
   168/36/36 assignments against the frozen plan, code/config/engine hashes, raw
   file hashes, features, target values, reference MPS, and actual class support.
5. Review frozen mappings/search spaces. Only after explicit user approval set
   `APPROVE_TRAINING=True`. Run training, validation, and freeze cells in order.
6. The frozen checkpoint is backed up to a new Drive folder BEFORE test evaluation.
   Only after separate explicit approval set `APPROVE_FINAL_TEST=True`.
7. Final test claims a persistent Drive registry entry keyed by the dataset session
   manifest hash before predicting. It refuses subsequent attempts, even from a new
   run ID. Interrupted attempts remain claimed: preserve the marker and investigate;
   there is no automatic retry or override. This is an accidental-reuse guard, not a
   security boundary against intentional manual editing or concurrent Drive clients.
8. Back up final outputs to a second, new Drive directory. Hash comparisons verify
   copies. Existing outputs, frozen checkpoints and registry entries are never overwritten.

## Training design

Four algorithms x four domains = 16 retained candidates. Each algorithm has three
prespecified configurations (maximum 144 CV fold fits + 16 train-only final fits
when every domain supports three folds). The [search spaces](configs/search_spaces.json)
are frozen in the package and copied to the run before fitting.

Training CV is stratified by observed domain score with identical fold membership
across algorithms within each domain. Use 3 folds, reduce to 2 for minimum observed
support 2, and skip search/use configuration 0 for singleton training classes.
If a training domain has fewer than two observed classes, the comparison stops
rather than substituting a different algorithm. Absent classes are reported and
not invented. Fold-local encoding maps observed ordinal labels to contiguous
integers for model compatibility and decodes them BEFORE computing ordinal MAE.

Every fold fits its own median imputer and, for SVM, StandardScaler. Complete V3
inputs must be finite; the pipeline still owns its preprocessing. Hyperparameters
use mean fold MAE, then mean fold macro F1, then ascending configuration index,
which is ordered from simpler to more complex before fitting. CV reports include
all configurations, fold metrics, and pooled out-of-fold predictions. They are
selection estimates, not nested-CV unbiased performance claims.

Each of the 16 candidates is fitted on training sessions only. Validation chooses
one algorithm per domain using MAE, macro F1, then RF/SVM/XGB/CatBoost. No additional
search and no refit after validation. All models, preprocessors, ordinal decoders,
features, configuration files, source snapshots, dependency versions and decisions
are hashed before test prediction. The final test evaluates all frozen candidates,
but the selected four-model system remains the validation-selected system.

RF/XGBoost/CatBoost use one training thread and CPU; SVM uses RBF without probability
calibration. No neural networks or GPU packages are required. No MPS regressor is
trained: the four predicted scores are summed and divided by 16.

## Reports and scientific limits

Each candidate has fixed-5-class macro F1 (zero division=0), exact accuracy,
within-1 accuracy, balanced accuracy over present true classes, MAE, 5x5 confusion
matrix, all class supports and sample size. Validation/test reports include Wilson
95% intervals for exact/within-1 accuracy and class recall, plus 2,000-replicate
session-bootstrap MAE intervals. MPS MAE has a session-bootstrap interval. These
intervals condition on this generator/model/sample and do not estimate clinical or
source-independent uncertainty. They are not multiplicity-adjusted; rare-class
and small-sample uncertainty must be acknowledged. Macro F1 and balanced accuracy
do not receive separate intervals in this version.

The unchanged PP1 engine is rerun on validation/test raw recordings at the relevant
stage as a target-defining consistency baseline. It must reproduce the synthetic
targets exactly. That is not independently learned accuracy. The preflight hashes
test raw bytes and inspects labels/support solely for integrity; no test predictions
are made before the freeze and separate test gate.

Only 36 validation and 36 test sessions exist. Models imitate arbitrary rules on
their own deciding features, so high agreement does not establish real-world
clinical performance. Do not change data, settings or model choices after test.
TS and RS remain sensor proxies; RS is not passive rigidity, BS has no amplitude
measurement, and MPS is not Parkinson probability. No real-device inference API is
introduced; future real-device mode must report features only until calibrated.
The optional generator-shift set is not supplied or evaluated by this package.

## Output layout

Each run has preflight/mapping/search-space reports, dataset hash snapshot and a
frozen source package. `training/<domain>/<algorithm>/` contains CV reports and OOF
predictions, `pipeline.joblib` (including ordinal decoder), `preprocessor.joblib`,
`estimator.joblib`, and metadata. Keep model components together; the standalone
estimator predicts encoded labels. `validation/` contains all comparisons,
confusion matrices, per-session predictions, selection and MPS reports.
`FROZEN.json` locks all artifacts before `test/` is created. The latter contains
all candidate metrics, selected-system predictions/MPS and final report.

Joblib files are trusted-run artifacts only; never load untrusted serialized models.
Stages refuse overwrites. Failed fitting/validation runs are not automatically
resumed. Preserve logs and rerun only the same reviewed design in a new run directory;
test attempts cannot be silently repeated.

## Local verification

`python -B -m unittest discover -s motor_scoring_research/v3/training/tests -v`

Tests use tiny hand-written tables/metric fixtures and file guards. They never
call model fitting or full-dataset inference, install packages, or read the remote
dataset. Static checks validate notebook cells and disabled gates. End-to-end
library integration and actual class supports remain Colab verification tasks.

Implementation references: scikit-learn documents fold-local preprocessing through
[pipelines](https://sklearn.org/stable/modules/compose.html); CPU/thread configuration
uses the [XGBoost Python API](https://xgboost.readthedocs.io/en/release_3.0.0/python/python_api.html)
and [CatBoost training parameters](https://catboost.ai/docs/en/references/training-parameters/).
