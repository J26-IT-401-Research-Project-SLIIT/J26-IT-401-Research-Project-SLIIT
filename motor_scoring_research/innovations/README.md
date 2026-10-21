# TRUST-MPS diagnostics and RECAP-MPS recommendations (Phase 3D)

This package implements input/output contracts, CSV quality validation, a
read-only **interface** to the four frozen V3 classifiers, finite TRUST-MPS
perturbation diagnostics and a recommendation-only RECAP-MPS policy. It contains
no dataset generation, training, retest execution or inferred retest outcomes.
Nothing in this package loads a model on import. Phase 3A scientific assumptions
and later-phase boundaries are in the [design audit](../innovation_design/PHASE_3A_DESIGN_AUDIT.md).

## Active protocol

| Domain | Recording | CSV columns | Selected model | Frozen feature columns |
|---|---|---|---|---|
| TS | 10 s, 100 Hz tri-axis acceleration | `time_s,ax_ms2,ay_ms2,az_ms2` | SVM | `TS_tremor_rms_ms2` |
| BS | 10 s touch events | `tap_time_s` | RF | `BS_tap_count`, `BS_tap_rate_per_min`, `BS_tap_interval_std_s` |
| RS | 10 s, 100 Hz tri-axis gyro | `time_s,gx_rad_s,gy_rad_s,gz_rad_s` | RF | `RS_mean_gyro_peak_rad_s`, `RS_movement_peaks` |
| GI | 180 s, 100 Hz tri-axis acceleration | `time_s,ax_ms2,ay_ms2,az_ms2` | RF | `GI_step_candidates`, `GI_cadence_steps_min`, `GI_step_interval_std_s` |

`TS`, `BS`, `RS`, `GI` remain integers 0–4 for complete, valid **synthetic demo**
sessions. `MPS=(TS+BS+RS+GI)/16`. This is a synthetic demo index, not Parkinson
probability or validated clinical severity. RS is active wrist mobility, not
passive rigidity; BS has no amplitude channel.

## Interfaces

- [contracts.py](contracts.py) defines frozen `SensorSeries`, `TapSeries`,
  `AcquisitionContext`, `ValidationResult`, `FeatureVector`, `FrozenPrediction`,
  `AssessmentResult`, `PerturbationDiagnostics`, `MpsSensitivity`,
  `TrustDiagnostics`, `ReassessmentAdvice`, and future `TrustAnalyzer` / `RecapPolicy`
  protocols. There is no implicit conversion from missing values to a score.
- [quality.py](quality.py) offers `validate_task_csv(domain, path, context=None)`
  and `validate_session(paths, contexts=None)`. They return status/reason codes and
  immutable raw tuples; invalid or ambiguous inputs have `input=None`. Metadata
  is optional and separate from unchanged CSV fields.
- [assessment.py](assessment.py) offers `assemble_assessment(mode, validations,
  features=(), predictions=None)`. It never calls a model or extracts features.
  Real-device mode refuses demo predictions and returns features with scores and
  MPS null. Synthetic demo mode requires four valid inputs, four finite ordered
  feature vectors and four frozen predictions before it emits MPS.
- [frozen_adapter.py](frozen_adapter.py) offers `inspect_frozen_run(run_root)`:
  read-only JSON and SHA256 verification of the pre-test freeze, selected models,
  feature order, decoder and algorithm choices. It does not open test reports or
  deserialize joblib. `load_verified_models(manifest, enable_inference=True)` is a
  *future* explicit Colab CPU entry point: it re-verifies the run and dependencies,
  then loads trusted full fitted pipelines. It was not called in Phase 3B.
  `FrozenModelAdapter.predict(feature)` preserves preprocessing and ordinal decoding.
  It does not expose a fitting method.

The original [PP1 feature engine](../reference_engine/pp1_motor_engine.py) takes
all four raw arrays together. This foundation deliberately does **not** make a
partial session appear complete with dummy recordings. Future feature extraction
must call that engine only on a complete validated session. A separate exact-parity
adapter for partial real-device features would need approval and tests first.

The older project README describes a different proposed fusion vector. This
package does not export any new numeric fusion payload. The existing fusion
contract must be captured as a golden fixture before integration. All quality,
sensitivity and reassessment results belong in an optional internal sidecar.

## Quality policy and safety

Hard failures cover missing files, wrong headers/row widths, blank/nonnumeric or
nonfinite values, inconsistent timestamps, short/long IMU recordings, wrong declared
units and known acquisition failures. IMU timestamps are checked against the exact
100-Hz nominal grid with the V3 adapter's 1e-9-second absolute tolerance.

Slow or irregular tapping, intermittent movement, low wrist speed, and altered gait
are **not** hard acquisition failures. Empty taps with unknown capture status yield
`UNKNOWN_TAP_ACQUISITION`; verified empty/single-tap captures yield
`UNSUPPORTED_FEATURE` because interval std is undefined for the frozen BS feature
schema. Neither is passed to the model imputer or converted to bradykinesia score 4.
No events or samples are inserted, sorted, truncated, padded or interpolated.
When optional declared task session IDs disagree, the combined session is rejected.
Without such metadata, CSV content alone cannot prove all four tasks came from
the same acquisition session.

CSV format alone cannot prove device identity, sensor calibration, full-scale
settings, packet integrity or a clinically valid task. Optional acquisition status
is accepted only when independently available; absence is recorded as unknown.
Raw acceleration/gyro units must be verified upstream if supplied. Large motion or
no detectable peaks is not automatically a sensor defect. Current Phase 3B quality
checks are protocol/data checks, not a calibrated detector of subtle sensor faults.

`PerturbationDiagnostics` and `MpsSensitivity` store future modeled sensitivity
only. They cannot be relabeled as clinical confidence intervals. `ReassessmentAdvice`
can only recommend; its `automatic_execution` must remain false, and nominal repeat
durations are fixed at 10/10/10/180 seconds. No repeated reading is created here.
An invalid new observation must never overwrite a previous valid result.

## Phase 3C TRUST-MPS API and assumptions

`trust.analyze_trust(validations, mode, extractor, predictor=None, *,
config=None, seed=0)` returns `(AssessmentResult, json_ready_sidecar)`. The
extractor must accept exactly four complete raw task objects and return ordered
`FeatureVector`s using the original monolithic engine. The predictor must
provide `predict(feature) -> FrozenPrediction`; the verified read-only adapter
is the eventual Colab implementation. This phase only uses fake predictors.
No model file or original engine is imported by TRUST-MPS. The caller must
verify the frozen engine, model hashes, feature schemas, and decoder before
Colab inference. All actual feature extraction and model calls belong there.

The sidecar has `schema_version=trust_mps.v1`, mode, quality reason codes,
assumptions/config hash, seed, baseline demo scores/MPS when allowed,
per-domain/per-family valid and invalid probe counts, reason counts, ordinal
histograms, score-change fraction among valid probes, mean absolute score
drift, and the nearest **observed** score-changing amplitude. `boundary_status`
is `not_observed_within_finite_search` when no tested amplitude flips a score;
it does not certify a continuous robustness radius. Joint scenarios store
paired four-score outcomes and their MPS. `observed_min/max/width` includes
the unperturbed baseline and only complete joint scenarios. A separately
labeled rectangular outer envelope uses independent marginal extrema; it is
never described as a jointly observed outcome. The sidecar is separate from
the unchanged numeric fusion payload. `clinical_confidence` and
`clinical_correctness` are null.

[trust_engineering_v1.json](configs/trust_engineering_v1.json) fixes 11 local
domain probes plus two paired joint probes (maximum 32 attempts). Offsets,
noise and gains are raw sensor observation-channel disturbances; tap jitter
preserves every detected event and rejects order/window collisions. The
declared amplitudes are uncalibrated engineering examples, not MPU6050 or
capacitive device tolerances. Noise is Gaussian with the declared per-axis
standard deviation; offsets are common to three axes; gain sign is seeded.
The paired joint noise scenario maps its numeric value to the domain's units
and halves the gyro noise. BS is unchanged under joint IMU scenarios because
there is no justified coupling from IMU noise to touch timestamps. This
coupling choice is a modeling assumption, not an observed covariance.

Quantization is implemented only when a verified device conversion step is
passed to `apply_probe`; the default scenario library disables it. GI physical
sampling-clock perturbation is deliberately unsupported because the frozen
CSV contract requires nominal 100-Hz timestamps and no continuous reference
trajectory or measured clock error is available. Simply shifting the CSV
clock or interpolating observed real-device data would silently change the
input semantics. A future synthetic-observation operator needs a documented
continuous trajectory and separate tests before it can be enabled.

Invalid or incomplete sessions receive no dummy tasks, no extraction and no
predictions. In real-device mode, the complete-session extractor may return
features, but the predictor is never called and all demo scores, MPS and
score-sensitivity diagnostics remain null. A stable score only means no flip
in the tested nuisance library; it does not establish clinical correctness.

## Phase 3D RECAP-MPS API and policy

`recap.recommend_reassessment(assessment, trust=None, *, config=None,
remediation=None, prior_repeats=None, completed_rounds=0,
recording_time_budget_seconds=None, operator_permits_repeat=True)` returns
`(ReassessmentAdvice, json_ready_sidecar)`. `assessment` is the unchanged
`AssessmentResult`. `trust` is the existing `trust_mps.v1` sidecar or `None`.
No model, extractor, CSV writer or physical-device API is called.

`remediation` is optional external, independently verified acquisition state
per domain: `corrected`, `unresolved`, or `unknown` (default). A hard-invalid
recording is recommended for a repeat only when its acquisition problem has
been corrected. Persistent and ambiguous faults, including empty taps with
unknown capture status, receive a clarify/repair decision with zero recording
seconds. Existing quality reason codes remain unchanged. A domain's symptoms
or score alone never become an acquisition fault.

Acquisition actions precede sensitivity ranking. For otherwise valid complete
synthetic-demo sessions, RECAP validates the TRUST mode, baseline scores/MPS,
quality, per-family counts, ordinal histogram, and drift arithmetic. Missing,
zero-probe or malformed diagnostics mean insufficient evidence, not zero
uncertainty. Eligible domains with observed score changes are ranked by
`mean_absolute_modeled_score_drift / (16 * fixed_task_duration_seconds)`;
equal values follow the configured domain order. The mean is descriptive over
the finite, configured probe library. This ratio is a **modeled sensitivity
per recording second heuristic**, not expected improvement or clinical benefit.

[recap_policy_v1.json](configs/recap_policy_v1.json) defaults to one recommended
task per round, 180 recording seconds, one repeat per domain, two rounds per
session, one valid probe and one observed score change as minimum sensitivity
evidence. These are unevaluated engineering policy settings, not optimized
thresholds. The request may set a smaller recording-time budget; an operator
may stop further recommendations. The JSON-ready `recap_mps.v1` sidecar records
decision status, recommended tasks and fixed durations, each domain's quality
and eligibility reasons, modeled ranking when permitted, total recording time,
stopping reason, assumptions/config hash and explicit limitations. It records
`automatic_execution=false`, `retest_outcomes=null`, and
`score_replacement=false`. Recording time excludes setup, recovery and
computation. It never changes the existing fusion numeric input.

Real-device mode ignores score-dependent TRUST data entirely. It can recommend
a new capture only after independently confirmed acquisition correction; scores
and MPS remain unavailable. Before Colab integration, the actual frozen model
manifest/feature order must be verified, the exact complete-session extractor
bridge connected, and a separate training-side paired-repeat benchmark
preregistered. The inspected V3 final test must not be reused for policy tuning.

The [Phase 3E-A integration guide](integration/README.md) documents the
code-only archive, source-derived checkpoint schema, exact complete-session
bridge and manually executed [read-only Colab preflight notebook](notebooks/PP1_TRUST_RECAP_Read_Only_Colab_Preflight.ipynb).
Preflight does not deserialize a pipeline or run feature extraction.

## Checkpoint provenance and local limitation

The user-reported final artifacts live at
`/content/drive/MyDrive/PP1_V3_training_runs/PP1_V3_training_20261020T214151Z_961db068_final`.
The remote files are not in this workspace. Expected relative pipelines are
`training/TS/SVM/pipeline.joblib`, `training/BS/RF/pipeline.joblib`,
`training/RS/RF/pipeline.joblib`, and `training/GI/RF/pipeline.joblib`.
The first future Colab action is read-only manifest inspection; stop if actual
selection, hashes, feature schema, dependencies or decoder differ. Never run the
old final test again.

The Phase 3A audit found that the previous generation notebook was renamed while
its preparation lock retained its former path. It also found that the local
training notebook differs from its package-lock hash. Neither notebook nor lock
was changed here, and this package does not silently repair them. The actual
archived final-run manifests must be checked independently.

## Lightweight checks

From the workspace root:

```powershell
python -B -m unittest discover -s motor_scoring_research/innovations/tests -v
```

Tests create only temporary, invented CSVs and fake model-file bytes. No model
fitting, deserialization, inference, dataset processing, perturbation experiment,
hardware call or heavy local dependency installation occurs. Future model/feature
parity and full-data work belong in an approved Google Colab CPU phase.
