# Phase 3A Design Audit: TRUST-MPS and RECAP-MPS

Date: 2026-10-21. Status: **design proposal, not implemented or evaluated**.

Only source inspection, hash comparisons and static Python parsing were performed.
No model was deserialized, fitted, refitted or queried. No dataset or perturbation
was generated. No final-test predictions/reports were read or rerun. This report
is the only new research artifact from Phase 3A; existing files are unchanged.

## 1. Evidence and immutable boundaries

Workspace: `D:/Motor Assesment + Portable device`.

| Verified local path | Role |
|---|---|
| [reference_engine/pp1_motor_engine.py](../reference_engine/pp1_motor_engine.py) | Original extraction, demo thresholds and real-mode behavior |
| [v3/pp1_v3.py](../v3/pp1_v3.py) | CSV adapter, feature naming, original generator and provenance |
| [v3/configs/evaluation_protocol_v3.json](../v3/configs/evaluation_protocol_v3.json) | Frozen model-selection and one-time test contract |
| [v3/training/src/pp1_training.py](../v3/training/src/pp1_training.py) | Pipeline bundle serialization, ordinal decoding, freeze and test registry |
| [v3/training/configs/input_contract.json](../v3/training/configs/input_contract.json) | Expected V3 backup, code/engine/config hashes |
| [v3/training/requirements-colab.txt](../v3/training/requirements-colab.txt) | Pinned training environment |
| [PADS + MJFF](../../PADS%20+%20MJFF) | Preserved V2; not an innovation calibration/evaluation source |
| [Parkinson_Motor_AI](../../Parkinson_Motor_AI) | Separate older PADS classification research; do not load these binary classifiers for motor scoring |

Verified SHA256:

- Original engine: `c39620fe8738ca9217b1b7f9d75141e0f2ab9a7b49d22ebd63b6234202c437e4`.
- V3 generator: `c659c2121b88cf3d41eaee2eb8161088a9b6199fbebb3a1bf748c036106e2df2`.
- Training module: `c46281d18c9cc4b7306c241cfd625c88e6df1de040c5f9e545511be37613edf2`.
- The older project's PP1 engine copy currently has the same engine hash, but the
  explicit `motor_scoring_research/reference_engine` path remains authoritative.

The user reports the selected models TS=SVM, BS=RF, RS=RF, GI=RF and final artifacts at:

`/content/drive/MyDrive/PP1_V3_training_runs/PP1_V3_training_20261020T214151Z_961db068_final`

These are **user-reported remote artifacts, not locally inspected checkpoints**.
Expected relative model locations, inferred from the verified serialization code:

| Domain | Relative path inside that final run |
|---|---|
| TS | `training/TS/SVM/pipeline.joblib` |
| BS | `training/BS/RF/pipeline.joblib` |
| RS | `training/RS/RF/pipeline.joblib` |
| GI | `training/GI/RF/pipeline.joblib` |

Before future inference, inspect only selection/manifests, dependency metadata and
hashes: `selected_models.json`, `FROZEN.json`, the four pipeline/metadata files and
`training/environment.json`. Do not open final-test predictions, call `final_test`,
or alter its persistent registry. Hash verification is not authorization to rerun
the original test. If the actual manifests differ, stop rather than substitute models.

The reported 94.44/100/100/97.22% domain accuracies, 33/36 joint agreement, and
0.005208333333333333 MPS MAE are historical synthetic benchmark results only.
They play **no role** in perturbation amplitudes, thresholds, weights or evaluation
sample selection. The previous 36 test sessions are permanently excluded here.

Local provenance observations:

- The preparation lock still names `Generate_and_Audit_V3.ipynb`, which is absent
  at that path; the prior notebook was renamed. Present locked preparation files match.
- The current training notebook differs from its package-lock hash. No inference
  is made about why, and no lock is rewritten. All other locked training files match.
- The root [README](../../README.md) contains older proposed GS/MS names and a
  weighted aggregate. That proposal is not the active PP1 contract. Phase 3 preserves
  TS/BS/RS/GI and `(TS+BS+RS+GI)/16` exactly.
- No implemented fusion transport/API schema was identified. Preserve the existing
  numeric payload and key names at its integration boundary; establish a golden
  contract fixture before dashboard/fusion integration. Do not insert diagnostics
  into the teammate's numeric feature vector.

## 2. Verified interfaces and architectural implications

### CSV and extraction contract

| Domain | CSV fields | Shape/time | Frozen extracted features |
|---|---|---|---|
| TS | `time_s,ax_ms2,ay_ms2,az_ms2` | 1,000 samples; 0–9.99 s; 100 Hz | `TS_tremor_rms_ms2` |
| BS | `tap_time_s` | Event list in [0,10), strictly increasing | `BS_tap_count`, `BS_tap_rate_per_min`, `BS_tap_interval_std_s` |
| RS | `time_s,gx_rad_s,gy_rad_s,gz_rad_s` | 1,000 samples; 0–9.99 s; 100 Hz | `RS_mean_gyro_peak_rad_s`, `RS_movement_peaks` |
| GI | `time_s,ax_ms2,ay_ms2,az_ms2` | 18,000 samples; 0–179.99 s; 100 Hz | `GI_step_candidates`, `GI_cadence_steps_min`, `GI_step_interval_std_s` |

`read_raw(folder, config)` validates exact CSV columns, finite measurements, row
counts and nominal IMU timestamps with absolute tolerance 1e-9 and relative tolerance
zero. It returns arrays without IMU timestamps. TRUST must inspect timestamps
before that information is discarded. Do not silently sort, interpolate, truncate,
pad, change units or resample to pass the current adapter.

`extract_features(ts_acc, bs_taps, rs_gyro, gi_acc, fs=100)` requires all four tasks:

- TS: RMS of `norm(acceleration)-9.81`; no tremor bandpass or de-meaning. Bias,
  gain and incidental movement can affect this proxy.
- BS: tap count, count*6 taps/min, population standard deviation of intervals.
  The original demo rule uses rate only, not irregularity. The frozen RF may use
  any of its three existing inputs; inspect actual model behavior without assuming it.
- RS: angular-speed magnitude; peaks separated by at least 45 samples, prominence
  0.2 rad/s; mean detected peak and count. No peaks yields a null mean, not a
  calibrated severe-rigidity observation.
- GI: acceleration magnitude, third-order 0.6–3 Hz Butterworth bandpass with
  `sosfiltfilt`, peaks separated by at least 40 samples and prominence 0.3 m/s².
  Count/3 is candidate cadence, not confirmed footfall cadence.

The original thresholds remain unchanged: TS .12/.24/.40/.70 m/s²; BS
210/165/120/75 taps/min; RS 2.4/1.8/1.2/.6 rad/s; GI 110/95/80/65 candidates/min.
Rule-threshold proximity is not the learned classifier's decision boundary.

`assess_motor(..., mode='real')` returns features and `scores=None, mps=None`.
Its demo mode invokes deterministic rules, not the frozen selected ML pipelines.
Do not accidentally replace the frozen ML baseline with this rule scorer.

### Model and output contract

Each saved pipeline bundle is a dictionary:

`{'pipeline': fitted_sklearn_Pipeline, 'ordinal_classes': [observed ordinal labels]}`.

The pipeline owns its fitted median imputer; SVM also owns a fitted StandardScaler.
The separate estimator predicts **encoded class indices**. Call the complete pipeline
on the exact ordered feature columns, then decode through `ordinal_classes`, as the
existing `predict` helper does. Never fit preprocessing again or treat encoded
indices as scores. Reject noninteger/out-of-range decoded predictions.

TS uses SVC with RBF and `probability=False`. Its decision function is not a
calibrated probability. RF vote fractions likewise are not clinical confidence.
The primary boundary diagnostic must therefore work through raw-domain perturbation
and the frozen predictor, not an invented common probability scale.

The training report exports per-session `TS_predicted` etc., `predicted_demo_MPS`
and `reference_demo_MPS`. Preserve the deployed numeric contract rather than rename
these report columns speculatively. Complete valid synthetic-demo sessions still
produce four integers 0–4 and the unchanged MPS. Diagnostics never reweight MPS.

The extraction interface is monolithic. First implementation should only call it
on complete, structurally valid sessions, keeping other three tasks fixed during
a one-domain perturbation. Invalid/missing tasks are reported before extraction;
never supply dummy recordings. For isolated-domain/partial real-device features,
a separately approved adapter may reproduce each domain's exact operators, with
parity tests against the original complete-session engine. Until parity passes,
report partial extraction as unavailable rather than invent values. No imputer is
used to convert invalid raw acquisition into a valid score.

## 3. Proposed architecture and APIs

Create a new sibling `motor_scoring_research/innovations/` after approval, outside
both the locked V3 preparation and training packages. Proposed locations:

| Proposed relative location | Responsibility |
|---|---|
| `src/contracts.py` | Immutable request/results, modes and status vocabulary |
| `src/frozen_adapter.py` | Verified selected-model loading, exact feature order and decoder |
| `src/quality.py` | Task-specific acquisition validation and reasons |
| `src/perturbations.py` | Versioned nuisance scenarios, provenance and finite budgets |
| `src/trust.py` | Score stability, boundary sensitivity, joint MPS propagation |
| `src/recap.py` | Recommendation only; no acquisition or score replacement |
| `configs/` | Preregistered physical assumptions, scenario library and policy |
| `benchmarks/` | Separate plans/manifests; no V3 edits |
| `notebooks/` | Colab CPU artifact inspection, diagnostic and paired-repeat evaluations |
| `tests/`, `docs/`, `results/<run_id>/` | Small contracts/tests, limitations, append-only results |

Flow:

```text
unchanged CSVs + optional acquisition context
        -> quality assessment
        -> valid synthetic demo: exact extractor -> frozen models -> unchanged scores/MPS
        -> real device: exact extractor -> features only, scores/MPS null

quality + raw data + frozen-model adapter (synthetic demo only)
        -> TRUST diagnostics sidecar
        -> RECAP recommendation sidecar
```

Optional context is internal metadata, not new sensor hardware or required CSV
columns: capture-complete status, verified units/ranges, timestamp resolution,
sequence continuity if actually available, operator task confirmation, known
acquisition failure, session/task association. Missing metadata is `unknown`, never
assumed affirmative. Synthetic provenance must not leak into classifier predictors.

Proposed API signatures (design notation, not implementation):

```text
load_frozen_bundle(run_root, expected_selection, accepted_hash_manifest) -> FrozenModels
validate_task(domain, raw_csv, acquisition_context, quality_spec) -> QualityReport
extract_existing_features(valid_session, engine_hash) -> FeatureBundle
predict_demo(valid_features, FrozenModels) -> ExistingScorePayload
analyze_trust(valid_session, mode, FrozenModels?, nuisance_spec, seed) -> TrustReport
recommend_reassessment(quality, trust?, constraints, history) -> ReassessmentAdvice
```

`QualityReport`: domain, `valid|suspect|invalid|unknown`, check results, severity,
observability, measured values/units, rejection reasons, remediability, input hash.
`TrustReport`: model/config hashes, scenario IDs/counts, valid/rejected probe counts,
unperturbed scores, ordinal histograms, flip/drift metrics, per-family results,
boundary search records and modeled MPS intervals. No single universal "trust
probability" is proposed.
`ReassessmentAdvice`: `none|clarify_acquisition|fix_acquisition_then_repeat|repeat_recommended|not_assessable`,
ranked task(s), reason codes, nominal added seconds, assumptions, stopping reason,
and `automatic_execution=false`. Failed eligibility and resource constraints are explicit.

Internal illustrative envelope:

```text
existing_score_payload: unchanged when valid synthetic-demo scoring succeeds
diagnostics_sidecar:
  schema_version, mode, quality_by_domain, trust, recap
  clinical_validity: unavailable
  interval_kind: modeled_perturbation_sensitivity
```

A failed new acquisition does not overwrite an earlier valid session. It receives
its own immutable ID and rejection record. Incomplete/invalid new sessions do not
emit a fabricated fusion vector. Existing valid domain scores may be retained as
historical observations, never silently mixed across sessions to complete MPS.

## 4. Task-aware quality checks

Hard rejection should rely on protocol/data contradictions, not motor impairment.
Warning thresholds require an explicit source or preregistered simulator assumption.

| Task | Hard checks | Warnings / unknowns, not symptom-based rejection |
|---|---|---|
| All | Parse/schema, dimensions, finite numeric values, correct units when documented, task/session association, completed duration where required | Firmware status, range, mounting and units may be unverified; don't infer exact faults from unusual motion alone |
| TS | 1,000 tri-axis samples and accepted 100-Hz time grid; declared packet/ADC failures | Repeated bit-identical samples, clipping plateaus when rails are known, unexpected baseline magnitude; high tremor or low activity alone is not failure |
| BS | One-dimensional finite times, strict increase, all in [0,10); contradictory capture metadata | Empty/single tap, long pauses, slow/irregular taps, possible bounce; these can be genuine performance or acquisition failure |
| RS | 1,000 gyro samples and time grid; declared capture faults | Constant trace/no detectable peaks, near-rail runs, suspicious quantization; low angular speed or intermittent movement may be genuine |
| GI | 18,000 acceleration samples and time grid; missing chunks/capture incomplete | Candidate-count failure, wrist interference, stops, cadence variation, apparent harmonics; do not reject a genuine slow or interrupted walk |

BS empty list with unknown capture status: report `unknown_acquisition`, feature
count may be zero but no score/MPS is inferred. Ask to confirm acquisition before
recommending a repeat. With verified completed acquisition and verified event
channel, zero taps is an observed event count, not confirmed bradykinesia. It is
outside the present V3 complete-feature support (interval std is null), so withhold
the frozen ML score rather than let imputation silently turn it into a severity claim.
Single-tap sessions receive the same feature-availability treatment.

No RS peaks or fewer than two GI candidate peaks produce undefined features in the
existing extractor. Do not invent those features or call low motion a broken sensor.
Report task/model applicability separately from confirmed acquisition invalidity.

For accelerometers, no motion near a gravity vector may be valid; all three axes
near zero throughout is suspicious given gravity-inclusive input but units/context
must be checked. Never estimate a patient's gravity offset from the TS mean and
silently subtract it: this would change the frozen feature definition.

The MPU6050 has selectable ranges/filters. Its active settings are not documented
here. Rail/sensitivity thresholds must use confirmed settings; do not assume the
smallest range. A datasheet maximum is not a measured noise distribution or a
software timestamp tolerance. [TDK product specification](https://product.tdk.com/system/files/dam/doc/product/sensor/mortion-inertial/imu/data_sheet/mpu-6000-datasheet1.pdf).

## 5. Perturbation library and assumptions

Separate (i) plausible measurement-nuisance probes, (ii) overt fault challenges,
(iii) genuine motor-state variation controls. Do not pool them into one reliability
number. Identity is always included. Applied-to-observation probes assess local
sensitivity; they do not reconstruct a noise-free latent signal.

| Operator | Domains | Scientific constraint and proposed role |
|---|---|---|
| Small additive axis bias | TS/RS/GI | Constant per task, with shared accelerometer bias scenarios for TS/GI; amplitude must come from calibration or explicitly assumed ranges |
| Additive noise | TS/RS/GI | Compare white and low-pass correlated noise; known bandwidth and units; Gaussian tails not silently clipped; no claim of measured device distribution |
| Small gain error | TS/RS/GI | Apply measurement gain to full sensor output, including gravity; do not separately amplify motor oscillation as a "sensor fault" |
| Quantization | TS/RS/GI | Conditional on confirmed full-scale range and ADC conversion; quantize physically, do not use arbitrary decimal rounding as device behavior |
| Fixed orthonormal rotation | TS/RS/GI | Rotate the entire vector including gravity; exact norm-invariance control, not independent evidence of real-world remounting robustness |
| Timing resolution/jitter | BS | Perturb observed event times under bounded timing-error assumptions; preserve count and order or mark the probe invalid; never sort/clip events to conceal collisions |
| Clock offset/drift | BS and optional IMU | Distinguish event-time metadata faults from physical sampling-time errors; fixed window/duration; out-of-window probes are reported, not dropped silently |
| Missing rows, duplicate timestamps, truncated capture | IMU | Fault-detection challenge; reject, never pad/interpolate as the production score path |
| Sample hold / clipping | IMU | Fault challenge when recorded or simulated provenance establishes it; flat motion by itself does not prove a hold fault |
| Dropped/duplicated tap observations | BS benchmark only | Allowed only in a separately simulated observation channel with an explicit complete synthetic event ledger; never invent missing taps in an observed input or claim reconstruction |
| Cadence changes, reduced wrist excursion, genuine pauses | Task-specific controls | Motor variation, not measurement noise; keep separate and assess false rejection |
| Incidental wrist movement during walking | GI | Ambiguous task behavior; analyze separately from electronic sensor disturbance; not automatically invalid |

Candidate engineering grids for later preregistration, **not hardware-calibrated
limits**: acceleration bias magnitudes 0/.01/.03 m/s²; gyro bias 0/.005/.015 rad/s;
incremental acceleration-noise std 0/.01/.03 m/s²; gyro noise std 0/.005/.015 rad/s;
gain deviation 0/±.5/±1%; tap timestamp error 0/±1/±5 ms; correlated-noise cutoff
5 Hz; coordinate rotations 5 and 15 degrees as invariance controls. These are
small exploratory assumptions, not guaranteed permissible device tolerances. Keep
them in a named synthetic sensitivity tier unless bench calibration supports them.
Do not silently broaden the range after seeing score changes or favorable metrics.

Physical timing errors for IMU cannot be implemented by merely changing timestamps
then feeding the original fixed-rate extractor. The existing adapter rejects such
timestamps. For separate synthetic observation-channel experiments, sample a known
continuous/interpolated reference trajectory at perturbed physical times and label
the reported nominal clock explicitly; disclose interpolation/bandwidth assumptions.
Exclude this operator from the first TRUST implementation until this mapping is
specified and tested. Do not resample real-device input under the frozen contract.

Rotation invariance is especially limited: `||Ra||=||a||` for an orthonormal R.
It therefore cannot validate robustness to gain errors, bias, impact, actual changed
movement after remounting, or missing taps. Similarly, BS timestamp jitter preserves
rate/count. Stability under jitter says nothing about missed events. Report tested
failure mechanisms and untested ones explicitly.

## 6. TRUST-MPS algorithm

For valid synthetic-demo input, let frozen unperturbed score be `s_d`. For each
preregistered scenario k, apply a raw-space nuisance and re-extract existing features
with unchanged algorithms. Let decoded score be `s_dk`. Invalid scenarios do not
receive scores and remain in the report denominator as rejected probes.

For N valid, equally sampled probes within a specified nuisance family:

- Flip fraction: `F_d = count(s_dk != s_d)/N`.
- Mean absolute ordinal drift: `D_d = mean(abs(s_dk-s_d))`.
- Signed drift and maximum absolute drift are also reported.
- Ordinal histogram includes all five bins, and the unperturbed score is explicit.
- Observed score envelope: `[min({s_d} union {s_dk}), max(...)]`.
- A one-step task shift changes MPS by exactly 1/16 = .0625.

Probe frequencies are frequencies under the chosen simulation measure, **not**
probabilities the original score is wrong. Do not mix deterministic grid nodes and
random draws as equally likely samples: report grid envelopes and Monte Carlo
frequencies separately. Always show N, discarded/invalid counts, seed, amplitudes,
and scenario families. With no valid probes or undefined base features, sensitivity
is unavailable, not zero. If sampling is exhausted, use `budget_exhausted` status.

Initial CPU proposal: identity + a small fixed signed grid + 32 seeded Monte Carlo
draws per domain, plus at most 32 boundary-query evaluations per domain. Set and
freeze exact counts in preregistration. No confidence claim attaches to 32 draws;
report convergence on development cases and finite-library scope. Batch feature
rows for prediction, cache unchanged domains, and use one CPU thread where applicable.

### Ordinal decision-boundary sensitivity

Primary measure: the smallest **observed** normalized perturbation amplitude in a
preregistered raw-space direction that changes the decoded score, recording signed
ordinal jump and physical units. Normalize amplitudes by their declared engineering
budgets, not by arbitrary cross-domain feature variance.

Use a finite ordered amplitude grid along each direction. For an observed flip,
optionally refine that bracket; RF/SVM predictions need not be monotonic along the
path. Report "nearest observed crossing in searched scenarios," never an exact
global robustness radius. No crossing yields "none observed up to tested budget,"
not infinite distance or a stability guarantee. Do not force the next class to be
adjacent: a jump of two or more remains a separate ordinal event.

Optional TS explanation: query the one-dimensional SVM along reachable TS-RMS
values after its existing scaler. Label this as feature-space sensitivity unless
each value has a matching admissible raw perturbation. RF split thresholds/vote
margins are optional explanations, not calibrated uncertainty. Do not calibrate
or refit SVC probabilities. [scikit-learn SVM documentation](https://scikit-learn.org/1.2/modules/svm.html).

### MPS propagation

For joint scenario k, use `MPS_k = sum_d(s_dk)/16`. Sample explicitly shared
accelerometer calibration/clock nuisances across relevant tasks and independent
per-recording noise; do not assume task independence by default. Report joint
observed min/max, mean absolute MPS drift and scenario quantiles where useful.

The rectangular envelope `[sum_d L_d/16, sum_d U_d/16]` is a conservative outer
envelope of combinations of the finite per-domain scenario results. It may include
incompatible extrema under shared nuisances. It is **not** a certified bound over
all continuous perturbations or true motor severity. If modeled covariance is
reported, `Var(MPS)=(sum Var(s_d)+2 sum_{d<e} Cov(s_d,s_e))/256`; estimate it only
under the explicitly chosen joint nuisance model, not from classifier probabilities.

All intervals must be titled **modeled perturbation-sensitivity intervals**.
No interval is computed for a complete MPS when any task is unscorable. Do not
substitute an artificial 0–4 task interval to manufacture a fusion value.

Pseudocode:

```text
inspect quality before any score
if mode is real_device:
    extract valid available features under verified adapter
    return scores=null, MPS=null; quality and optional feature sensitivity only
if required task invalid/unknown or features unsupported:
    return rejection/unavailable reasons; never invent a numeric fusion vector
base = frozen_predict(extract_original(session))
for declared nuisance family and scenario:
    copy raw input in memory; apply declared observation-channel transform
    validate transformed input; retain failures as invalid probes
    for valid probe: exact extraction -> same frozen pipeline -> ordinal decode
summarize finite-scenario stability, boundary crossings and joint MPS sensitivity
return base unchanged plus diagnostics in a separate sidecar
```

## 7. RECAP-MPS recommendation policy

This is a decision-support heuristic, not a model of expected clinical benefit.
Use four stages, preventing a time ratio from hiding serious acquisition failures:

1. **Clarify or repair first.** Unknown capture status, wrong units/configuration,
   repeated communication failure or missing metadata may require confirmation or
   repair; repeating unchanged conditions is not automatically useful.
2. **Acquisition-failure tier.** Recommend an affected task only after a remediable
   confirmed failure is resolved and a repeat is feasible. A 180-second GI failure
   cannot be ignored simply because a 10-second task is cheaper.
3. **Sensitivity tier.** Among otherwise valid tasks, identify those crossing a
   preregistered instability/boundary threshold. Rank by `D_d/(16*t_d)` as a clearly
   labeled modeled MPS-drift-per-second heuristic, followed by interval-width/time,
   then fixed task order TS/BS/RS/GI. Boundary crossings are supporting reasons;
   they do not count twice as independent evidence. Without calibrated likelihood
   weights, use per-family scenario values rather than imply a population expectation.
4. **Abstain/stop.** No eligible task, no modeled variation, nonremediable failure,
   fatigue or inability to safely repeat, no time budget, or exhausted repeat limit:
   recommend none/clarification with reasons. A high valid score alone is never a
   reason to repeat, and stable predictions do not establish correctness.

Durations are fixed: TS=10, BS=10, RS=10, GI=180 seconds. Initial full battery is
210 seconds. Repeat-all adds 210. These are recording times only: log setup,
instructions, recovery and computation separately rather than pretending the
patient's total time equals recording time.

Initial policy: recommend at most one task per round, at most one new observation
per task and at most two rounds per session, subject to user/operator constraints.
Treat these as preregistration proposals, not evaluated optimal values. Threshold
calibration uses only the new development partition described below. A finite
candidate policy grid can be proposed there; never select it from V3 final results.

Recommendation quality does not equal a guaranteed error decrease. Do not write
"expected improvement" before paired-repeat evidence. Unstable output may reflect
a genuine state near an arbitrary demo boundary. Conversely, systematic bias can
produce confidently stable but distorted scores.

The service returns advice only. No device command, fabricated repeat, automatic
acquisition, averaging of scores, or overwrite is allowed. An actual later repeat
has a new observation ID and immutable lineage. Existing valid scores stay intact.
Choosing a repeat for a separate research comparison is an explicit evaluator action,
not a silent production replacement policy. Real-device RECAP uses acquisition
quality only, because demo-score/MPS sensitivity is unavailable in that mode.

## 8. Separate disturbance and paired-repeat benchmark

### Preregistration and partitioning

Use only the existing **168 training-side session IDs and their archived parameters**.
The 36 original validation and 36 inspected test sessions are forbidden. Establish
a denylist and check parent IDs before any new benchmark processing. Never invoke
the original final-test entry point or reuse its report to choose cases.

Proposed deterministic split: 84 parent states for innovation development and 84
for locked innovation audit, balancing only synthetic profile (42 each per split)
via a published seed/hash ordering, before looking at innovation outcomes. Keep
every nuisance realization and repeat of a parent in its one partition. If the
training-side profile inventory differs, stop and preregister revised counts.

The frozen classifiers have already seen all these parent states during training.
The innovation audit evaluates disturbance/reassessment policies on held-out
parents relative to innovation development, **not classifier generalization to
new participants or independent generator families**. Original generator families
remain shared and must be reported. New nuisance seeds do not create new data sources.

Two development resources are distinct:

- Analytic identity/invariance and malformed-input fixtures for software checks.
- A finite development benchmark for nuisance/policy threshold choices. Before
  generating it, freeze disturbance definitions, target false-rejection constraint
  (proposed <=5% on technical no-fault development captures), selection objective
  and tie rules. This constraint is an engineering target, not a clinical standard;
  report uncertainty and failure to meet it rather than relax it post hoc.

Freeze the selected policy, code/model hashes, seeds, counts, outcomes and exclusions
before generating/accessing the 84-parent innovation-audit outcomes. No iterative
reuse of that audit set. A separately named exploratory run is required for any
later method redesign; it cannot replace the preregistered result.

### Observation channels and paired repeats

Leave original V3 raw files and generator untouched. A separate benchmark builder
will create descendants under new IDs and directories from training-side anchors.
Use the unchanged archived generator if reconstructing an anchor from parameters;
verify its hash and numerical agreement first. Alternatively use the stored parent
waveform as a fixed reference trajectory, explicitly retaining its existing noise.
Do not call that reference a known noise-free human motor state.

For each parent, generate an initial synthetic observation plus one pre-generated
repeat per task. All policies receive the same potential repeat measurements,
but can access them only after choosing an action from initial diagnostics. This
prevents the policy from selecting whichever future repeat happens to look best.
The benchmark evaluator, not RECAP, manages these synthetic counterfactual records.

Required paired regimes:

1. Transient observation disturbance: repeat has an independent observation-noise
   draw under the same state. Do not force it to be cleaner or closer to the reference.
2. Persistent fault/bias: nuisance persists across repeats; repeating without repair
   should not be credited with improvement. Include zero/negative gains.
3. Confirmed repaired acquisition fault: after an explicit repair assumption, repeat
   is sampled from the ordinary no-fault channel, not guaranteed error-free.
4. Motor-state-change negative control: genuine rhythm/excursion/pauses change;
   analyze separately because a different score is not measurement error.
5. No-fault control across mild/strong and irregular patterns, to expose false rejection.

Known observation faults include missing blocks, sample holds, rail clipping at
declared settings and synthetic tap-event loss. Tap faults require an explicit
latent synthetic event ledger; no imputation of missing real observed taps occurs.
Record severity grids and include nuisance mechanisms not used inside TRUST's
own probe library (for example persistent holds when TRUST probes small bias/noise).
Otherwise the evaluation merely rewards recognition of its own perturbation model.

Predefine equal exposure counts per family for experimental comparison, not a
claimed real-world fault prevalence. Report family-wise outcomes; any pooled rate
must state its chosen mixture weights. Session manifests include parent/split,
generator/operator hashes, sensor assumptions, initial/repeat seeds, fault truth,
repair/persistence state, reference outputs and observation lineage.

### Ablation arms

| Arm | Behavior |
|---|---|
| A | Frozen ML baseline with unavoidable parser/protocol checks; no fabricated scores on malformed input |
| B | A + task-aware quality gate; no score sensitivity |
| C | A + modeled stability diagnostics; only structurally valid inputs are probed |
| D | Full quality + stability + ordinal boundary/MPS diagnostics + RECAP advice |

Additional component ablations: D without boundary diagnostics, without MPS
propagation, and without time normalization. Numeric baseline predictions on the
same accepted unperturbed input must remain identical across all arms. Quality
gates affect coverage, not the underlying score. Report rejected cases separately
to avoid inflated accuracy from selective exclusion.

Retest comparators: no repeat; repeat-all; uniformly random eligible task with a
preregistered seed; quality-only priority; instability-only priority; full RECAP.
Use identical feasibility rules, initial inputs and repeat bank. Evaluate one-action
comparisons and recording-time budgets (proposed 10/20/180/210 seconds) separately;
random tasks have unequal costs, so equal action count is not equal time. An oracle
using all potential repeats is an evaluator-only upper comparator, never a deployed
policy or a source of runtime recommendations.

### Metrics and estimands

| Question | Required outcomes |
|---|---|
| Fault detection | Sensitivity/specificity per known injected fault/severity, precision with stated mixture, unknown/abstain rate; false rejection among technically valid no-fault observations |
| Stability | Flip fraction, absolute/signed ordinal drift, >=2-score jump fraction, invalid-probe fraction; per nuisance family and base score |
| Boundary sensitivity | Nearest observed crossing, tested amplitude units/budget, censored no-crossing rate, query count |
| MPS sensitivity | Absolute/signed shift from unperturbed MPS, joint modeled envelope width and quantiles; interval coverage of separately drawn modeled repeats, explicitly not clinical coverage |
| Selective output | Accepted/complete-session coverage, conditional agreement/drift, rejection reasons and abstention; no treating rejection as an accurate prediction |
| Retest targeting | Recommendation rate, known remediable-fault targeting precision/recall, unnecessary-repeat rate on no-fault cases, persistent-fault repeat rate, positive/zero/negative paired effect fractions |
| Reassessment effect | Paired change in score/MPS distortion, coverage recovery, improvement per extra second and regret versus same-budget oracle; measured only after repeats, not promised by the heuristic |
| Time/cost | Initial 210 s, incremental recording seconds, recorded setup/rest cost, total requested time, CPU wall time and model/feature-extraction call counts |

Use two distinct references: (i) frozen model output on the untouched training-side
anchor, measuring prediction consistency; (ii) original engine demo score on that
anchor, measuring rule-label agreement. Neither is clinician ground truth. For a
motor-state-change control, use its own intended state reference and do not score
a valid changed state as a nuisance-induced error. Preserve all V3 labels unchanged.

For valid initial/repeat observations, signed paired benefit is
`abs(MPS_initial - MPS_reference) - abs(MPS_after_selected_repeat - MPS_reference)`;
report negative values rather than censor them. For initially invalid inputs,
numeric error is undefined: report valid-coverage recovery and subsequent error
separately, not a fabricated large initial error. The "after" assembly is a labeled
benchmark counterfactual only; original production outputs are not overwritten.

Use parent-state cluster bootstrap intervals and paired comparisons across policies;
perturbations of one parent are not independent sample units. Report counts and
uncertainty by domain, pattern severity and nuisance mechanism. Broad simultaneous
claims need multiplicity control; initial analyses should name a small set of
primary endpoints and label others exploratory. Suggested primary endpoints are
technical false rejection, remediable-fault targeting, and paired MPS distortion
change at the fixed budgets. No target 90–95% accuracy is imposed.

## 9. Dependencies, safety and implementation phases

Use the existing CPU-compatible versions first: NumPy 2.2.6, SciPy 1.15.3,
scikit-learn 1.6.1 and joblib 1.4.2. Selected RF/SVM inference does not require
XGBoost or CatBoost, but the original full training environment may be retained
when verifying saved artifacts. Match actual remote dependency metadata before
deserialization. Standard-library dataclasses, json, csv, hashlib and pathlib are
sufficient for contracts/provenance. Optional Matplotlib for offline plots only.
No PyTorch, GPU, Gradio server or heavy local installation is needed.

Safety/edge cases to test: wrong domain/session ordering; wrong units; absent range
metadata; tap collisions/out-of-window times; empty/single tap; zero RS peaks;
missing IMU samples; NaN/Inf; clipped/held samples versus genuine stillness; schema
or model-hash mismatch; class decoder missing a score; unsupported features despite
an imputer; all probes invalid; zero empirical variation; conflicting joint extrema;
query/time-budget exhaustion; unsafe or declined repeat; persistent fault; multiple
repeat rounds; audit-set parent leakage; and real-mode accidental model invocation.

Treat joblib as trusted serialization: verify expected artifact hashes first.
Loaded models, input arrays, baseline scores and manifests are read-only by contract.
Every derived observation/report receives a new versioned path and ID. Random
seeds make scenarios reproducible, not independently sourced. Never call training,
selection, freeze or original-test routines from the innovations package.

Implementation plan, each requiring the appropriate approval:

1. **Phase 3B — interfaces and tests only.** Create the sibling package, immutable
   contracts, frozen-artifact inspection adapter, quality checks and real-mode guard.
   Small invented local fixtures and static checks only. No full inference.
2. **Phase 3C — diagnostic implementation.** Add explicit nuisance operators,
   finite boundary search, TRUST summaries and RECAP advisory policy. Test identity,
   unit consistency, no mutation, ordinal decoding and MPS arithmetic on tiny fixtures.
   Keep all Colab execution switches FALSE.
3. **Phase 3D — preregister and freeze.** Finalize nuisance assumptions, state splits,
   grid/seed budgets, policy grid, technical false-rejection target, endpoints,
   repeat-bank rules and exclusion list before generating benchmark descendants.
4. **Phase 3E — approved Colab development.** Verify remote selected artifacts;
   run feature/prediction parity on training-side inputs only, build the development
   observation/repeat bank, select policy thresholds under the registered rule.
   Freeze innovation settings without touching V3 or its models.
5. **Phase 3F — separate approved innovation audit.** Generate/process the locked
   innovation-audit descendants once; compare arms and retest policies; report all
   failures and uncertainty. Leave original validation/test data unopened.
6. **Phase 3G — optional integration.** Add an optional internal diagnostics/dashboard
   section after golden fusion-contract checks. No Gradio launch in earlier phases.

**Exact next step after approval:** create `motor_scoring_research/innovations/`
with contract definitions, the hash-verifying selected-artifact adapter interface,
task-quality reason codes, and tiny tests proving real-mode score suppression,
unmodified numeric payloads, empty-tap handling and no mutation. Do not generate
benchmark data or execute model inference in that step. Remote selected-artifact
inspection and bench-informed nuisance ranges remain prerequisites for experiments.

TRUST-MPS and RECAP-MPS are proposed research contributions, not claims of unique
worldwide novelty, clinical correctness, calibrated severity or disease probability.
Approval is required before innovation implementation begins.
