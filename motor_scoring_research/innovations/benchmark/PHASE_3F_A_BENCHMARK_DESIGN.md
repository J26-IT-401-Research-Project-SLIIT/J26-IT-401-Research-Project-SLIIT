# Phase 3F-A preregistration: TRUST-MPS and RECAP-MPS

Date: 2026-10-22. Status: **design and metadata-only parent split frozen**.
No benchmark recording, synthetic repeat, model inference or locked-audit
outcome has been generated or evaluated. The machine-readable companion is
[preregistration_v1.json](preregistration_v1.json).

## Scientific question and claims

This benchmark asks whether task-aware quality checks reject known acquisition
faults without rejecting technically valid simulated motor patterns, whether
TRUST's finite perturbation response helps identify predictions sensitive to
independent same-state acquisition variation, and whether RECAP chooses useful
repeats under a fixed recording-time budget. These are **within-simulator,
frozen-model** questions. The observed 0–4 scores imitate synthetic demo
rules, and MPS remains `(TS+BS+RS+GI)/16`. Neither the scores nor any modeled
TRUST interval is a clinically calibrated severity, confidence interval or
Parkinson's probability. Acquisition quality, score stability, simulator-rule
agreement and clinical validity are separate outcomes.

All four frozen classifiers have already seen the 168 V3 training parents
during their own model fitting. The innovation audit therefore does **not**
measure classifier generalization to unseen participants. All 168 records
share synthetic source reference `SIM_ASSUMPTIONS_V3`; family templates also
cross splits. Different observation seeds do not establish source independence.

## Parent source and locked split

The source is only the metadata in
[frozen_session_plan.csv](../../v3/training/configs/frozen_session_plan.csv),
whose SHA256 is `134ddcd16393f5122818f3eabb14d6661d443d1efc1d33f13a267fbf9a1c76bf`.
It lists exactly 168 training, 36 original validation and 36 original test
sessions. Training contains 84 healthy-like and 84 symptom-like synthetic
profiles. The original 72 validation/test IDs form a fixed denylist; neither
those cases nor their scores, signals or prior performance enter this work.

[freeze_parent_split.py](freeze_parent_split.py) sorts each profile's 84
training IDs by `(SHA256("PP1_INNOVATION_PARENT_SPLIT_V1|20261022|" + ID), ID)`.
The first 42 per profile are innovation-development; the remaining 42 are
locked innovation-audit. The resulting [168-row manifest](frozen_parent_split_v1.csv)
has SHA256 `b988fd10fd6e13cd4ff39761e5756fed3f6c3c70b77931d4dcde8dd2facd7581`,
recorded in [its lock](frozen_parent_split_v1.lock.json). The split is exactly
84/84 and 42/42 by profile. Development has 44 amplitude-modulated, 20
continuous and 20 intermittent families; audit has 33, 20 and 31. This
imbalance is reported, not corrected after assignment. Every initial
observation, perturbation, repeat and outcome inherits its parent split.
No descendant may cross it. All later source manifests must verify the
parent manifest hash and original denylist hash before touching a raw case.

## Fixed scenario bank and assumptions

The benchmark uses **seven fixed slots per parent** (588 initial observations
per 84-parent side). Within each side, sort parents by their frozen split key
and ID, then assign a zero-based rank. Slot rotations depend only on rank,
never signals, labels or outcomes. A seed for each parent/slot/role/domain/
attempt is the unsigned big-endian first eight bytes of SHA256 of
`master_seed|parent_id|slot|role|domain|attempt`, with master seed `30702026`.
The random comparator and bootstrap have separate seeds `7032026` and
`9022026`. Role includes `initial`, `repeat`, `trust_probe` or `motor_control`.
No generator state may be shared unintentionally across roles.

| Slot | Initial observation | Pre-generated repeat mechanism | Primary-analysis role |
|---|---|---|---|
| S0 | No injected fault; retain frozen parent observation | Independent small observation draw | Technical no-fault/false-rejection control |
| S1 | One rotating IMU nuisance (below) | Independent nuisance draw under same nominal state | Valid score sensitivity and paired effect |
| S2 | Bounded BS timestamp jitter; event count/order kept or probe rejected | Independent jitter draw | BS timing sensitivity |
| S3 | Rotating acquisition fault, with correction explicitly confirmed before decision | New capture from ordinary channel after repair | Coverage recovery; improvement is **not** guaranteed |
| S4 | Rotating acquisition fault, unresolved | Fault persists across repeat | Cost of futile repeat, zero/negative effects |
| S5 | Initial technically valid; repeat receives a new acquisition fault | Worsened acquisition mechanism | Negative-repeat control |
| S6 | Controlled simulated motor-state change between captures | New intended simulated motor pattern | Separate genuine-variation control; excluded from same-state error-benefit estimand |

The S0 and repaired-S3 ordinary repeat channel adds an independent per-axis
Gaussian draw (acceleration std 0.01 m/s²; gyro std 0.005 rad/s) and bounded
BS jitter of ±0.001 s to the frozen nominal anchor. A timing collision is
an invalid repeat, not repaired or redrawn. S1/S2 repeat the same nuisance
family and severity with an independent seed/sign/draw, so improvement is
not built in. S4 retains the same unresolved fault family and placement on
repeat; only ordinary observation noise may differ. In S5, target `rank % 4`
maps to a 0.01-s TS timestamp gap, one removed BS ledger tap, declared RS
acquisition failure, or a 1,000-row GI tail truncation. If the BS ledger is
ineligible, record the failure without replacing that parent/scenario.

For S1, `rank % 4` selects TS accelerometer offset, RS gyroscope offset, GI
accelerometer noise, or a shared TS/GI accelerometer gain change. Each has
21 audit parents. The two engineering severities are: acceleration offset
0.01/0.03 m/s², gyro offset 0.005/0.015 rad/s, per-axis Gaussian acceleration
noise standard deviation 0.01/0.03 m/s², and full-vector gain magnitude
0.005/0.01. The signed offset/gain direction is fixed by derived seed parity.
Noise is independent per sample/axis and per acquisition; no claim of measured
MPU6050 noise spectrum follows. Gain acts on the entire observed vector,
including gravity. The shared-gain scenario applies the same signed gain to
TS and GI, preserving that modeled acquisition correlation. S2 uses uniform
per-tap jitter bounded by 0.001 or 0.005 s, preserving all actually observed
tap events and order; collision/window failures are recorded as invalid
probes, never silently sorted or clipped. Severities are set by the rank rule
in [the machine specification](preregistration_v1.json).

S3/S4 `rank % 6` rotates: one invalid IMU timestamp displaced by 0.005/0.01 s
while row count remains fixed; truncated IMU capture (100 trailing TS/RS rows
or 1,000 GI rows); explicitly wrong declared unit; declared acquisition
failure; deletion of one BS tap; insertion of one duplicate BS event. This
gives 14 parents per structural family per slot and 28 per family across
S3/S4 on each split. The target for IMU structural faults rotates TS, RS and
GI by `floor(rank/6) % 3`, so fault family and target domain are not locked
together. For tap deletion/insertion, the archived generated tap list
is an **assumed synthetic event ledger**, not an inferred complete real-device
event set. Require at least three original taps and a >2 ms admissible gap for
insertion. If these prerequisites fail, record a preregistered ineligible
scenario with its denominator; do not substitute another parent or invent a
tap. A missing or duplicated event may still pass the raw schema. Neither
quality checks nor jitter probes are entitled to detect an unobservable event
fault. Do not use latent fault truth as a model predictor. Even in S3, a
repaired synthetic tap-event fault is **not** reported to policies as
confirmed corrected unless an independently simulated acquisition log
contains that fact; the default bank provides no such log. Its correction
exists only in the evaluator's potential repeat record.

S5's new fault targets `rank % 4` in TS/BS/RS/GI. S6 changes one archived
generator parameter after copying the parent's parameter dictionary: by
`rank % 4`, multiply `ts_amp`, `bs_hz`, `rs_amp` or `gi_hz` by 0.90/1.10
(GI 0.92/1.08), with factor chosen by `floor(rank/4) % 2`. Clamp to the
original V3 configured parameter range and report no-change boundary cases;
never substitute a new parent. Re-synthesize with a separately derived signal
seed, preserve all other nominal parameters, and label this as a **changed
simulator state**, not sensor-quality failure. Its reference is the original
engine's demo score on that intended changed state, never the original
parent's score. S6 checks whether quality rules wrongly reject plausible
performance differences. It is not evidence of a known human state change.

The original V3 raw files and generator are immutable. A future builder may
use each archived **training** waveform as a fixed reference trajectory,
retaining its pre-existing simulated noise, or reconstruct from archived
parameters only if original generator/hash and numerical parity are verified.
The waveform is not a noise-free human motor state. Repeat disturbances are
independent conditional on this nominal anchor except for explicitly
persistent/correlated scenarios. Even here, unchanged *real* motor state cannot
be guaranteed; no real paired patient measurements exist. "Improved",
"unchanged" and "worsened" refer to **measured** simulator-based outcomes,
not outcomes forced by label manipulation. Repaired captures may worsen and
new-fault captures may not alter a thresholded score; report those facts.

The active [TRUST probe config](../configs/trust_engineering_v1.json) and
[RECAP policy config](../configs/recap_policy_v1.json) are frozen for this
initial benchmark. Probe magnitudes overlap some injected nuisance ranges,
so family-wise results and an independent-repeat endpoint are mandatory.
Structural faults, tap-count loss and genuine motor variation are challenge
mechanisms outside TRUST's small measurement probe set. Quantization remains
disabled without a verified conversion factor. IMU physical clock warping is
excluded without a defensible continuous trajectory. None of these ranges
are calibrated hardware tolerances.

## Policy comparison and information rules

Before any policy acts, generate **one potential repeat per task** using the
same parent/scenario seeds. Store it in an evaluator-only, append-only bank.
Policy inputs contain the initial observation, available acquisition context,
quality report and (where allowed) TRUST sidecar only. They never expose
future repeat values, scenario truth, simulator parameters, rule labels or
reference MPS. The evaluator reveals only repeats selected by a policy;
repeat-all reveals all four. A selected invalid repeat stays invalid; it is
not redrawn. The initial score payload is immutable. Any "after" score is a
separate counterfactual research result, never a device action or overwrite.

Every policy uses identical initial parser/protocol safety checks and the
same potential repeat bank. Fixed durations are TS/BS/RS 10 s and GI 180 s;
initial battery 210 s. Additional budgets are 10, 20, 180 and 210 recording
seconds, with **180 s primary**. Setup/rest/computation are logged separately.

| Policy | Registered decision rule |
|---|---|
| A no retest | Keep initial valid outputs only; zero added recording time. |
| B repeat all | Repeat all four, additional 210 s; infeasible at 10/20/180 s and reported as such. Simulated comparator only. |
| C random eligible | Use seed 7032026; first consider confirmed-corrected failures, otherwise valid domains. Uniform among feasible domains under the same budget; stop for unresolved/unknown acquisition. One task, one repeat. |
| D quality-only | Repeat a confirmed-corrected acquisition failure, ordered TS/BS/RS/GI, if feasible. Otherwise no repeat; ambiguous/persistent faults prompt clarification or repair. |
| E TRUST instability-only | On complete valid synthetic-demo inputs, rank domains with observed valid-probe score changes by `mean_absolute_ordinal_drift/(16*duration)` and fixed TS/BS/RS/GI tie order. No repeat for an invalid/unknown acquisition or missing TRUST evidence. One task. |
| F full TRUST + RECAP | The unchanged [RECAP implementation](../recap.py) with its frozen config: acquisition safety/remediability first, then the same modeled sensitivity-per-second heuristic, one task/round and at most one repeat/domain. |

All except B use one action in the primary comparison. B is shown only at
210 s and is not credited as an equal-cost 180 s comparator. If GI is
infeasible at 10/20 s, record it as budget-blocked; do not treat a cheap
unrelated task as curing a GI acquisition failure. An optional oracle that
looks at all repeats is evaluator-only, exploratory and never a deployable
policy. Real-device mode has no score-dependent policy or MPS.

## Estimands, metrics and denominators

Report each scenario family and severity separately, with actual eligible,
invalid and missing counts. Any pooled rate uses **equal slot weight within
each parent**, an experimental exposure mixture, not an estimate of real
fault prevalence. Average within parent before across-parent comparisons.

1. **Acquisition fault detection.** For the four directly observable
   structural faults (timestamp, incomplete, wrong declared unit, declared
   failure), report true rejection/clarification sensitivity. For missing or
   duplicated BS events, report separately because the existing sensor input
   may not reveal them. For S0 and technically valid S1/S2/S6 observations,
   report false hard-rejection rate. Unknown/abstain is a separate denominator,
   not silently counted correct. Report accepted complete-session coverage.
2. **Score sensitivity.** Among structurally valid observations with scores,
   record per-domain exact flip against the untouched parent model output,
   signed/absolute ordinal drift, >=2-step jumps, TRUST probe invalid fraction,
   tested score-change fraction and no-crossing count. An independent same-state
   repeat score flip is the prespecified prospective instability event. Compare
   its frequency among TRUST-flagged (`>=1` valid score-changing probe) and
   unflagged cases. A stable probe result is not evidence of clinical truth.
3. **MPS.** For complete valid observations, record predicted MPS versus the
   untouched parent's original-engine **demo-rule** MPS and absolute deviation.
   Also report deviation from the untouched frozen-model MPS as a separate
   consistency outcome. A rule-defined target is not clinician ground truth.
   TRUST's paired joint-probe min/max/width is a modeled sensitivity interval;
   evaluate whether an independently drawn same-state repeat falls inside it
   only as finite-nuisance coverage, not a clinical interval guarantee.
4. **Retest targeting and effect.** Report rate of recommending the known
   corrected fault, unnecessary repeat on S0, repeat on unresolved S4,
   budget block, and additional recording seconds. For parents with complete
   meaningful MPS before and after a selected repeat, define paired benefit
   `|MPS_initial-ref| - |MPS_after-ref|`; report positive/zero/negative
   fractions. If initial MPS is unavailable, report coverage recovery and
   subsequent deviation separately: no artificial initial error is assigned.
   A failed/invalid repeat can reduce coverage. For S6 state-change controls,
   compare to the changed-state reference separately; exclude from same-state
   paired-benefit primary endpoint.
5. **Time and compute.** Report total initial 210 s, additional recording
   seconds, any separately observed setup/rest time, feature-extraction/model
   call counts and Colab CPU wall time. These are not patient-time estimates
   unless setup/rest are actually measured.

Primary comparison at 180 s: for each eligible same-state parent, average
`error_after_C - error_after_F` across its applicable slots. Positive means
F had less simulator-reference MPS deviation than the seeded random policy.
No post hoc exclusion for a policy's recommendation or poor repeat result.
Report complete-case paired error and a separate coverage-intention-to-assess
table for all parents. D and E provide mechanistic comparators; B is compared
only at 210 s. Do not call any agreement here clinical accuracy.

## Uncertainty and success decision

The **parent** is the independent resampling unit. Use 2,000 paired,
profile-stratified cluster bootstrap resamples with seed `9022026`: sample
42 healthy-like and 42 symptom-like audit parent IDs with replacement, taking
*all* their scenario descendants together. Derive percentile 95% intervals
for descriptive rates and paired policy contrasts. Do not treat 588 audit
observations or thousands of TRUST probes as independent human samples.
Report the number of parent clusters supporting each endpoint; small
structural-family strata (~28 per audit side) warrant wide-interval cautions.

Four preregistered success checks apply to the single-use audit:

- Technical false hard-rejection point rate <=5% on S0 and valid S1/S2/S6
  observations. This is an engineering target, not a clinical standard.
- Directly observable structural-fault detection point rate >=95% across
  timestamp-gap, incomplete, invalid-unit and declared-failure challenges.
  BS event-count faults are **not** included in this target.
- TRUST-flagged valid same-state observations have an independent-repeat
  score-flip rate at least 0.10 higher than unflagged observations. If either
  group is empty, this criterion is indeterminate, not a pass.
- F's parent-paired mean `error_after_C - error_after_F` at 180 s is positive.
  If no parents have comparable complete MPS, this is indeterminate.

For the two directional comparisons, require the lower 2.5th percentile of
each parent-cluster bootstrap distribution to exceed zero. These two 97.5%
one-sided intervals apply a Bonferroni family-wise alpha of at most 0.05.
Passing an effect-size target without its interval check is suggestive, not
confirmed. The first two engineering targets are reported with cluster
intervals but are not p-value tests. Declare **overall success only if all
four checks pass**. Report failures and indeterminate results without
changing thresholds or excluding hard cases. Other budgets, F vs D/E/B,
family-specific effects and oracle contrasts are secondary/exploratory.

## Audit lock, exclusions and stopping rules

Before development generation, verify the original source and split SHA256,
the denylist, engine/training hashes, selected checkpoint hashes, TRUST/RECAP
config hashes and CPU environment. Record all in a new versioned benchmark
run directory. No original file is written. Development descendants may be
generated only after separate approval. Software faults found in development
may lead to a new versioned preregistration **before** any audit access; record
the change and preserve this v1. No V3 validation/test data may guide it.

Before **any** audit descendant is generated, freeze the final scenario,
operator, policy, metric and code hashes plus development decision log.
Create an append-only `AUDIT_STARTED.json` claim in Google Drive with exclusive
creation; a failed claim still counts as use. Generate/read the 84 audit
parents once. Do not rerun, repair selectively, tune policy settings or
reopen it as a new "final" audit. Report any implementation failure and
partial coverage. Unexpected missing training parent, hash mismatch, split
leakage, no device/unit provenance, unavailable dependency, seed collision,
out-of-budget runtime or score-contract violation stops before audit access.

Exclusions are fixed: original validation/test parents and descendants;
non-training IDs; absent/corrupt archived parent raw file; S3/S4 tap-fault
subscenario without ledger prerequisites; S6 from same-state benefit; cases
without defined MPS from numeric MPS-error denominators. Each exclusion has
an explicit count and reason; no replacement parent or redraw. Unknown tap
acquisition remains unscored and eligible for clarification, not imputed.
Slow tapping, intermittent wrist motion and altered gait remain potentially
valid motor patterns, not automatic sensor-quality failures.

## Practical Colab CPU execution

Use the verified frozen four-model adapter and complete-session extractor in
a fresh matching Colab CPU environment. Work in a new versioned innovation
benchmark directory; mount Drive only for read-only original source and
append-only new outputs. Build development cases in small parent chunks,
stream one parent/slot and its pre-generated repeat bank at a time, and
write manifests/results atomically with checksums. Never keep the entire
18,000-row GI repeat bank in memory or copy raw V3 data into the repository.
Cache only unchanged-domain features when their exact input bytes are
identical; preserve the full-session extractor contract and validate parity.
Limit TRUST to its fixed 32-attempt maximum; log actual CPU calls and wall
time. A runtime interruption resumes only from completed checksummed
development chunks. The audit claim makes audit restart a reportable
exception, never a silent second evaluation.

**Exact next implementation step after approval:** implement a streaming
development-only benchmark builder that reads the 84 IDs marked
`development` in the frozen parent manifest, enforces the 72-ID original
validation/test denylist, and emits only the preregistered S0–S6 scenario
and paired-repeat manifests. Test its lineage and seed determinism with
invented fixtures before any Colab generation. No audit generation or
inference is authorized by this design document.
