# Phase 3F-B implementation (no benchmark executed)

The immutable scientific design is [PHASE_3F_A_BENCHMARK_DESIGN.md](PHASE_3F_A_BENCHMARK_DESIGN.md),
with machine specification [preregistration_v1.json](preregistration_v1.json).
Phase 3F-B adds implementation code only. It does not revise the design,
disturbance magnitudes, thresholds, parent split, scoring models, or historical
V3 files. The runner is Colab CPU only and requires a manual approval flag.

## Entrypoint and gates

Run the notebook [PP1_3F_B_TRUST_RECAP_Benchmark_CPU.ipynb](PP1_3F_B_TRUST_RECAP_Benchmark_CPU.ipynb)
from a fresh CPU runtime. The first cell checks the runtime without loading
data or models. The notebook mounts Drive, restores the code ZIP to a new
`/content/PP1_3F_B_workspace`, compares the recorded training Python/package
versions, installs matching packages in Colab only, then calls hash-only
`verify_design` and `verify_frozen_models`. The selected-model manifest hash
must be supplied from the earlier trusted preflight. No default hash is
assumed. Model deserialization is permitted only inside the separately gated
development run and only after frozen file hashes are reverified.

`runner.run(..., split='development', approved_development=True)` reads only
the 84 development training parents. It writes to a fresh child directory of
`/content/drive/MyDrive/PP1_V3_innovation_benchmark_runs`. Each parent file
contains scenario/repeat lineage hashes, quality/score sidecars, policy
decisions, coverage and paired outcomes. The summary reports parent-cluster
intervals, policy costs and exclusions. Potential repeat recordings never
enter `policies.decide`; the evaluator reveals only selected repeats. All
outcomes remain research counterfactuals, with no physical device action or
score overwrite. Real-device output remains governed by the existing
features-only contract and is never passed to the demo runner.

The audit cell defaults to `False`. A future separately approved audit run
requires a reviewed `DEVELOPMENT_LOCK.json` at the single fixed Drive registry
and its exact hash. `AUDIT_STARTED.json` is created exclusively **before**
reading audit raw cases. A failed audit attempt consumes the single-use
claim. No code should remove or rename the claim marker. The generated
development `DEVELOPMENT_FREEZE_CANDIDATE.json` is a review candidate, not an
automatic audit authorization.

## Interpretation and limits

The original frozen scoring classifiers were fitted using the original 168
V3 training parents. This benchmark measures quality detection, perturbation
response and retest decisions within that simulator. It does not establish
classifier generalization to unseen people, clinical correctness, source
independence or calibrated sensor tolerances. The seven scenarios are fixed
experimental exposures, not estimated real-world fault prevalence. The S6
changed-state control is excluded from same-state retest benefit. For an
initially unscorable session, coverage recovery is counted separately;
no artificial baseline MPS error is assigned.

The code-only ZIP contains the innovation modules, locked design/spec/split,
original engine and generator source, V3 configuration and frozen training
plan/source. It excludes raw V3 recordings, joblib checkpoints, validation/
final-test records and prior results. Upload it to the notebook's designated
Drive path before running setup. Full results may require material Colab CPU
time; no local full-data experiment is authorized by this implementation.
