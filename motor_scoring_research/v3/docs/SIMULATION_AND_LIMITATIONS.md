# V3 assumptions and scientific limits

All numeric values in [configuration](../configs/simulation_v3.json) are explicit
engineering assumptions. No public participant records, empirical anchors, or
clinical severity distributions are used in V3. Public-source task context is
documented in the [source manifest](../manifests/source_manifest.json).
[PADS](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/) supports
the use of wrist acceleration and rotation tasks as research modalities; it does
not calibrate this PP1 simulator or its four ordinal scores. Other listed sources
are inherited V2 references; their numerical parameters are not used here.

## Sampling before scoring

Main dataset: 240 sessions, 120 per healthy-like/symptom-like profile. This design
balance is not disease prevalence. Profile latent draws use Beta(1.2,4.0) and
Beta(1.8,1.5). Each domain uses a mixture: 25% perturbed shared session latent,
55% independent profile latent, 20% uniform latent. This intentionally broadens
the simulator's input range and allows overlap without imposing equal score counts.
No thresholds appear in generation equations. Scores are assigned once by the
original engine after signal creation. No class quotas, score-conditioned rejection,
post-generation retries, or performance-driven edits are allowed for this version.
Coverage is attempted, not promised. Missing/rare classes are retained and reported.

The three main generator families (continuous, amplitude-modulated, intermittent)
share mechanics and may occur in every split. These family labels are descriptive,
not evidence of independence. A separate optional 40-session stress set uses
chirped rhythm, 1.8x sensor noise, and 2x gait interference. It is a predetermined
shift on the same underlying simulator, not an independent clinical source.

## Parameter ranges

| Component | Configured range and meaning |
|---|---|
| TS | Log-interpolated peak acceleration .035–2.6 m/s²; 4–9 Hz; gravity-axis cosine .4–1; weak harmonic and .35 Hz incidental motion |
| BS | Nominal .7–4.5 taps/s; lognormal interval CV .015–.30; 0–10% hesitation probability with .15–.65 s extra delay; up to 25% interval lengthening over the trial; minimum interval .09 s |
| RS | Nominal peak .30–3.8 rad/s; .45–1.4 Hz active oscillation; small secondary axes, amplitude modulation and optional pause |
| GI | Nominal .78–2.2 candidate cycles/s; dynamic peak .7–2.6 m/s²; 1–8% smooth frequency variation; harmonic, amplitude variation, .02–.30 m/s² independent wrist interference |
| Acceleration noise | .008–.045 m/s² Gaussian std per axis, .012 m/s² constant-bias std |
| Gyro noise | .004–.020 rad/s Gaussian std per axis, .004 rad/s constant-bias std |
| Orientation | Session/task rotation; smooth 0–.15 rad coordinate drift rotating gravity and movement together |
| Intermittency | Smooth .3–1.2 s attenuation, center 25–75% through recording; minimum envelope fraction .15, not a guaranteed complete stop |

Amplitudes are waveform parameters, not exact output RMS/peak measurements.
Nominal tap rate is reduced by hesitations/fatigue; it is not the resulting count.
Noise is Gaussian and hence not mathematically bounded; no ADC quantization or
hard clipping is modeled because hardware configuration is unconfirmed. Gravity
is 9.81 m/s² in accelerometer signals and absent from gyro signals. The models for
separate tasks are not a coupled rigid-body biomechanics simulation. Slow coordinate
drift is not a reconstruction of clinically observed posture.

Source and sensor realism are limited: no measured MPU6050 noise calibration,
mounting distribution, firmware serialization, validated movement statistics,
clinician ratings, or physiological reference cutoffs. Mechanically plausible
illustrations must not be described as realistic population distributions.

## Leakage controls and interpretation

Split assignments precede signal and target creation: 168 train, 36 validation,
36 test, balanced by profile only. Session ID binds all four tasks. Separate
64-bit deterministic seeds are recorded, but seeds do not imply source independence.
No empirical numeric sources are used, so source-disjoint generalization is neither
claimed nor tested. An empirical mode must first acquire eligible identified
sources, group linked records, partition them, and record the resulting memberships.

Only the nine domain-specific engine features enter the future ML matrices.
IDs are join keys only. Profiles, split names, seeds, parameters, family/source IDs,
scores and MPS are prohibited predictors. Targets and features are separate files;
the full session manifest is provenance, never an ML feature table.

Learning these targets is learning deterministic rules. High agreement may be
expected when their deciding features are provided. This is inherent in the
research objective, not evidence of clinical validity; adding noise cannot convert
rule imitation into clinical scoring. The deterministic baseline defines labels
and should reproduce them, subject to integrity verification.

All four protocols remain unchanged. BS has no amplitude measurement. RS measures
active motion, not passive rigidity. GI is wrist step-candidate counting. MPS is
the four-score sum divided by 16 and is never disease probability. Real-device mode
must return features only with scores/MPS=null. Original PADS binary classifiers
and their metrics remain separate. Training is not implemented in Phase 2A.

Full data generation, extraction, integrity scoring and any future fitting must
run in Colab CPU. Local tests use only small invented fixtures. Outputs are exclusive
new directories; failures leave an incomplete directory without COMPLETE.json.
Never resume into or overwrite that directory. A new approved run needs a new name.
