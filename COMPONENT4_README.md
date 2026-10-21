# Component 4 · moderate-overlap PP1 demonstration (v3.0)

This local implementation combines three simulated component scores (MPS, VPS, HPS) using a saved preprocessing + Random Forest pipeline. It returns a five-class profile for PD/ET/CA/AP/DLB, TreeSHAP and LIME explanations, review warnings, JSON output and a printable report. A missing input produces an incomplete assessment.

**Scope:** all five-condition profiles are assumptions. The active moderate-overlap scenario (noise SD 0.50) was chosen after a sensitivity study and user approval. This is a complete input-to-report software demonstration, not clinically validated differential diagnosis. Healthy is not a supported class. Original patient archives were not used to manufacture matched multimodal patients.

## Start

Double-click START_COMPONENT4.cmd, or run `.\.venv\Scripts\python.exe -m multimodal_fusion serve` from the repository folder. The Assessment page is http://127.0.0.1:8040 and Evaluation is /evaluation. Dataset exploration (/dataset), API reference (/docs) and preprocessing evidence (/api/preprocessing) remain available directly; they are not dashboard navigation tabs. The server listens only on this computer. No internet is needed for the trained demonstration.

## Reproduce

Use Python 3.12. On another Windows computer, open a terminal in this repository folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-component4.txt
```

Then double-click `START_COMPONENT4.cmd`. The active saved model is included, so retraining is not needed for the demonstration. Do not copy an existing `.venv` between computers.

To reproduce training and checks, run:

```powershell
.\.venv\Scripts\python.exe -m multimodal_fusion train
.\.venv\Scripts\python.exe -m multimodal_fusion audit
.\.venv\Scripts\python.exe -m pytest tests_component4 -q
```

The retained original hard CSV and stress CSV must be available under multimodal_fusion/data. Training will not silently recreate or overwrite them. The active `artifacts/fusion_model.joblib` and the archived baseline model needed by preservation tests are included. Challenger models, duplicate generated reports/charts, logs and local environments are ignored. Load only artifacts created by this project, not untrusted joblib files.

## Active training protocol

- Generate 1,000 moderate-overlap synthetic records, seed 1401, 200 per class. Profile centers are explicitly assumed in synthetic.py; labels are selected before overlapping scores are generated.
- Stratify into 600 training / 200 validation / 200 test participants. Metadata is excluded from features.
- Compare four Random Forest configurations with five-fold training CV; select on validation macro F1. Fit preprocessing/model only on the 600 training records.
- Evaluate once on the reserved 200-record cohort test, then generate a separate 500-record confirmation sample with fixed seed 1402 after selection. Both samples are now inspected; subsequent model changes need new reserved evaluation.
- Compare with logistic fusion and three single-input baselines on the same moderate test.
- Evaluate the moderate-trained model on the retained original hard test (200 already-inspected records, noise .85) and the retained imbalanced stress sample. These are robustness benchmarks, not independent clinical validation.

Read measured results from artifacts/evaluation.json. The current run records moderate test accuracy 85.5%, macro F1 0.853, confirmation accuracy 83.8%, and retained hard-benchmark accuracy 62.5%. Higher moderate performance is a change of synthetic assumptions, not proof that the original hard task was solved.

## Preprocessing

Read PREPROCESSING_GUIDE.md. The fitted pipeline checks schema, excludes metadata, orders features, validates numeric/finite/range values and passes the existing [0,1] scores to the forest. It does not invent extra image/audio preprocessing. No scaler is needed for this tree; the logistic baseline has a training-only scaler. Missing live inputs are not silently imputed.

## Prediction contract

POST /api/predict:

```json
{"motor_pd_probability":0.78,"voice_pd_probability":0.72,"handwriting_pd_probability":0.69,"session_id":"PP1-example","include_lime":true}
```

The score must mean probability of PD, not confidence in whichever class was predicted. Return fields include version, model checksum, scenario, input values, preprocessing evidence, five-class profile and explanations. Probabilities sum to one over the supported simulated classes; they are not validated disease risk. POST the same body to /api/report for printable HTML. The interface stores no patient records.

TreeSHAP baseline plus contributions is checked against the selected class probability. Contributions are signed model explanations, not modality weights. LIME is an approximate local surrogate; its R² is displayed. Engineering review thresholds are not clinical thresholds.

## Data and evidence

- data/synthetic_moderate_participants.csv: active 1,000-record demonstration cohort.
- data/synthetic_moderate_fresh_test.csv: 500-record moderate confirmation sample.
- data/synthetic_participants.csv: original hard cohort, preserved unchanged.
- artifacts/evaluation.json and preprocessing_report.json: active results, protocol, checksums and preprocessing state.
- artifacts/test_predictions.csv, moderate_fresh_predictions.csv, hard_benchmark_predictions.csv: inspectable outcomes.
- artifacts/baseline_v1/: archived original hard-model artifacts.
- artifacts/improvement_evaluation.json: historical harder-scenario challenger study, not active model performance.
- artifacts/leakage_and_simulation_audit.json: active pipeline audit.

The dataset explorer separates each source and supports CSV downloads and sending a row to the active moderate-trained model. Demo presets use fixed validation IDs, not successful test examples.

## Guides

- ASSESSMENT_VIVA_GUIDE.md: prediction demonstration and code walkthrough.
- EVALUATION_VIVA_GUIDE.md: evaluation presentation, metrics and code walkthrough.
- SYNTHETIC_DATASET_VIVA_QA.md: dataset purpose, columns, assumed profiles and generation code.
- PREPROCESSING_GUIDE.md: score validation and saved pipeline details.

Older duplicate guides remain local and are excluded from Git. The committed JSON/CSV evidence retains the original harder benchmark and historical comparison required for reproducibility; those results are not the active model's main dashboard metrics.

## Later research

Obtain supervisor/clinical review of class-profile assumptions and the remembered disease-interval instruction. Integrate real component outputs through agreed score/calibration contracts. Proper five-disorder research needs matched participant scores and appropriate specialist labels, plus a healthy/non-target policy and independent validation. A three-score synthetic classifier does not establish those capabilities.
