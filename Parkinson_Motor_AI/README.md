# PADS Motor Assessment ML — THREE MODELS (RF, SVM, XGBoost)

**Purpose:** Preliminary **PD vs Healthy** classification on publicly available PADS wrist movement recordings, not direct 0–4 TS/RS/BS/GI labels. This does not modify your physical device or replace the previous developer's equations.

## A. Data sources and compatibility
- PADS: https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/ — open access; 100-Hz accelerometer/gyroscope; diagnostic labels from `patients/patient_XXX.json`. `Parkinson's`=1, `Healthy`=0, exclude differential/other conditions from this initial binary experiment. PADS tests are 10.24s (some 20.48s); this software uses 9-second cleaned subsets for public-data benchmarking, **not** a change to your device's fixed 10-second protocol.
- Device clinical feature tasks remain: TS 10s (accel), RS 10s (gyro movement proxy), BS 10s (capacitive tap timestamps), GI 180s (accel).
- For BS, request mPower controlled access separately: https://www.synapse.org/Synapse:syn4993293/wiki/247860 . Open alternative for provisional tapping-pattern experiments: https://github.com/innovation-center-etf/FINGER-TAPPING-PARKINSONISMS-DATABASE (gyro on fingers, not capacitance).
- For GI, PhysioNet gait dataset: https://physionet.org/content/gaitpdb/1.0.0/ (foot-force, **not** MPU6050; separate exploratory feature study).
- No public dataset above gives complete matched [TS, BS, RS, GI] measurements on YOUR hardware and independently confirmed target. Final MPS requires those data later.

## B. Download PADS
Unzip the official dataset (about 735MB compressed; approximately 1.4GB uncompressed). Make sure this structure exists inside project root:

```
data/pads_dataset/
    patients/patient_001.json
    movement/observation_001.json
    movement/timeseries/001_Relaxed_LeftWrist.txt
```

The `patients/` and `movement/` folders must be direct children of `data/pads_dataset` — **not** double nested.

## C. Setup (Python 3.10+)
From this project's top-level folder:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## D. Data preparation and training

```bash
python -m scripts.prepare --root data/pads_dataset --out outputs
python -m scripts.train_classical --out outputs
python -m scripts.select_best --out outputs
```

The first command creates `features.csv`, `metadata.csv`, `participant_splits.csv`, `signals.npy`. The second trains three pipelines, saves fitted `.joblib` files and saves **validation** comparisons. The third chooses the winner on validation results only and evaluates it once on held-out **test participants**.

## E. Exact outputs

- `outputs/classical_validation_metrics.csv` — side-by-side RF/SVM/XGBoost comparison
- `outputs/RandomForest_pipeline.joblib`, `SVM_pipeline.joblib`, `XGBoost_pipeline.joblib` — trained candidates
- `outputs/model_validation_ranking.csv` — validation-ranked models
- `outputs/selected_test_metrics.json` — final held-out participant-level metrics
- `outputs/selected_test_subject_predictions.csv` — participant-level predictions

Models use 18 engineered movement features plus task/wrist indicators, with training-only preprocessing. Participant-wise splitting is enforced; no person appears in more than one data partition. Check group counts, class counts and whether label categories are adequately represented before accepting metrics.

## F. Test the code without real data (for software verification only)

```bash
python -m scripts.make_synthetic_demo --root data/synthetic_demo --n 48
python -m scripts.prepare --root data/synthetic_demo --out outputs/demo
python -m scripts.train_classical --out outputs/demo
python -m scripts.select_best --out outputs/demo
python -m pytest -q
```

**Do not use synthetic example scores as clinical/research results.**

## G. Scientific limitations
- PADS train/val/test metrics only evaluate PD vs healthy identification using a wrist-watch dataset under its own protocol. They **do not** validate TS/RS/BS/GI 0–4 prediction for your device.
- RS gyroscope voluntary movement is a mobility proxy, **not** direct passive rigidity.
- Unmodified old device equations are the engineering baseline; numerical improvements can only be established on appropriately labelled comparable device recordings.
- Do not replace PADS labels with labels from any formula and claim it is supervised clinical severity learning.
- The current dataset is under a noncommercial/share-alike research license; follow the official license, and don't redistribute PADS recordings in a public repo.
