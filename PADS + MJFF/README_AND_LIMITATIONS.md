# PP1 Reference-informed Synthetic Dataset v2

**SIMULATED. NOT A DATASET OF REAL HEALTHY/PARKINSON'S PATIENTS.**

This archive contains **200 independent fully synthetic assessment sessions**:
- Healthy-like motor patterns (simulated): 100
- Parkinsonian-symptom-like motor patterns (simulated): 100
- Four CSVs per session: TS 10s accel, BS 10s capacitive-touch timestamps, RS 10s active wrist gyro, GI 180s wrist accel.
- Sample rate: 100 Hz for accelerometer/gyroscope; taps are event timestamps.
- Acceleration units: m/s^2, gyro: rad/s. Exact original PP1 software input column names are used.
- No real person or patient is represented by a case ID. A case is not a clinically diagnosed patient.

## Reference provenance (critical)

Research source descriptions support the choice of tasks and the broad movement modalities, but **do not fully determine the synthetic signals**.

* **GI numeric anchors actually used**: a user-provided, previously extracted 51-person wrist gait feature table. It has 23 healthy-control and 28 PD-labeled records. Group-specific gait-frequency and dynamic-acceleration RMS measurements were summarized and used as contextual numeric anchors for simulated 180-second gait. Neither original real signals nor real participant identifiers were copied. This table is associated with a WearGait-PD derived workflow, but access rights should be checked before redistribution.
* **TS tremor / RS wrist mobility**: PADS and Arm Tremor public repositories provide real clinical control/symptomatic modality references, but their raw participant-level recordings were NOT downloaded or numerically fitted in this build. Frequency/amplitude and movement-velocity rules are **simulation assumptions**, not extracted real-dataset distributions. RS is NOT a measurement of clinical rigidity.
* **BS tapping**: healthy smartphone finger tapping dataset (30s) and mPower (20s healthy/PD with access registration) are literature references. Neither dataset was accessed for numeric BS fitting; the 10-second capacitive timestamps were simulated with hand-selected rate/irregularity settings. Finger tapping amplitude is unmeasured.

See `public_source_reference_status.csv` and verified URLs. The fact that clinical datasets are listed as sources does NOT mean a case is derived from a real patient's four tests.

## Data format

Each `cases/HC_SYN_001/` or `cases/SYM_SYN_001/` has:

| File | Header | Samples / range |
|---|---|---|
| `TS_10s.csv` | time_s,ax_ms2,ay_ms2,az_ms2 | 1000 rows, 0–9.99s |
| `BS_10s.csv` | tap_time_s | timestamps in [0,10) |
| `RS_10s.csv` | time_s,gx_rad_s,gy_rad_s,gz_rad_s | 1000 rows, 0–9.99s |
| `GI_180s.csv` | time_s,ax_ms2,ay_ms2,az_ms2 | 18000 rows, 0–179.99s |

`features_extracted_by_existing_PP1_engine.csv` contains actual engine feature extraction results. `demo_rule_scores_for_fusion.csv` contains pseudo-labels from the **uncalibrated existing PP1 software rules**. Formula `MPS=(TS+BS+RS+GI)/16` is a temporary normalized demo index, NOT the probability or severity of Parkinson's disease. In real-device mode the existing engine must continue outputting `scores=None, mps=None` pending calibration.

The unverified hardware protocol should be mapped through a future adapter rather than adjusting hardware or inventing extra sensors.

## Research and training boundaries

This is suitable for software integration and controlled simulator experiments. It cannot substantiate clinical exact severity accuracy, independent patient diagnosis performance, or 85% real-world MPS accuracy. Any Random Forest/XGBoost/CNN trained to reproduce these scores would learn the simulator and the software thresholds. Models trained on PADS 9-second wrist data cannot directly score these four device tasks.

`synthetic_cohort_profile` records simulated group generation, not clinical condition. It must never be reported as an observed medical label.

For independent evaluation collect real PP1-device signals with clinician-rated reference 0–4 ratings and appropriate oversight; keep training/validation/test subjects disjoint.

## Public data sources

* PADS, PhysioNet: https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/ (CC BY-NC-SA 4.0)
* WearGait-PD: https://www.nature.com/articles/s41597-026-06806-2 (Synapse registration / CC BY 4.0)
* Brazilian finger tapping healthy reference: https://doi.org/10.6084/m9.figshare.26940823.v1
* mPower motor tasks: https://www.synapse.org/mPower (Synapse access conditions)
* Arm Tremor Database: https://zenodo.org/records/4772130 (CC BY 4.0)

## Caution on licenses

This package includes independently generated simulated numerical values and derived group summary statistics, not redistributed patient-level datasets. Source dataset access and attribution terms must be honored when you later retrieve real records. Preserve this provenance file in any academic use.
