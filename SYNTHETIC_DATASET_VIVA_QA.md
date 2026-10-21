# Synthetic Dataset — PP1 Viva Guide

## 1. How to introduce your dataset

“My component uses a synthetic, score-level multimodal dataset with 1,000 simulated participants. Each record contains motor, voice and handwriting scores and one of five simulated disease labels. It lets me train and demonstrate the complete fusion pipeline while a matched, clinically labelled multimodal dataset is unavailable. These are simulated records, not real patients.”

The five labels are **PD** (Parkinson’s disease), **ET** (essential tremor), **CA** (cerebellar ataxia), **AP** (atypical parkinsonism) and **DLB** (dementia with Lewy bodies). There is no healthy class in this version.

## 2. What each column means

| Column | Purpose | Used as model input? |
|---|---|---|
| `synthetic_id` | Identifier for a simulated participant. | No |
| `motor_pd_probability` | Simulated motor component PD score, from 0 to 1. | Yes |
| `voice_pd_probability` | Simulated voice component PD score, from 0 to 1. | Yes |
| `handwriting_pd_probability` | Simulated handwriting component PD score, from 0 to 1. | Yes |
| `label` | Simulated disease category the model learns to predict. | Target only |
| `provenance` | Marks the record as `synthetic_assumption_based`. | No |
| `split` | Identifies training, validation or test membership. | No |
| `scenario` | Identifies the `moderate_overlap` simulation. | No |
| `generator_noise_sd` | Records the simulation noise setting, `0.50`. | No |

**Important:** The three scores imitate component outputs; they are not actual predictions from your teammates’ models or clinically calibrated probabilities.

### Score patterns used for each label

We used a different combination of motor, voice and handwriting scores for each label. These are the exact starting profiles in our generator:

| Assigned label | Motor score | Voice score | Handwriting score | Pattern we assumed |
|---|---|---|---|---|
| **PD** | 0.78 | 0.74 | 0.74 | All three relatively high |
| **ET** | 0.76 | 0.32 | 0.68 | Higher motor and handwriting, lower voice |
| **CA** | 0.50 | 0.75 | 0.40 | Voice highest, motor intermediate, handwriting lower |
| **AP** | 0.68 | 0.65 | 0.38 | Higher motor and voice, lower handwriting |
| **DLB** | 0.38 | 0.48 | 0.62 | Handwriting highest, motor and voice lower |

The generator assigns a label first, then generates scores around that profile with random variation. These are starting profiles, not exact values for every record; the class distributions overlap.

**These profiles are illustrative simulation assumptions, not established clinical disease characteristics or values estimated from your teammates' datasets.**

## 3. Questions and short answers

**Why use synthetic data ?**

“For now i do not have verified matching participants across all three modalities with all five disease labels from other components. Joining unrelated people would not create a valid real multimodal cohort. Synthetic data supports a transparent implementation demonstration.”

**How did you create it? Where do the labels come from?**

“The generator assigns a simulated class first, then samples three correlated scores around assumed class profiles with random variation and noise. Classes overlap, so prediction is not an exact lookup. The labels come from the simulation, not clinical diagnosis.”

**Why 1,000 records and equal classes?**

“There are 200 records per class, giving every class equal representation for this prototype. This is a design choice, not real disease prevalence or proof that 1,000 records are clinically sufficient.”

**Did you preprocess it?**

“Yes. The pipeline selects and orders the three score inputs, validates numeric values, and rejects missing, infinite or out-of-range scores. Dataset checks inspect IDs, labels and duplicate score rows. Random Forest does not require additional scaling here because inputs already use a 0–1 scale. Raw audio, image and sensor preprocessing belongs to the upstream components.”

**How did you split it and check leakage?**

“The stratified split is 600 training, 200 validation and 200 test records. The saved pipeline is fitted on training records, and validation selects the configuration. Checks found no shared IDs or exact score rows between splits. Only the three scores enter the model; labels and metadata are excluded from inputs.”

**What accuracy can you claim?**

“The moderate-overlap test accuracy is 85.5%; a separate 500-record synthetic confirmation set gives 83.8%. These measure performance under simulation assumptions, not clinical accuracy.”

**Can it diagnose real patients now?**

“Clinical use requires matched real participant data, verified labels and independent clinical validation. The current model demonstrates the fusion workflow.”

## 4. What to show when asked

Open this folder in File Explorer:

`E:\University\Research\Shehani\multimodal_fushion_implementaion`

1. **Dataset:** Open `multimodal_fusion\data\synthetic_moderate_participants.csv`. Show its header, example rows, class counts and split column. The downloaded development CSV is suitable for this.
2. **Creation code:** Open `multimodal_fusion\synthetic.py` in a text editor. Find `PROFILES` and `def generate`. Explain assumed class profiles, label assignment and random score generation.
3. **Active settings:** Open `multimodal_fusion\scenarios.py`. Show `DATA_SEED=1401`, `NOISE=.50` and `generate(1000, DATA_SEED, noise=NOISE, prefix='MOD')`. This explicitly overrides the general generator’s harder default noise of `0.85`.
4. **Preprocessing code:** Open `multimodal_fusion\preprocessing.py`. Show `ScorePreprocessor` and `inspect_dataset` to explain input validation and dataset checks.

Showing the CSV answers “show your dataset.” Keep the code files ready for “show how you generated or processed it.” You do not need to retrain the model just to show these files.
