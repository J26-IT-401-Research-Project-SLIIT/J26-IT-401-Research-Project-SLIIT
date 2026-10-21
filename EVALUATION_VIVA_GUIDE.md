# Evaluation Tab — Viva Presentation Guide

Open the application and click **Evaluation**. Allow about 2–3 minutes for this part.

## 1. Introduce the evaluation

Say:

> “This page shows the Random Forest fusion model’s performance on 200 held-out synthetic test records, with 40 records for each of five classes. The full dataset has 1,000 records: 600 for training, 200 for validation and 200 for testing. Validation data was used to select the Random Forest configuration.”

**Be clear:** These results evaluate assumed synthetic patterns, not real-patient diagnostic accuracy.

## 2. Show the four main metrics

| Metric | Result | What to say |
|---|---:|---|
| Accuracy | 85.5% | “The model correctly classified 171 out of 200 test records.” |
| Macro precision | 85.5% | “For each class, we measure how often predictions of that class are correct, then average across classes.” |
| Macro recall | 85.5% | “For each class, we measure how many records belonging to that class were identified, then average across classes.” |
| Macro F1 | 85.3% | “This balances precision and recall, averaged across the five classes.” |

**Macro** means each class has equal importance. The first three metrics happen to round to the same number; they are different measures. An F1 of 85.3% is the same value as approximately 0.853.

## 3. Explain “Results by class”

Say:

> “The class table shows where the model performs better or worse. For example, PD recall is 95%, meaning 38 of 40 PD records were identified correctly. AP recall is 70%, meaning 28 of 40 AP records were identified correctly. This shows that the model has more difficulty with AP.”

If asked why accuracy is absent from this table:

> “Overall accuracy is shown at the top. Precision, recall and F1 give a more useful breakdown for individual classes.”

## 4. Expand “More evaluation details”

### Training and confirmation

Say:

> “A separate 500-record sample generated from the same synthetic assumptions achieved 83.8% accuracy. This checks performance on another simulated sample; it is not independent clinical validation.”

### Confusion matrix

Point to the axis labels, then say:

> “Rows show assigned synthetic classes and columns show predicted classes. Diagonal cells are correct predictions; other cells are errors. For example, seven AP records were predicted as CA, showing confusion between those two classes.”

Each row totals 40. The diagonal totals 171 correct predictions.

### Model comparison

Say:

> “We compared the three-input fusion models with models using only one score. Random Forest fusion achieved 85.5%, compared with 48.5% for motor only, 42.5% for voice only and 39.5% for handwriting only. Combining inputs performed better on this synthetic test.”

Then acknowledge:

> “Logistic regression fusion achieved 86.5%, slightly higher than Random Forest. That is two more correct predictions out of 200. Random Forest is our deployed tree-based prototype; this comparison does not show it is the best-performing model.”

The single-input models here are comparison models trained on the synthetic scores. **These are not the original accuracies of the teammates’ motor, voice and handwriting models.**

## 5. Likely panel questions

**Why use Random Forest if logistic regression scored higher?**

“We implemented Random Forest as the tree-based fusion prototype. The simpler baseline scored slightly higher on this test, so we report that openly. Further model selection should use validation data and a new reserved evaluation set, rather than repeatedly tuning against this inspected test set.”

**Does this prove multimodal fusion works for real patients?**

“It demonstrates an improvement over single-input baselines under our simulation assumptions. Matched real participant scores and verified diagnoses are needed to assess clinical performance.”

**Was the test set used for training?**

“The saved model and preprocessing pipeline were fitted on the 600 training records. Validation results selected the Random Forest configuration. The 200 test records were used to evaluate it.”

## Closing sentence

> “The evaluation demonstrates a working fusion prototype and makes its class-level errors visible. The next step is integration with real component outputs and validation using matched clinical data.”

**Presentation tip:** Show the main metrics and class table first. Expand the additional details when discussing errors, confirmation or comparisons. Do not describe 85.5% as real clinical accuracy or claim Random Forest outperformed logistic regression.


## If the panel asks to see the repository code

Open the relevant file in a text editor. Search for the function or field listed below and explain that section. Links below refer to files in this repository. If the project is moved, open the same files inside its new folder.

| Panel request | File to open | What to show and say |
|---|---|---|
| Show the Evaluation page code | [evaluation.html](multimodal_fusion/web/evaluation.html) and [evaluation.js](multimodal_fusion/web/evaluation.js) | The HTML defines the page. The JavaScript fetches saved results from `/api/evaluation` and displays metrics, class results and comparisons. The numbers are not manually typed into the dashboard. |
| How does the page receive results? | [api.py](multimodal_fusion/api.py) | Find `/api/evaluation` and `evaluation()`. The endpoint reads the saved evaluation JSON. Opening the page does not retrain or retest the model. |
| Where are accuracy, precision, recall and F1 calculated? | [train.py](multimodal_fusion/train.py) | Find `metrics()`. It compares assigned labels with model predictions using scikit-learn metrics. `classification_report` provides class-level and macro-average precision, recall and F1; `accuracy_score` provides overall accuracy. |
| Show the split and model comparison code | [scenarios.py](multimodal_fusion/scenarios.py) | Find `train_pp1()`. Show stratified splitting, training-only fitting, validation selection, and evaluation of Random Forest, logistic regression and the three single-input models. |
| How was the confusion matrix created? | [train.py](multimodal_fusion/train.py) and [scenarios.py](multimodal_fusion/scenarios.py) | `metrics()` calculates `confusion_matrix` from assigned and predicted labels. `plot_scenarios()` draws and saves the chart. Class ordering is explicitly specified. |
| Show the recorded results | [evaluation.json](multimodal_fusion/artifacts/evaluation.json) | Find `results` → `Random Forest fusion`. Show `accuracy`, `classification_report`, `macro avg` and `confusion_matrix`. `fresh_results` contains the separate confirmation results. This file is evidence, not source code. |
| Can we inspect individual test predictions? | [test_predictions.csv](multimodal_fusion/artifacts/test_predictions.csv) | Show the saved test outcomes to inspect individual predictions and errors behind the summary metrics. |

**Quick walkthrough:** `evaluation.js` displays results → `api.py` serves `evaluation.json` → `scenarios.py` runs the experiment → `train.py` calculates the metrics.

Show the saved evidence during the viva. Retrain only if specifically requested and there is time; keep the current results available for the demonstration.
