# Assessment Tab — Viva Presentation Guide

Allow about 3–4 minutes. This guide covers the live prediction demonstration; use the separate Evaluation guide for test results.

## Before starting

Open the project folder and double-click **START_COMPONENT4.cmd**. Keep its command window open. Visit http://127.0.0.1:8040/ and select **Assessment**. Refresh with Ctrl+F5 if an older design appears.

The application uses the saved trained model. You do not need to retrain it during the viva.

## 1. Introduce the component

Say:

> “My component combines motor, voice and handwriting scores using a Random Forest fusion model. It produces a five-class prediction, explains the result using SHAP and LIME, and generates a report. This prototype currently uses synthetic score patterns; it is not clinically validated.”

The supported classes are PD, ET, CA, AP and DLB. A healthy class is not included. The three inputs imitate upstream component outputs; live integration with teammates’ models is still future work.

## 2. Load an example and run it

Choose **ET example** from **Example record**, then click **Run assessment**. Selecting an example only loads its scores; click the button to update the prediction.

Say:

> “Each input is a PD-related component score between zero and one. The model uses the combination of all three scores. The example’s assigned synthetic class is shown for reference, but the label is not passed to the prediction model.”

The examples are fixed validation records, not real patients. An assigned class and a predicted class can differ.

## 3. Explain the prediction

Point to the highest-scoring class and the five bars.

Say:

> “The highest-scoring class is the model’s selected prediction. The bars show its probability estimates across the five supported simulated classes and total approximately 100%, allowing for rounding. They are not validated patient disease risks.”

**Do not call this percentage model accuracy.** A record’s prediction score and the model’s test accuracy are different quantities.

## 4. Explain SHAP

Show **Prediction explanation**.

Say:

> “SHAP explains how each input moves the selected class score away from a reference value. Positive contributions increase that score, while negative contributions decrease it. The reference plus all contributions equals the final prediction, apart from displayed rounding.”

For an ET result showing reference 19.9% and contributions +0.231, +0.472 and +0.073:

**19.9% + 23.1 + 47.2 + 7.3 percentage points ≈ 97.5%.**

Use the actual values on screen; they change with the inputs. The reference is calculated by SHAP from the trained model, not manually fixed at 20%. Contributions are not fixed modality weights or medical causes.

## 5. Show LIME briefly

Leave **Calculate LIME explanation** selected, then expand **LIME explanation** after running.

Say:

> “LIME slightly changes the input scores and observes the model’s outputs. It builds a simpler explanation around this example, showing which score conditions support or oppose the selected class.”

A positive condition supports the selected class; a negative one opposes it in the approximation. LIME weights are not percentages and do not need to match SHAP or total 100%. Its thresholds are explanation intervals, not clinical rules.

If asked about **local fit R²**:

> “It measures how well LIME’s simpler explanation matches the model around this example. It is not prediction accuracy. For example, 0.549 indicates an imperfect local fit.”

## 6. Demonstrate that the result responds to inputs

Change one score and click **Run assessment** again.

Say:

> “The application calculates a new prediction and explanation using the saved model. It does not simply display a hardcoded result.”

The top class may stay the same; compare its probability and SHAP contributions instead. Changing a score does not retrain the model.

## 7. Demonstrate missing input and the report

Under Voice, select **Mark unavailable**, then run again. Show the incomplete assessment.

Say:

> “The current model requires all three scores. If one is missing, the application reports an incomplete assessment instead of inventing a score.”

Uncheck it and run again. Click **View report** to show input scores, class estimates and SHAP contributions. Use **Print / save PDF** if requested. **Download JSON** exports structured results for inspection or future integration.

Export buttons are disabled after input changes until a new assessment is run. Reports and downloads do not mean a patient database is connected; this prototype uses local files.

## If the panel asks to see the repository code

Open files in a text editor and search for the named function. Explain the relevant section rather than reading the entire file. Links below refer to files in this repository. If the project is moved, open the same files inside its new folder.

| Request | File to open | What to show |
|---|---|---|
| Show the dashboard | [index.html](multimodal_fusion/web/index.html) and [app.js](multimodal_fusion/web/app.js) | Page layout; `request()`, `predict()` and `render()` collect scores, call the API and display results. |
| Show the backend connection | [api.py](multimodal_fusion/api.py) | `FusionRequest`, `/api/predict` and `/api/report`: validate requests and call the prediction engine. |
| Show prediction and explanations | [engine.py](multimodal_fusion/engine.py) | `FusionEngine.predict()`: saved pipeline prediction, missing-input handling, TreeSHAP and LIME. `report_html()` creates the printable report. |
| Show preprocessing | [preprocessing.py](multimodal_fusion/preprocessing.py) | `ScorePreprocessor`: orders scores and validates numeric, finite and 0–1 values. It is shared by training and prediction. |
| Show how the model was trained | [scenarios.py](multimodal_fusion/scenarios.py) | `train_pp1()`: active generation settings, data splits, preprocessing + Random Forest pipeline, validation selection and evaluation. |
| Show synthetic data generation | [synthetic.py](multimodal_fusion/synthetic.py) | `PROFILES` and `generate()`: assumed class profiles and label-first score simulation. Active noise 0.50 is passed from `scenarios.py`. |
| Show the dataset | [synthetic_moderate_participants.csv](multimodal_fusion/data/synthetic_moderate_participants.csv) | Three input columns, simulated label and split; 1,000 generated records. |
| Where is the trained model? | [fusion_model.joblib](multimodal_fusion/artifacts/fusion_model.joblib) | The saved model bundle loaded for predictions. It is a binary artifact; show its location, not its contents in a text editor. |

**Short code walkthrough:** Start with `app.js` → `api.py` → `engine.py`. Open training or generation files only when asked how the model or dataset was created.
