# Before model training

* **Do not label these samples as real patients**. HC-like and symptom-like are simulation profiles.
* **Do not call these labels clinically rated TS/BS/RS/GI 0–4 values**. Existing engine outputs are temporary rule pseudo-labels.
* A model can show 95%+ **synthetic-rule imitation accuracy** without having any real-world clinical scoring value; do not advertise it as clinical performance.
* For a demo, use synthetic subject-disjoint splits and report `synthetic simulator benchmark` as the evaluation.
* For actual PP1 evaluation use original portable-device data + independent clinician ratings on the same subjects.
* Missing sensor channels and missing whole tests must be flagged; do not invent absent tap events or label invalid MPS.
* PADS trained RF/SVM/XGBoost/CNN/Hybrid CNN models are PD-vs-Healthy research classifiers with nonmatching protocols. Keep checkpoints separate.
