"""Create inspectable explanation and robustness evidence after training."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .engine import FusionEngine, report_html
from .settings import ARTIFACTS, DATA, FEATURES, LABELS, SEED,ACTIVE_DATASET
from .train import metrics

def generate_explanation_evidence():
    engine=FusionEngine()
    data=pd.read_csv(DATA/engine.bundle.get('dataset_filename',ACTIVE_DATASET))
    test=data[data.split=="test"]
    values=np.asarray(engine.explainer.shap_values(test[FEATURES]))
    global_importance=np.abs(values).mean(axis=(0,2))
    fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
    ax.barh(["Motor","Voice","Handwriting"],global_importance,color="#087e8b")
    ax.set(xlabel="Mean absolute SHAP contribution across five classes",title="Global model explanation · 200 synthetic test participants")
    ax.invert_yaxis();fig.savefig(ARTIFACTS/"shap_global.png",dpi=150);plt.close(fig)
    cases=json.loads((ARTIFACTS/"demo_cases.json").read_text())
    explanations=[engine.predict(case["scores"],include_lime=True,session_id=case.get("synthetic_id",case["name"])) for case in cases]
    (ARTIFACTS/"example_explanations.json").write_text(json.dumps(explanations,indent=2),encoding="utf-8")
    (ARTIFACTS/"example_report.html").write_text(report_html(explanations[0]),encoding="utf-8")
    robustness={"scope":"Engineering checks and synthetic score perturbations only","missing_each_input":{},"noisy_test_scores":{}}
    for missing in FEATURES:
        scores={feature:.7 for feature in FEATURES};scores[missing]=None
        response=engine.predict(scores)
        robustness["missing_each_input"][missing]={"status":response["status"],"prediction_returned":response["predicted_class"] is not None}
    rng=np.random.default_rng(SEED+200)
    noisy=test.copy();noisy[FEATURES]=np.clip(noisy[FEATURES]+rng.normal(0,.10,(len(test),3)),0,1)
    robustness["noisy_test_scores"]=metrics(engine.pipeline,noisy)
    (ARTIFACTS/"robustness.json").write_text(json.dumps(robustness,indent=2),encoding="utf-8")
    print("Generated local SHAP/LIME examples, global SHAP chart, report, and robustness evidence.")

if __name__=="__main__":generate_explanation_evidence()
