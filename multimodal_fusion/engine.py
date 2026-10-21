import hashlib, html
from datetime import datetime, timezone
from threading import Lock
import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from .settings import ARTIFACTS, FEATURES, LABELS, NAMES, SCOPE

class FusionEngine:
    def __init__(self, model_path=None):
        path=model_path or ARTIFACTS/"fusion_model.joblib"
        self.bundle=joblib.load(path); self.model=self.bundle["model"]
        self.pipeline=self.bundle.get('pipeline',self.model)
        if self.bundle["feature_columns"]!=FEATURES or self.bundle["labels"]!=LABELS:
            raise ValueError("Model schema does not match component configuration")
        self.model_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        self.explainer=shap.TreeExplainer(self.model); self.lock=Lock()

    def predict(self,scores,include_lime=False,session_id="PP1-demo"):
        if set(scores)-set(FEATURES): raise ValueError("Unknown score field")
        values=[]; missing=[]
        for name in FEATURES:
            value=scores.get(name)
            if value is None: missing.append(name); values.append(None); continue
            if isinstance(value,bool) or not isinstance(value,(int,float,np.number)) or not np.isfinite(value) or not 0<=value<=1:
                raise ValueError(f"{name} must be a finite number between 0 and 1")
            values.append(float(value))
        result={"version":self.bundle["version"],"model_sha256":self.model_hash,"session_id":str(session_id)[:100],
                "created_utc":datetime.now(timezone.utc).isoformat(),"evidence_scope":self.bundle.get('scope',SCOPE),"clinical_validation":False,"inputs":dict(zip(FEATURES,values)),
                "scenario":self.bundle.get('scenario','historical_hard'),"preprocessing":self.bundle.get('preprocessing',{'steps':['Validate and order 0–1 scores']})}
        if missing:
            return {**result,"status":"incomplete","missing_inputs":missing,"predicted_class":None,
                    "likeness_profile":None,"warnings":["Complete all three component scores; no prediction was fabricated."]}
        x=pd.DataFrame([values],columns=FEATURES); raw=self.pipeline.predict_proba(x)[0]
        model_labels=list(self.model.classes_); p=raw[[model_labels.index(label) for label in LABELS]]
        chosen=int(np.argmax(p)); label=LABELS[chosen]; index=model_labels.index(label); sorted_p=np.sort(p)
        flags=[]
        if float(p.max())<.50 or float(sorted_p[-1]-sorted_p[-2])<.10: flags.append("Ambiguous simulated profile; review the class probabilities.")
        if max(values)-min(values)>=.50: flags.append("Component scores disagree substantially.")
        explanation=self.explainer.shap_values(x); contributions=np.asarray(explanation)[0,:,index]
        baseline=float(np.asarray(self.explainer.expected_value)[index])
        if not np.isclose(baseline+contributions.sum(),raw[index],atol=1e-6): raise RuntimeError("SHAP additivity check failed")
        shap_report={"method":"TreeSHAP","explained_class":label,"baseline_probability":baseline,"predicted_probability":float(raw[index]),
            "additivity_error":float(abs(baseline+contributions.sum()-raw[index])),
            "contributions":[{"feature":f,"value":v,"contribution":float(c)} for f,v,c in zip(FEATURES,values,contributions)],
            "interpretation":"Contributions explain this simulator-trained model, not medical causes. They are not modality weights."}
        result.update(status="review" if flags else "complete",predicted_class=label,predicted_class_name=NAMES[label],
            confidence=float(p.max()),probability_margin=float(sorted_p[-1]-sorted_p[-2]),
            likeness_profile={label:float(prob) for label,prob in zip(LABELS,p)},warnings=flags,shap=shap_report)
        if include_lime:
            with self.lock:
                lime=LimeTabularExplainer(self.bundle["training_background"],feature_names=FEATURES,class_names=model_labels,
                    mode="classification",random_state=self.bundle["seed"],discretize_continuous=True)
                def predict_array(array): return self.pipeline.predict_proba(pd.DataFrame(array,columns=FEATURES))
                local=lime.explain_instance(np.asarray(values),predict_array,labels=[index],num_features=3,num_samples=1800)
                result["lime"]={"method":"LIME","explained_class":label,
                    "conditions":[{"condition":condition,"contribution":float(weight)} for condition,weight in local.as_list(label=index)],
                    "local_surrogate_r2":float(local.score),
                    "interpretation":"Approximate local surrogate; its weights need not match SHAP or sum to the model probability."}
        return result

def report_html(result):
    escape=lambda v:html.escape(str(v))
    names=dict(zip(FEATURES, ['Motor score', 'Voice score', 'Handwriting score']))
    profile=result.get("likeness_profile") or {}
    rows="".join(f'<tr class="{"selected" if k == result.get("predicted_class") else ""}"><td>{escape(NAMES[k])} ({escape(k)})</td><td>{v*100:.2f}%</td></tr>' for k,v in profile.items())
    inputs="".join(f"<tr><td>{escape(names.get(k,k))}</td><td>{'Unavailable' if v is None else f'{v:.6f}'.rstrip('0').rstrip('.')}</td></tr>" for k,v in result["inputs"].items())
    shap=result.get('shap',{})
    contributions="".join(f"<tr><td>{escape(names.get(r['feature'],r['feature']))}</td><td>{r['contribution']*100:+.2f} pp</td></tr>" for r in shap.get('contributions',[]))
    if profile:
        summary=f'<div class="outcome"><span>Highest-scoring class</span><h2>{escape(result["predicted_class_name"])} <strong>{result["confidence"]*100:.2f}%</strong></h2></div>'
        explanation=f'<p class="note">Explaining {escape(shap["explained_class"])}: reference score {shap["baseline_probability"]*100:.2f}% → prediction {shap["predicted_probability"]*100:.2f}%.</p>'
    else:
        summary='<div class="outcome"><h2>Incomplete assessment</h2><p>All three input scores are required.</p></div>'
        explanation=''
    warnings=" ".join(result.get('warnings',[]))
    warning_html=f'<p class="notice">{escape(warnings)}</p>' if warnings else ''
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Multimodal fusion report</title>
    <style>body{{font:15px/1.6 "Segoe UI",Arial,sans-serif;color:#193741;max-width:780px;margin:35px auto;padding:24px}}.heading{{border-bottom:2px solid #087e8b;padding-bottom:18px}}.heading p{{margin:0;color:#5e747b;font-size:12px}}h1{{font-size:28px;line-height:1.25;margin:8px 0 10px}}h2{{font-size:19px;margin:25px 0 10px}}table{{border-collapse:collapse;width:100%;margin:10px 0 24px}}td,th{{text-align:left;border-bottom:1px solid #dce7e8;padding:10px 12px}}th{{font-size:12px;color:#5e747b}}td:last-child,th:last-child{{text-align:right}}.selected{{background:#edf7f7;font-weight:600}}.outcome{{background:#edf7f7;padding:16px 20px;margin:22px 0;border-radius:6px}}.outcome span,.note{{font-size:12px;color:#5e747b}}.outcome h2{{margin:5px 0}}.outcome strong{{float:right;color:#087e8b}}.notice{{font-size:13px;color:#795a22}}button{{font:inherit;background:#087e8b;color:white;border:0;border-radius:5px;padding:10px 18px;cursor:pointer}}@media print{{body{{margin:0;max-width:none;padding:12px}}button{{display:none}}.outcome,.selected{{print-color-adjust:exact}}}}</style></head><body>
    <header class="heading"><p>COMPONENT 4</p><h1>Multimodal fusion report</h1><p>Synthetic data demonstration · Not clinically validated</p></header>
    {summary}{warning_html}
    <h2>Input scores</h2><table><thead><tr><th>Component</th><th>Score (0–1)</th></tr></thead><tbody>{inputs}</tbody></table>
    <h2>Class estimates</h2><table><thead><tr><th>Simulated class</th><th>Model score</th></tr></thead><tbody>{rows or '<tr><td colspan="2">No prediction available.</td></tr>'}</tbody></table>
    <h2>Prediction explanation · SHAP</h2>{explanation}<table><thead><tr><th>Component</th><th>Contribution</th></tr></thead><tbody>{contributions or '<tr><td colspan="2">Unavailable for incomplete input.</td></tr>'}</tbody></table>
    <p class="note">Contributions are percentage-point changes (pp) from the reference score. Positive values increase the selected class score; negative values decrease it.</p>
    <button onclick="window.print()">Print / save PDF</button></body></html>'''
