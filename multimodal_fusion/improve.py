"""Fixed-data tree comparison and validation-only review rule, version 2.

Do not modify simulator profiles or noise to reach an accuracy target. The old
200-record test is retained as historical evidence. Final v2 evaluation uses a
predeclared 500-record synthetic sample only after model/gate selection.
"""
import hashlib, json, shutil
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .settings import ARTIFACTS,DATA,FEATURES,LABELS,SEED,SCOPE
from .synthetic import generate
from .train import metrics,probabilities

def accepted_mask(probabilities_array,score_array,threshold):
    ordered=np.sort(probabilities_array,axis=1)
    return (ordered[:,-1]>=threshold)&((ordered[:,-1]-ordered[:,-2])>=.10)&(np.ptp(score_array,axis=1)<.50)

def selective_metrics(model,frame,threshold):
    p=probabilities(model,frame[FEATURES]); mask=accepted_mask(p,frame[FEATURES].to_numpy(),threshold)
    predictions=np.array(LABELS)[p.argmax(axis=1)]; count=int(mask.sum())
    correct=int((predictions[mask]==frame.label.to_numpy()[mask]).sum())
    return {"total":len(frame),"accepted":count,"review":len(frame)-count,"coverage":count/len(frame),
            "accepted_correct":correct,"accepted_accuracy":correct/count if count else None,
            "threshold":threshold,"rule":"max probability >= threshold, top-two margin >= .10, component spread < .50"}

def improve():
    history=ARTIFACTS/"baseline_v1"; history.mkdir(exist_ok=True)
    for path in ARTIFACTS.iterdir():
        if path.is_file() and not (history/path.name).exists(): shutil.copy2(path,history/path.name)
    frame=pd.read_csv(DATA/"synthetic_participants.csv")
    original_hash=hashlib.sha256((DATA/"synthetic_participants.csv").read_bytes()).hexdigest()
    training=frame[frame.split=="train"];validation=frame[frame.split=="validation"]
    specs=[]
    for depth,leaf in [(4,8),(6,5),(8,4),(None,6)]:
        specs.append((f"Random Forest depth={depth} leaf={leaf}",RandomForestClassifier(n_estimators=400,max_depth=depth,min_samples_leaf=leaf,max_features=None,class_weight="balanced",random_state=SEED,n_jobs=1)))
    for leaf,depth in [(2,None),(5,None),(8,8),(12,6)]:
        specs.append((f"Extra Trees depth={depth} leaf={leaf}",ExtraTreesClassifier(n_estimators=400,max_depth=depth,min_samples_leaf=leaf,max_features=None,class_weight="balanced",random_state=SEED,n_jobs=1)))
    for leaves,l2 in [(5,5),(9,5),(15,10),(5,20)]:
        specs.append((f"Histogram Gradient Boosting leaves={leaves} l2={l2}",HistGradientBoostingClassifier(max_iter=180,learning_rate=.05,max_leaf_nodes=leaves,l2_regularization=l2,min_samples_leaf=20,early_stopping=False,random_state=SEED)))
    rows=[];models=[]
    with threadpool_limits(limits=1):
        for name,model in specs:
            cv=cross_val_score(model,training[FEATURES],training.label,scoring="f1_macro",cv=StratifiedKFold(5,shuffle=True,random_state=SEED),n_jobs=1)
            model.fit(training[FEATURES],training.label)
            valid=metrics(model,validation)
            row={"name":name,"cv_macro_f1":float(cv.mean()),"cv_std":float(cv.std()),"validation_macro_f1":valid["macro_f1"],"validation_accuracy":valid["accuracy"]}
            rows.append(row);models.append(model);print(json.dumps(row),flush=True)
    chosen=max(range(len(rows)),key=lambda i:(rows[i]["validation_macro_f1"],rows[i]["cv_macro_f1"]))
    model=models[chosen]; validation_p=probabilities(model,validation[FEATURES])
    gate_options=[selective_metrics(model,validation,float(t)) for t in np.arange(.40,.91,.025)]
    eligible=[option for option in gate_options if option["accepted"]>=20 and option["accepted_accuracy"]>=.80]
    gate=max(eligible,key=lambda option:option["coverage"]) if eligible else selective_metrics(model,validation,.60)
    threshold=gate["threshold"]
    protocol={"version":"2.0.0-pp1","created_utc":datetime.now(timezone.utc).isoformat(),
        "original_dataset_sha256":original_hash,"simulator_changed":False,"training_records":600,"validation_records":200,
        "old_test":"200 records already inspected; historical only, not used in this selection",
        "selection_rule":"highest validation macro F1, training CV mean tie break","selected":rows[chosen],"candidates":rows,
        "review_gate":{"validation_result":gate,"target_validation_accepted_accuracy":.80,"target_reached":bool(eligible),"minimum_validation_accepted":20,
            "note":"Accepted-case accuracy must always be shown alongside coverage and all-case accuracy. No accuracy guarantee."},
        "new_final_test":{"n":500,"seed":20261021,"noise":.85,"class_proportions":[.2]*5,"generated_after_selection":True}}
    (ARTIFACTS/"v2_frozen_protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    fresh=generate(500,20261021,noise=.85,prefix="FINAL2");fresh["split"]="fresh_final_test"
    fresh.to_csv(DATA/"synthetic_fresh_final_test.csv",index=False)
    stress=pd.read_csv(DATA/"synthetic_stress_participants.csv")
    serving_name=rows[chosen]["name"]
    comparison={serving_name:(model,FEATURES)}
    historical_bundle=joblib.load(history/"fusion_model.joblib")
    baseline=historical_bundle["model"]
    hard_scope=historical_bundle.get("scope","Historical harder-overlap synthetic experiment; not the active moderate demonstration")
    comparison["Original Random Forest baseline"]=(baseline,FEATURES)
    linear=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=SEED));linear.fit(training[FEATURES],training.label)
    comparison["Logistic regression fusion"]=(linear,FEATURES)
    for title,column in zip(["Motor only","Voice only","Handwriting only"],FEATURES):
        single=RandomForestClassifier(n_estimators=160,max_depth=5,min_samples_leaf=8,random_state=SEED,n_jobs=1)
        single.fit(training[[column]],training.label);comparison[title]=(single,[column])
    results={name:metrics(estimator,fresh,columns) for name,(estimator,columns) in comparison.items()}
    stress_results={name:metrics(estimator,stress,columns) for name,(estimator,columns) in comparison.items()}
    bundle={"model":model,"feature_columns":FEATURES,"labels":LABELS,"training_background":training[FEATURES].to_numpy(),
        "seed":SEED,"scope":hard_scope,"version":"2.0.0-pp1","model_name":serving_name,"review_threshold":threshold}
    joblib.dump(bundle,ARTIFACTS/"challenger_model.joblib")
    previous=json.loads((history/"evaluation.json").read_text())
    manifest={**previous,"version":"2.0.0-pp1-experiment","serving_model":serving_name,"selected_challenger":serving_name,"promoted_to_live_demo":False,"model_selection":protocol,
        "results":results,"stress_results":stress_results,"final_test_n":500,
        "final_test_scope":"Fresh synthetic sample from unchanged simulator after model selection; not real participant validation",
        "selective":selective_metrics(model,fresh,threshold),"validation_selective":gate,
        "data":{**previous["data"],"original_dataset_unchanged":original_hash==previous["data"]["sha256"],"fresh_final_test_n":500},
        "created_utc":datetime.now(timezone.utc).isoformat()}
    (ARTIFACTS/"improvement_evaluation.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    pd.DataFrame([{"model":name,**{k:v for k,v in result.items() if isinstance(v,(int,float))}} for name,result in results.items()]).to_csv(ARTIFACTS/"improvement_model_comparison.csv",index=False)
    p=probabilities(model,fresh[FEATURES]);export=fresh[["synthetic_id","label"]].copy();export["prediction"]=np.array(LABELS)[p.argmax(axis=1)]
    for i,label in enumerate(LABELS):export[f"p_{label}"]=p[:,i]
    export["accepted"]=accepted_mask(p,fresh[FEATURES].to_numpy(),threshold)
    export.to_csv(ARTIFACTS/"improvement_test_predictions.csv",index=False)
    print(json.dumps({"selected":serving_name,"fresh_test":results[serving_name],"selective":manifest["selective"],"original_dataset_unchanged":manifest["data"]["original_dataset_unchanged"]},indent=2),flush=True)
    plot_results(manifest)
    return manifest

def plot_results(manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results=manifest["results"];name=manifest["serving_model"]
    fig,ax=plt.subplots(figsize=(8,5),layout="constrained");matrix=results[name]["confusion_matrix"];image=ax.imshow(matrix,cmap="Blues")
    ax.set(xticks=range(5),yticks=range(5),xticklabels=LABELS,yticklabels=LABELS,xlabel="Predicted class",ylabel="Simulated class",title="Selected tree model · fresh synthetic test (n=500)")
    for i,row in enumerate(matrix):
        for j,value in enumerate(row):ax.text(j,i,str(value),ha="center",va="center",color="white" if value>50 else "#17334b")
    fig.colorbar(image,ax=ax);fig.savefig(ARTIFACTS/"improvement_confusion_matrix.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained");names=list(results);values=[results[k]["macro_f1"] for k in names]
    ax.barh(names,values,color=["#087e8b"]+["#9ab9c3"]*(len(names)-1));ax.set(xlim=(0,1),xlabel="Macro F1",title="Same fresh synthetic test for every model (n=500)");ax.invert_yaxis()
    for i,value in enumerate(values):ax.text(value+.01,i,f"{value:.3f}",va="center")
    fig.savefig(ARTIFACTS/"improvement_model_comparison.png",dpi=150);plt.close(fig)

if __name__=="__main__":improve()
