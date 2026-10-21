import json, hashlib
from collections import Counter
from functools import lru_cache
from typing import Annotated, Literal
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from .engine import FusionEngine, report_html
from .settings import ROOT, ARTIFACTS, DATA, SCOPE, FEATURES, LABELS,ACTIVE_DATASET,VERSION

app=FastAPI(title="Component 4 · Explainable Multimodal Fusion",version=VERSION,description=SCOPE)
Probability=Annotated[float,Field(ge=0,le=1,allow_inf_nan=False,strict=True)]
class FusionRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    motor_pd_probability:Probability|None=None
    voice_pd_probability:Probability|None=None
    handwriting_pd_probability:Probability|None=None
    session_id:str=Field(default="PP1-demo",max_length=100)
    include_lime:bool=False

@lru_cache
def get_engine():
    try: return FusionEngine()
    except FileNotFoundError: raise HTTPException(503,"Model not trained. Run python -m multimodal_fusion train first.")

def run(request):
    fields=request.model_dump(); session=fields.pop("session_id"); lime=fields.pop("include_lime")
    try: return get_engine().predict(fields,include_lime=lime,session_id=session)
    except ValueError as error: raise HTTPException(422,str(error))

@app.get("/api/health")
def health():
    return {"status":"ready" if (ARTIFACTS/"fusion_model.joblib").exists() else "not_trained","version":VERSION,"scope":SCOPE}

@app.post("/api/predict")
def predict(request:FusionRequest): return run(request)

@app.post("/api/report",response_class=HTMLResponse)
def report(request:FusionRequest): return report_html(run(request))

@app.get("/api/evaluation")
def evaluation():
    path=ARTIFACTS/"evaluation.json"
    if not path.exists(): raise HTTPException(503,"Train the model first")
    return json.loads(path.read_text(encoding="utf-8"))

@app.get("/api/demo-cases")
def cases():
    path=ARTIFACTS/"demo_cases.json"
    if not path.exists(): raise HTTPException(503,"Train the model first")
    return json.loads(path.read_text(encoding="utf-8"))

@app.get("/api/improvement-study")
def improvement_study():
    path=ARTIFACTS/"improvement_evaluation.json"
    if not path.exists():raise HTTPException(404,"No improvement study available")
    return json.loads(path.read_text(encoding="utf-8"))

@app.get('/api/preprocessing')
def preprocessing():
    path=ARTIFACTS/'preprocessing_report.json'
    if not path.exists():raise HTTPException(503,'Train the current pipeline first')
    return json.loads(path.read_text(encoding='utf-8'))

@app.get("/api/chart/{name}")
def chart(name:str):
    if name not in {"confusion_matrix.png","model_comparison.png","shap_global.png"}: raise HTTPException(404,"Unknown chart")
    path=ARTIFACTS/name
    if not path.exists(): raise HTTPException(404,"Chart not generated")
    return FileResponse(path)

DatasetSource=Literal["development","moderate_fresh","hard","fresh_test","stress"]
DATASETS={"development":ACTIVE_DATASET,"moderate_fresh":"synthetic_moderate_fresh_test.csv","hard":"synthetic_participants.csv","fresh_test":"synthetic_fresh_final_test.csv","stress":"synthetic_stress_participants.csv"}
SCENARIO_TEXT={'development':'Active moderate-overlap PP1 simulation: noise SD 0.50','moderate_fresh':'Separate moderate-overlap 500-record confirmation sample: noise SD 0.50','hard':'Retained original harder 1,000-record benchmark: noise SD 0.85','fresh_test':'Historical harder 500-record comparison sample: noise SD 0.85','stress':'Historical imbalanced stress sample: noise SD 1.05'}

@app.get("/api/dataset")
def dataset(source:DatasetSource="development",label:str="all",split:str="all",synthetic_id:str="",offset:int=Query(0,ge=0),limit:int=Query(25,ge=1,le=100)):
    import csv
    path=DATA/DATASETS[source]
    if not path.exists():raise HTTPException(404,"Dataset not generated")
    if label not in ["all",*LABELS]:raise HTTPException(422,"Unknown class")
    with path.open(newline="",encoding="utf-8") as stream:rows=list(csv.DictReader(stream))
    for row in rows:
        for key in FEATURES:row[key]=float(row[key])
        row.setdefault("split","stress_test" if source=="stress" else "unspecified")
    all_splits=sorted({row["split"] for row in rows})
    if split not in ["all",*all_splits]:raise HTTPException(422,"Unknown split for selected dataset")
    filtered=[row for row in rows if (label=="all" or row["label"]==label) and (split=="all" or row["split"]==split) and synthetic_id.lower() in row["synthetic_id"].lower()]
    metadata={"source":source,"filename":path.name,"total_rows":len(rows),"filtered_rows":len(filtered),
        "class_counts":dict(Counter(row["label"] for row in rows)),"split_counts":dict(Counter(row["split"] for row in rows)),
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"columns":list(rows[0]) if rows else [],
        "synthetic":True,"scope":SCENARIO_TEXT[source]+'. Assumed synthetic profiles, not clinical measurements.',"scenario_description":SCENARIO_TEXT[source],"numeric_features":FEATURES,
        "label_origin":"Assigned simulated class; not a clinician diagnosis",
        "score_origin":"Assumed overlapping class profiles; no uploaded patient measurements copied",
        "note":"Synthetic IDs represent simulated records, not real patients. Same-dataset test performance only measures these assumptions."}
    return {"metadata":metadata,"offset":offset,"limit":limit,"rows":filtered[offset:offset+limit]}

@app.get("/api/dataset/download")
def dataset_download(source:DatasetSource="development"):
    path=DATA/DATASETS[source]
    if not path.exists():raise HTTPException(404,"Dataset not generated")
    return FileResponse(path,media_type="text/csv",filename=path.name)

@app.get("/dataset")
def dataset_page():return FileResponse(ROOT/"web"/"dataset.html")

app.mount("/static",StaticFiles(directory=ROOT/"web"),name="static")
@app.get("/evaluation")
def evaluation_page(): return FileResponse(ROOT/"web"/"evaluation.html")

@app.get("/")
def home(): return FileResponse(ROOT/"web"/"index.html")
