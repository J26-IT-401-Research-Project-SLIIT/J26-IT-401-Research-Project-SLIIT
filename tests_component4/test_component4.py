import hashlib, itertools, json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from multimodal_fusion.api import app
from multimodal_fusion.engine import FusionEngine, report_html
from multimodal_fusion.settings import ARTIFACTS, DATA, FEATURES, LABELS,ACTIVE_DATASET
from multimodal_fusion.synthetic import generate

@pytest.fixture(scope="module")
def engine():return FusionEngine()

@pytest.fixture(scope="module")
def client():return TestClient(app)

def test_synthetic_generator_reproducible():
    first=generate();second=generate()
    pd.testing.assert_frame_equal(first,second)
    assert first.label.value_counts().to_dict()==dict.fromkeys(LABELS,200)
    assert first.synthetic_id.nunique()==1000
    assert first[FEATURES].ge(0).all().all() and first[FEATURES].le(1).all().all()

def test_participant_splits_and_background(engine):
    frame=pd.read_csv(DATA/ACTIVE_DATASET)
    assert frame.synthetic_id.is_unique
    assert frame.split.value_counts().to_dict()=={"train":600,"test":200,"validation":200}
    for _,part in frame.groupby("split"):
        assert part.label.nunique()==5
    # Saved SHAP/LIME background contains exactly training inputs, no holdout rows.
    actual={tuple(r) for r in engine.bundle["training_background"]}
    expected={tuple(r) for r in frame.loc[frame.split=="train",FEATURES].to_numpy()}
    assert actual==expected
    assert list(engine.model.feature_names_in_)==FEATURES

@pytest.mark.parametrize("scores",[[0.,0.,0.],[1.,1.,1.],[.7,.4,.6],[.9,.2,.5]])
def test_probabilities_and_shap_additivity(engine,scores):
    result=engine.predict(dict(zip(FEATURES,scores)))
    assert list(result["likeness_profile"])==LABELS
    assert sum(result["likeness_profile"].values())==pytest.approx(1)
    assert all(0<=p<=1 for p in result["likeness_profile"].values())
    assert result["shap"]["additivity_error"]<1e-6
    assert result["clinical_validation"] is False

@pytest.mark.parametrize("mask",[mask for mask in itertools.product([False,True],repeat=3) if any(mask)])
def test_missing_inputs_never_fabricated(engine,mask):
    scores={f:None if missing else .7 for f,missing in zip(FEATURES,mask)}
    result=engine.predict(scores)
    assert result["status"]=="incomplete" and result["likeness_profile"] is None
    assert result["predicted_class"] is None

@pytest.mark.parametrize("value",[-.1,1.1,float("nan"),float("inf"),True,"0.5"])
def test_invalid_inputs_rejected(engine,value):
    with pytest.raises(ValueError):engine.predict(dict(zip(FEATURES,[value,.5,.6])))

def test_lime_repeatability_and_class_alignment(engine):
    request=dict(zip(FEATURES,[.73,.64,.68]))
    a=engine.predict(request,include_lime=True);b=engine.predict(request,include_lime=True)
    assert a["lime"]["conditions"]==b["lime"]["conditions"]
    assert a["lime"]["explained_class"]==a["predicted_class"]
    assert np.isfinite(a["lime"]["local_surrogate_r2"])

def test_report_escapes_warning_text(engine):
    result=engine.predict(dict.fromkeys(FEATURES,.6),session_id='<script>alert(1)</script>')
    result['warnings']=['<script>alert(1)</script>']
    report=report_html(result)
    assert 'Session:' not in report and 'Model SHA-256:' not in report
    assert '<script>' not in report and '&lt;script&gt;' in report
    assert 'Synthetic data demonstration' in report

def test_saved_metrics_independently_recomputed():
    from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
    frame=pd.read_csv(ARTIFACTS/"test_predictions.csv")
    report=json.loads((ARTIFACTS/"evaluation.json").read_text())
    result=report["results"]["Random Forest fusion"]
    assert len(frame)==200
    assert result["accuracy"]==pytest.approx(accuracy_score(frame.label,frame.prediction))
    assert result["macro_f1"]==pytest.approx(f1_score(frame.label,frame.prediction,average="macro"))
    assert result["confusion_matrix"]==confusion_matrix(frame.label,frame.prediction,labels=LABELS).tolist()
    assert hashlib.sha256((DATA/ACTIVE_DATASET).read_bytes()).hexdigest()==report["data"]["sha256"]

def test_api_complete_and_lime(client):
    response=client.post('/api/predict',json={**dict.fromkeys(FEATURES,.7),"include_lime":True})
    assert response.status_code==200
    assert response.json()['lime']['conditions']

def test_api_validation_and_incomplete(client):
    assert client.post('/api/predict',json={"motor_pd_probability":2}).status_code==422
    assert client.post('/api/predict',json={"motor_pd_probability":True}).status_code==422
    assert client.post('/api/predict',json={"unknown":.5}).status_code==422
    assert client.post('/api/predict',json={"motor_pd_probability":.5}).json()['status']=='incomplete'

def test_api_evidence_and_report(client):
    assert client.get('/api/health').json()['status']=='ready'
    assert len(client.get('/api/demo-cases').json())==8
    assert client.get('/api/evaluation').json()['data']['n']==1000
    assert client.get('/api/chart/confusion_matrix.png').headers['content-type']=='image/png'
    assert client.get('/api/chart/fusion_model.joblib').status_code==404
    assert client.post('/api/report',json=dict.fromkeys(FEATURES,.7)).status_code==200
    assert client.get('/').status_code==200
