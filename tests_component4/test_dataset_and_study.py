import csv, io, json, hashlib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from multimodal_fusion.api import app
from multimodal_fusion.settings import DATA,ARTIFACTS,FEATURES
from multimodal_fusion.improve import accepted_mask

client=TestClient(app)

def test_dataset_counts_provenance_and_filters():
    response=client.get('/api/dataset').json()
    assert response['metadata']['total_rows']==1000
    assert len(response['rows'])==25
    assert response['metadata']['class_counts']==dict.fromkeys(['PD','ET','CA','AP','DLB'],200)
    assert response['metadata']['synthetic'] is True
    filtered=client.get('/api/dataset',params={'label':'PD','split':'train','limit':100}).json()
    assert filtered['metadata']['filtered_rows']==120
    assert all(row['label']=='PD' and row['split']=='train' for row in filtered['rows'])

def test_dataset_pagination_download_and_record():
    first=client.get('/api/dataset').json();second=client.get('/api/dataset?offset=25').json()
    assert {r['synthetic_id'] for r in first['rows']}.isdisjoint({r['synthetic_id'] for r in second['rows']})
    record=first['rows'][0]
    found=client.get('/api/dataset',params={'synthetic_id':record['synthetic_id']}).json()
    assert found['rows']==[record]
    downloaded=client.get('/api/dataset/download')
    assert len(list(csv.DictReader(io.StringIO(downloaded.text))))==1000
    assert hashlib.sha256(downloaded.content).hexdigest()==first['metadata']['sha256']

@pytest.mark.parametrize('source',['fresh_test','stress'])
def test_separate_evaluation_datasets(source):
    response=client.get('/api/dataset',params={'source':source}).json()
    assert response['metadata']['total_rows']==500
    assert response['metadata']['synthetic'] is True
    assert client.get('/api/dataset/download',params={'source':source}).status_code==200

def test_invalid_dataset_queries_rejected():
    for query in ['source=../../README.md','label=healthy','offset=-1','limit=1000','split=invalid']:
        assert client.get('/api/dataset?'+query).status_code==422
    assert client.get('/dataset').status_code==200

def test_original_hard_data_and_archived_model_preserved():
    original=json.loads((ARTIFACTS/'baseline_v1/evaluation.json').read_text())
    assert hashlib.sha256((DATA/'synthetic_participants.csv').read_bytes()).hexdigest()==original['data']['sha256']
    assert (ARTIFACTS/'baseline_v1/fusion_model.joblib').is_file()
    active=json.loads((ARTIFACTS/'evaluation.json').read_text())
    assert active['hard_benchmark']['original_file_unchanged']
    assert active['hard_benchmark']['source_sha256']==original['data']['sha256']

def test_study_metrics_and_selective_coverage_are_recomputable():
    from sklearn.metrics import accuracy_score,f1_score
    study=client.get('/api/improvement-study').json()
    predictions=pd.read_csv(ARTIFACTS/'improvement_test_predictions.csv')
    result=study['results'][study['selected_challenger']]
    assert study['promoted_to_live_demo'] is False
    assert result['accuracy']==pytest.approx(accuracy_score(predictions.label,predictions.prediction))
    assert result['macro_f1']==pytest.approx(f1_score(predictions.label,predictions.prediction,average='macro'))
    p=predictions[['p_'+c for c in ['PD','ET','CA','AP','DLB']]].to_numpy()
    fresh=pd.read_csv(DATA/'synthetic_fresh_final_test.csv')
    mask=accepted_mask(p,fresh[FEATURES].to_numpy(),study['selective']['threshold'])
    assert mask.sum()==study['selective']['accepted']
    assert mask.mean()==study['selective']['coverage']
    assert accuracy_score(predictions.label[mask],predictions.prediction[mask])==pytest.approx(study['selective']['accepted_accuracy'])
