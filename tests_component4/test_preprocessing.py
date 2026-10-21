import json
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn.pipeline import Pipeline
from fastapi.testclient import TestClient
from multimodal_fusion.preprocessing import ScorePreprocessor,inspect_dataset
from multimodal_fusion.settings import DATA,ARTIFACTS,ACTIVE_DATASET,FEATURES
from multimodal_fusion.engine import FusionEngine
from multimodal_fusion.api import app

def test_preprocessing_reorders_and_preserves_scores():
    frame=pd.DataFrame([[.2,.3,.4]],columns=FEATURES)
    transformer=ScorePreprocessor().fit(frame)
    actual=transformer.transform(frame[FEATURES[::-1]])
    pd.testing.assert_frame_equal(actual,frame)
    assert list(transformer.get_feature_names_out())==FEATURES

@pytest.mark.parametrize('value',[None,np.nan,np.inf,-.1,1.01,True])
def test_preprocessing_rejects_invalid_not_imputes_or_clips(value):
    frame=pd.DataFrame([[value,.4,.5]],columns=FEATURES)
    with pytest.raises(ValueError):ScorePreprocessor().fit(frame)

def test_preprocessing_rejects_metadata_and_duplicate_ids():
    frame=pd.DataFrame([[.2,.3,.4]],columns=FEATURES)
    with pytest.raises(ValueError):ScorePreprocessor().fit(frame.assign(label='PD'))
    cohort=pd.read_csv(DATA/ACTIVE_DATASET)
    cohort.loc[1,'synthetic_id']=cohort.loc[0,'synthetic_id']
    with pytest.raises(ValueError):inspect_dataset(cohort)

def test_fitted_pipeline_matches_engine_and_training_partition():
    engine=FusionEngine();frame=pd.read_csv(DATA/ACTIVE_DATASET);training=frame[frame.split=='train']
    pre=engine.pipeline.named_steps['score_preprocess']
    assert pre.fit_rows_==600
    assert pre.training_min_==training[FEATURES].min().to_dict()
    expected=engine.pipeline.predict_proba(training[FEATURES].iloc[:1])[0]
    response=engine.predict(training[FEATURES].iloc[0].to_dict())
    for label,p in zip(engine.model.classes_,expected):assert response['likeness_profile'][label]==pytest.approx(p)
    assert response['scenario']=='moderate_overlap'
    assert response['version'].startswith('3.')

def test_cv_fits_preprocessing_inside_each_fold():
    frame=pd.read_csv(DATA/ACTIVE_DATASET);training=frame[frame.split=='train']
    pipe=Pipeline([('score_preprocess',ScorePreprocessor()),('classifier',RandomForestClassifier(n_estimators=2,random_state=1))])
    result=cross_validate(pipe,training[FEATURES],training.label,cv=StratifiedKFold(3),return_estimator=True)
    assert all(estimator.named_steps['score_preprocess'].fit_rows_==400 for estimator in result['estimator'])

def test_scenario_and_preprocessing_evidence_api():
    client=TestClient(app)
    pre=client.get('/api/preprocessing').json();assert pre['fit_rows']==600
    evidence=client.get('/api/evaluation').json()
    assert evidence['scenario']=='moderate_overlap'
    assert evidence['data']['logit_noise_sd']==.5
    assert evidence['hard_benchmark']['noise_sd']==.85
    assert evidence['fresh_test_n']==500
    assert not any(evidence['leakage_checks']['training_feature_overlap_with_evaluation'].values())
    assert client.get('/api/dataset?source=hard').json()['metadata']['total_rows']==1000
    assert client.get('/api/dataset?source=moderate_fresh').json()['metadata']['total_rows']==500
