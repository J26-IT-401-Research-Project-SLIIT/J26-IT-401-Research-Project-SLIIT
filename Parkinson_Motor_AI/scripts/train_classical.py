#!/usr/bin/env python3
"""Fit RF/SVM/XGBoost on same training/validation subjects; evaluate selected on test."""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from pads_motor.evaluate import metrics_subject, aggregate_by_participant

META_COLS={'participant_id','label','condition','task','wrist','segment','source','split'}


def train(out):
    out=Path(out)
    df=pd.read_csv(out/'features.csv',dtype={'participant_id':str})
    split=pd.read_csv(out/'participant_splits.csv',dtype={'participant_id':str})
    df=df.merge(split[['participant_id','split']],on='participant_id',how='left',validate='many_to_one')
    if df.split.isna().any():raise ValueError('Some participants missing splits')
    nums=[c for c in df.columns if c not in META_COLS]
    cats=['task','wrist']
    pre=ColumnTransformer([
        ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]), nums),
        ('categorical', OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats),
    ])
    train=df[df.split=='train']; val=df[df.split=='val']; test=df[df.split=='test']
    models={
       'RandomForest':RandomForestClassifier(n_estimators=200,min_samples_leaf=2,
                                           class_weight='balanced',n_jobs=-1,random_state=42),
       'SVM':SVC(kernel='rbf',C=1.0,class_weight='balanced',probability=True,random_state=42),
       'XGBoost':XGBClassifier(n_estimators=180,max_depth=3,learning_rate=.05,
                                subsample=.85,colsample_bytree=.85,eval_metric='logloss',
                                n_jobs=4,random_state=42),
    }
    summaries=[]; fitted={}
    for name, model in models.items():
        pipe=Pipeline([('prepare',pre),('classifier',model)])
        pipe.fit(train,train.label)
        probs=pipe.predict_proba(val)[:,list(pipe.named_steps['classifier'].classes_).index(1)]
        scores=metrics_subject(val,probs)
        scores.update(model=name,split='validation')
        summaries.append(scores)
        fitted[name]=pipe
        aggregate_by_participant(val,probs).to_csv(out/f'{name}_val_subject_predictions.csv',index=False)
        print(f'{name}: val balanced accuracy={scores["balanced_accuracy"]:.3f}, F1={scores["f1_pd"]:.3f}')
    summary=pd.DataFrame(summaries)
    summary.to_csv(out/'classical_validation_metrics.csv',index=False)
    # TEST IS RESERVED FOR FINAL SELECTED MODEL; Only classic models participate in this package.
    # All three estimators are saved for later selection, no test data evaluated here.
    for name,model in fitted.items():
        joblib.dump(model,out/f'{name}_pipeline.joblib')
    print('Models and VALIDATION metrics written. Run scripts.select_best to choose among RF/SVM/XGBoost on validation and evaluate test once.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs');a=p.parse_args();train(a.out)
