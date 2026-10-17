#!/usr/bin/env python3
"""Select best among RF, SVM and XGBoost on validation participants; evaluate test only once."""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from pads_motor.evaluate import metrics_subject, aggregate_by_participant


def main(out):
    out=Path(out)
    results=pd.read_csv(out/'classical_validation_metrics.csv')
    if not {'RandomForest','SVM','XGBoost'}.issubset(set(results.model)):
        raise ValueError('Train all three models first using scripts.train_classical')
    # validation selection only, predefined metric/tiebreak
    ranked=results.sort_values(['balanced_accuracy','f1_pd','model'], ascending=[False,False,True])
    ranked.to_csv(out/'model_validation_ranking.csv',index=False)
    best=str(ranked.iloc[0].model)
    print('Best model selected on validation participants:',best)
    features=pd.read_csv(out/'features.csv',dtype={'participant_id':str})
    split=pd.read_csv(out/'participant_splits.csv',dtype={'participant_id':str})
    features=features.merge(split[['participant_id','split']],on='participant_id',validate='many_to_one')
    test=features.loc[features.split=='test'].copy()
    if test.empty: raise ValueError('No test participants found')
    model=joblib.load(out/f'{best}_pipeline.joblib')
    classes=list(model.named_steps['classifier'].classes_)
    proba=model.predict_proba(test)[:,classes.index(1)]
    score=metrics_subject(test,proba)
    score.update(selected_model=best,dataset='PADS',target='PD_vs_Healthy',split='held_out_test',
                 note='Public-data baseline only, not clinically validated device severity scoring')
    (out/'selected_test_metrics.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
    aggregate_by_participant(test,proba).to_csv(out/'selected_test_subject_predictions.csv',index=False)
    print(json.dumps(score,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='outputs')
    a=parser.parse_args()
    main(a.out)
